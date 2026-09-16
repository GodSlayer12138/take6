"""Checks for the research harness; run with ntw-ai Python."""

import re
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from reproduce_eth_dirv import (
    ROOT, ParallelSelfPlay, SearchAgent, Table, bootstrap_interval, discounted_returns,
    normalize, play_game, puct_values, train,
)


def upstream_environment():
    # Run the original local rules with ONLY gym metadata / NumPy alias shims.
    # Do not import its package __init__: that pulls in unused gym/numba/multi_elo.
    path = ROOT / "external/rl-6-nimmt/rl_6_nimmt/env.py"
    source = path.read_text(encoding="utf-8")
    source = source.replace("from gym import Env", "").replace("from gym.spaces import Discrete, Box", "")
    source = re.sub(r"np\.int\b", "int", source)
    source = re.sub(r"np\.float\b", "float", source)
    namespace = {"Env": object, "Discrete": lambda n: SimpleNamespace(n=n), "Box": lambda **kw: SimpleNamespace(**kw)}
    exec(compile(source, str(path), "exec"), namespace)
    return namespace["SechsNimmtEnv"]


class ReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_rules_and_observations_match_original_for_500_turns(self):
        original = upstream_environment()
        rng = np.random.default_rng(9823)
        for seed in range(50):
            table = Table.deal(seed)
            upstream = original(4, verbose=False)
            upstream.reset_to([row.copy() for row in table.rows], [hand.copy() for hand in table.hands])
            for turn in range(10):
                np.testing.assert_array_equal(table.states(), np.array(upstream._create_states()[0]))
                actions = [int(rng.choice(hand)) for hand in table.hands]
                rewards = table.step(actions)
                _, upstream_rewards, done, _ = upstream.step(actions)
                np.testing.assert_array_equal(rewards, upstream_rewards)
                self.assertEqual(table.rows, upstream._board)
                self.assertEqual(done, turn == 9)

    def test_forced_row_tie_uses_upstream_first_index(self):
        table = Table([[9, 11], [39, 41], [10], [21]], [[0], [5]])
        # First two rows cost 4 each, remaining rows cost 5 each.
        rewards = table.step([0, 5])
        self.assertEqual(rewards[0], -4)
        self.assertEqual(table.rows[0], [0, 5])

    def test_normalization_matches_original(self):
        path = ROOT / "external/rl-6-nimmt/rl_6_nimmt/utils/preprocessing.py"
        namespace = {}
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
        original = namespace["SechsNimmtStateNormalization"]()
        states = torch.tensor(Table.deal(42).states())
        torch.testing.assert_close(normalize(states), original(states))

    def test_returns_include_current_and_final_capture(self):
        np.testing.assert_allclose(discounted_returns([-2, -3], .99), [-4.97, -3])

    def test_puct_constant_returns_are_finite(self):
        result = puct_values(np.array([10, 0]), np.zeros(2), [0.] * 10, np.array([.5, .5]))
        self.assertTrue(np.isfinite(result).all())
        self.assertGreater(result[1], result[0])

    def test_hidden_hands_do_not_enter_observation(self):
        table = Table.deal(81)
        before = table.states()[0].copy()
        table.hands[1], table.hands[2] = table.hands[2], table.hands[1]
        np.testing.assert_array_equal(before, table.states()[0])

    def test_training_updates_both_agents_and_evaluation_is_frozen(self):
        for kind in ("alpha", "dirv"):
            with self.subTest(kind=kind):
                agent = SearchAgent(kind, 42, simulations=2)
                before = {k: v.clone() for k, v in agent.model.state_dict().items()}
                first, _ = play_game([agent, None, None, None], 900, 901)
                repeated, _ = play_game([agent, None, None, None], 900, 901)
                np.testing.assert_array_equal(first, repeated)
                self.assertTrue(all(torch.equal(before[k], v) for k, v in agent.model.state_dict().items()))
                _, losses = play_game([agent, None, None, None], 900, 901, training=True)
                self.assertTrue(all(np.isfinite(list(losses[0].values()))))
                self.assertFalse(all(torch.equal(before[k], v) for k, v in agent.model.state_dict().items()))
                if kind == "dirv":
                    self.assertFalse(torch.equal(before["value.weight"], agent.model.state_dict()["value.weight"]))

    def test_numpy_inference_matches_torch_before_and_after_learning(self):
        table = Table.deal(501)
        for kind in ("alpha", "dirv"):
            agent = SearchAgent(kind, 41, simulations=2)
            for _ in range(2):
                expected_policy, expected_value = agent.policy_values(table.states(), table.hands)
                actual_policy, actual_value = agent.predict(table.states(), table.hands)
                np.testing.assert_allclose(actual_policy, expected_policy.detach().numpy(), rtol=2e-6, atol=1e-7)
                if expected_value is not None:
                    np.testing.assert_allclose(actual_value, expected_value.detach().numpy(), rtol=2e-6, atol=1e-7)
                play_game([agent, None, None, None], 503, 504, training=True)

    def test_parallel_players_preserve_serial_game_and_model_updates(self):
        for kind in ("alpha", "dirv"):
            config = SimpleNamespace(kind=kind, seed=801, simulations=2, hidden=100, learning_rate=.001, gamma=.99, device="cpu")
            serial = [SearchAgent(kind, config.seed + seat, simulations=2) for seat in range(4)]
            expected_scores, _ = play_game(serial, 811, 812, training=True)
            parallel = ParallelSelfPlay(config)
            try:
                actual_scores, _ = parallel.game(811, 812)
                np.testing.assert_array_equal(actual_scores, expected_scores)
                saved = parallel.checkpoint()
                for agent, state in zip(serial, saved):
                    for key, tensor in agent.model.state_dict().items():
                        torch.testing.assert_close(tensor, state["model"][key], rtol=0, atol=0)
            finally:
                parallel.close()

    def test_bootstrap_requires_independent_blocks(self):
        self.assertIsNone(bootstrap_interval([.2], 1))
        self.assertEqual(bootstrap_interval([.25, .25, .25], 1, 100), [.25, .25])

    def test_checkpoint_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            complete = str(Path(temporary) / "complete")
            resumed = str(Path(temporary) / "resumed")
            config = dict(command="train", kind="dirv", seed=861, simulations=2, hidden=100,
                          learning_rate=.001, gamma=.99, device="cpu", workers=1, threads=1, save_every=1)
            with contextlib.redirect_stdout(io.StringIO()):
                train(SimpleNamespace(**config, output=complete, games=2, resume=False))
                train(SimpleNamespace(**config, output=resumed, games=1, resume=False))
                train(SimpleNamespace(**config, output=resumed, games=2, resume=True))
            reference = torch.load(Path(complete) / "checkpoint.pt", weights_only=True)
            result = torch.load(Path(resumed) / "checkpoint.pt", weights_only=True)
            self.assertEqual(result["games"], 2)
            for left, right in zip(reference["agents"], result["agents"]):
                for key in left["model"]:
                    torch.testing.assert_close(left["model"][key], right["model"][key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
