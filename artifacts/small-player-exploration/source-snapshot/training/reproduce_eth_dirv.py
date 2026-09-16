"""Independent, assumption-explicit reproduction of ETH GA-2021-02, section 5.6.

Run with D:/Programs/miniconda3/envs/ntw-ai/python.exe. Only numpy/torch required.
This does not contain the authors' trained models and is not an exact reproduction.
The PUCT baseline follows external/rl-6-nimmt/rl_6_nimmt/agents/mcts.py.
Upstream is MIT, copyright (c) 2020 Johann Brehmer and Marcel Gutsche;
see external/rl-6-nimmt/LICENSE.md. The vendored source is kept unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

# Tiny matrix products should not start a BLAS thread pool per search worker.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
REPORT_URL = "https://pub.tik.ee.ethz.ch/students/2021-HS/GA-2021-02.pdf"
FORMAT = "ntw-eth-dirv-reimplementation-v1"


def bullheads(card):
    value = int(card) + 1
    return 7 if value == 55 else 5 if value % 11 == 0 else 3 if value % 10 == 0 else 2 if value % 10 == 5 else 1


POINTS = np.array([bullheads(card) for card in range(104)])


class Table:
    """Zero-based, 104-card rules; equal-cost forced rows choose first index (upstream)."""

    def __init__(self, rows, hands):
        self.rows = [list(map(int, row)) for row in rows]
        self.hands = [list(map(int, hand)) for hand in hands]

    @classmethod
    def deal(cls, seed, players=4):
        cards = np.random.default_rng(seed).permutation(104).tolist()
        hands = [sorted(cards[p * 10:(p + 1) * 10]) for p in range(players)]
        return cls([[cards[-1 - row]] for row in range(4)], hands)

    def states(self):
        board = np.full((4, 6), -1.0, dtype=np.float32)
        for i, row in enumerate(self.rows):
            board[i, :len(row)] = row
        public = np.concatenate((
            [len(self.hands)], [len(row) for row in self.rows],
            [row[-1] for row in self.rows],
            [POINTS[row].sum() for row in self.rows], board.ravel(),
        )).astype(np.float32)
        return np.stack([np.concatenate((hand + [-1] * (10 - len(hand)), public)) for hand in self.hands]).astype(np.float32)

    def step(self, actions):
        if len(actions) != len(self.hands) or len(set(actions)) != len(actions):
            raise ValueError("One distinct legal card is required per player")
        if any(card not in hand for card, hand in zip(actions, self.hands)):
            raise ValueError("Illegal move")
        rewards = np.zeros(len(self.hands), dtype=np.float32)
        for player in sorted(range(len(actions)), key=lambda p: actions[p]):
            card = int(actions[player])
            eligible = [i for i, row in enumerate(self.rows) if row[-1] < card]
            forced = not eligible
            row_id = max(eligible, key=lambda i: self.rows[i][-1]) if eligible else min(range(4), key=lambda i: int(POINTS[self.rows[i]].sum()))
            if forced or len(self.rows[row_id]) == 5:
                rewards[player] = -POINTS[self.rows[row_id]].sum()
                self.rows[row_id] = [card]
            else:
                self.rows[row_id].append(card)
            self.hands[player].remove(card)
        return rewards


def normalize(states):
    """Exactly the 47-feature scaling used by the upstream state preprocessor."""
    result = states.clone()
    result[..., :10] = 2 * result[..., :10] / 103 - 1
    result[..., 10] = 2 * result[..., 10] / 6 - 1
    result[..., 11:15] = 2 * (result[..., 11:15] - 1) / 4 - 1
    result[..., 15:19] = 2 * result[..., 15:19] / 103 - 1
    result[..., 19:23] = 2 * (result[..., 19:23] - 1) / 9 - 1
    result[..., 23:] = 2 * result[..., 23:] / 103 - 1
    return result


class Network(nn.Module):
    def __init__(self, kind, hidden=100):
        super().__init__()
        self.kind = kind
        self.body = nn.Sequential(nn.Linear(48 if kind == "alpha" else 47, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.policy = nn.Linear(hidden, 1 if kind == "alpha" else 104)
        self.value = nn.Linear(hidden, 1) if kind == "dirv" else None

    def forward(self, states):
        latent = self.body(states)
        return self.policy(latent), self.value(latent).squeeze(-1) if self.value is not None else None


def discounted_returns(rewards, gamma):
    result = np.empty(len(rewards), dtype=np.float32)
    value = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        value = float(rewards[index]) + gamma * value
        result[index] = value
    return result


def puct_values(counts, totals, outcomes, priors, exploration=2.0):
    if len(outcomes) < 10:
        lower, upper, default = -10.0, 0.0, -5.0
    else:
        lower, upper, default = min(outcomes), max(outcomes), float(np.median(outcomes))
    means = np.divide(totals, counts, out=np.full_like(totals, default), where=counts > 0)
    # Upstream divides by zero if all returns coincide; use zero exploitation then.
    q = np.zeros_like(means) if upper == lower else np.clip((means - lower) / (upper - lower), 0, 1)
    return q + exploration * priors * math.sqrt(float(counts.sum()) + 1e-9) / (1 + counts)


class SearchAgent:
    def __init__(self, kind, seed, simulations=50, hidden=100, learning_rate=0.001, gamma=0.99, device="cpu"):
        self.kind = kind
        self.device = torch.device(device)
        self.gamma = gamma
        self.simulations = simulations
        self.hidden = hidden
        self.learning_rate = learning_rate
        # Initialization and simulation randomness never alter the deal RNG.
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.model = Network(kind, hidden).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.rng = np.random.default_rng(seed)
        self.available = set(range(104))
        self.decision_seconds = []

    def predict(self, states, hands):
        """Inference only. NumPy avoids eager-Torch overhead on tiny CPU batches.

        CPU weight views refer to the current Torch parameters, including updates.
        Training always uses policy_values() and autograd. CUDA uses Torch directly.
        """
        if self.device.type != "cpu":
            with torch.no_grad():
                probs, values = self.policy_values(states, hands)
            return probs.cpu().numpy(), None if values is None else values.cpu().numpy()
        x = np.asarray(states, dtype=np.float32).copy()
        x[:, :10] = 2 * x[:, :10] / 103 - 1
        x[:, 10] = 2 * x[:, 10] / 6 - 1
        x[:, 11:15] = 2 * (x[:, 11:15] - 1) / 4 - 1
        x[:, 15:19] = 2 * x[:, 15:19] / 103 - 1
        x[:, 19:23] = 2 * (x[:, 19:23] - 1) / 9 - 1
        x[:, 23:] = 2 * x[:, 23:] / 103 - 1
        cards = np.asarray(hands, dtype=np.int64)
        batch, actions = cards.shape
        if self.kind == "alpha":
            expanded = np.broadcast_to(x[:, None, :], (batch, actions, 47))
            card_feature = (2 * cards.astype(np.float32) / 103 - 1)[..., None]
            x = np.concatenate((card_feature, expanded), axis=2).reshape(-1, 48)
        for layer in (self.model.body[0], self.model.body[2]):
            x = np.maximum(x @ layer.weight.detach().numpy().T + layer.bias.detach().numpy(), 0)
        policy = self.model.policy
        logits = x @ policy.weight.detach().numpy().T + policy.bias.detach().numpy()
        if self.kind == "alpha":
            logits, values = logits.reshape(batch, actions), None
        else:
            logits = np.take_along_axis(logits, cards, axis=1)
            value = self.model.value
            values = (x @ value.weight.detach().numpy().T + value.bias.detach().numpy()).ravel()
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs, values

    def begin_game(self, seed):
        self.rng = np.random.default_rng(seed)
        self.available = set(range(104))

    def policy_values(self, states, hands):
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        normalized = normalize(states)
        cards = torch.tensor(hands, device=self.device, dtype=torch.long)
        if self.kind == "alpha":
            batch, actions = cards.shape
            expanded = normalized[:, None, :].expand(-1, actions, -1)
            inputs = torch.cat(((2 * cards.float() / 103 - 1)[..., None], expanded), dim=-1)
            logits, _ = self.model(inputs.reshape(batch * actions, 48))
            logits, values = logits.reshape(batch, actions), None
        else:
            logits, values = self.model(normalized)
            logits = logits.gather(1, cards)
        return torch.softmax(logits, dim=1), values

    def select(self, state, hand):
        """Only own state/hand cross this boundary; real opposing hands never do."""
        started = time.perf_counter()
        board = [[int(card) for card in row if card >= 0] for row in state[23:].reshape(4, 6)]
        self.available.difference_update(hand)
        self.available.difference_update(card for row in board for card in row)
        if len(hand) == 1:
            result = hand[0]
        else:
            with torch.no_grad():
                result = self.search(state, hand, board)
        self.decision_seconds.append(time.perf_counter() - started)
        return result

    def search(self, state, hand, board):
        n = len(hand)
        players = int(state[10])
        priors = self.predict(state[None, :], [hand])[0][0]
        counts = np.zeros(n, dtype=np.int64)
        totals = np.zeros(n, dtype=np.float64)
        outcomes = []
        unseen = np.array(sorted(self.available))
        for _ in range(min(self.simulations, 10 * math.factorial(n))):
            action_id = int(np.argmax(puct_values(counts, totals, outcomes, priors)))
            sampled = self.rng.choice(unseen, (players - 1) * n, replace=False).reshape(players - 1, n)
            table = Table(board, [hand] + [sorted(row.tolist()) for row in sampled])
            probs, _ = self.predict(table.states(), table.hands)
            actions = [hand[action_id]] + [table.hands[p][int(self.rng.choice(n, p=probs[p] / probs[p].sum()))] for p in range(1, players)]
            outcome = float(table.step(actions)[0])
            if self.kind == "dirv":
                if table.hands[0]:
                    _, values = self.predict(table.states()[0:1], [table.hands[0]])
                    outcome += self.gamma * float(values[0])
            else:
                # Alpha0.5: the same actor guides all players to the end of the game.
                while table.hands[0]:
                    probs, _ = self.predict(table.states(), table.hands)
                    actions = [hand_[int(self.rng.choice(len(hand_), p=prob / prob.sum()))] for hand_, prob in zip(table.hands, probs)]
                    outcome += float(table.step(actions)[0])
            counts[action_id] += 1
            totals[action_id] += outcome
            outcomes.append(outcome)
        means = np.divide(totals, counts, out=np.full(n, -np.inf), where=counts > 0)
        return hand[int(np.argmax(means))]

    def learn_episode(self, trajectory):
        policy_terms, predictions = [], []
        for state, hand, action, reward in trajectory:
            probs, value = self.policy_values(state[None, :], [hand])
            policy_terms.append(-torch.log(probs[0, hand.index(action)].clamp_min(1e-12)))
            if value is not None:
                predictions.append(value[0])
        policy_loss = torch.stack(policy_terms).sum()
        value_loss = torch.tensor(0.0, device=self.device)
        if self.kind == "dirv":
            # Current action's reward is included, including the final capture.
            targets = torch.tensor(discounted_returns([step[3] for step in trajectory], self.gamma), device=self.device)
            value_loss = nn.functional.mse_loss(torch.stack(predictions), targets)
        loss = policy_loss + value_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        self.optimizer.zero_grad()
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None):
            raise FloatingPointError("Non-finite gradients")
        self.optimizer.step()
        return {"policy_loss": float(policy_loss.item()), "value_mse": float(value_loss.item())}

    def checkpoint(self):
        return {"kind": self.kind, "hidden": self.hidden, "gamma": self.gamma,
                "learning_rate": self.learning_rate, "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict()}


def play_game(agents, deal_seed, action_seed, training=False):
    table = Table.deal(deal_seed, len(agents))
    scores = np.zeros(len(agents), dtype=np.float32)
    histories = [[] for _ in agents]
    rng = np.random.default_rng(action_seed)
    for seat, agent in enumerate(agents):
        if agent is not None:
            agent.begin_game(action_seed + seat * 65537)
    for _ in range(10):
        states = table.states()
        hands = [hand.copy() for hand in table.hands]
        actions = [agent.select(state, hand) if agent is not None else int(rng.choice(hand)) for agent, state, hand in zip(agents, states, hands)]
        rewards = table.step(actions)
        scores += rewards
        for seat in range(len(agents)):
            histories[seat].append((states[seat], hands[seat], actions[seat], float(rewards[seat])))
    losses = [agent.learn_episode(history) for agent, history in zip(agents, histories) if agent is not None] if training else []
    return scores, losses


def dump_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def provenance():
    upstream = ROOT / "external/rl-6-nimmt/rl_6_nimmt"
    paths = [Path(__file__), upstream / "agents/mcts.py", upstream / "env.py", upstream / "utils/preprocessing.py"]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def selfplay_worker(connection, config, seat):
    """One persistent independent player, receiving only its own observations."""
    torch.set_num_threads(1)
    agent = SearchAgent(config["kind"], config["seed"] + seat, config["simulations"], config["hidden"], config["learning_rate"], config["gamma"], config["device"])
    history = []
    try:
        while True:
            message = connection.recv()
            command = message[0]
            if command == "stop":
                break
            if command == "restore":
                import io
                saved = torch.load(io.BytesIO(message[1]), map_location=config["device"], weights_only=True)
                agent.model.load_state_dict(saved["model"])
                agent.optimizer.load_state_dict(saved["optimizer"])
                connection.send(True)
            elif command == "begin":
                agent.begin_game(message[1])
                history = []
                connection.send(True)
            elif command == "select":
                state, hand, previous_reward = message[1:]
                if history:
                    history[-1] = (*history[-1][:3], previous_reward)
                action = agent.select(state, hand)
                history.append((state, hand, action, 0.0))
                connection.send(action)
            elif command == "finish":
                history[-1] = (*history[-1][:3], message[1])
                connection.send(agent.learn_episode(history))
            elif command == "checkpoint":
                # Serialize explicitly, avoiding multiprocessing tensor-sharing hooks.
                import io
                buffer = io.BytesIO()
                torch.save(agent.checkpoint(), buffer)
                connection.send_bytes(buffer.getvalue())
    finally:
        connection.close()


class ParallelSelfPlay:
    def __init__(self, args):
        context = mp.get_context("spawn")
        self.connections, self.processes = [], []
        for seat in range(4):
            parent, child = context.Pipe()
            process = context.Process(target=selfplay_worker, args=(child, vars(args), seat), daemon=True)
            process.start()
            child.close()
            self.connections.append(parent)
            self.processes.append(process)

    def game(self, deal_seed, action_seed):
        table = Table.deal(deal_seed)
        for seat, connection in enumerate(self.connections):
            connection.send(("begin", action_seed + seat * 65537))
        for connection in self.connections:
            connection.recv()
        scores, previous_rewards = np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32)
        for _ in range(10):
            states = table.states()
            for seat, connection in enumerate(self.connections):
                connection.send(("select", states[seat], table.hands[seat].copy(), float(previous_rewards[seat])))
            actions = [connection.recv() for connection in self.connections]
            previous_rewards = table.step(actions)
            scores += previous_rewards
        for seat, connection in enumerate(self.connections):
            connection.send(("finish", float(previous_rewards[seat])))
        return scores, [connection.recv() for connection in self.connections]

    def checkpoint(self):
        import io
        for connection in self.connections:
            connection.send(("checkpoint",))
        return [torch.load(io.BytesIO(connection.recv_bytes()), weights_only=True) for connection in self.connections]

    def restore(self, states):
        import io
        for connection, saved in zip(self.connections, states):
            buffer = io.BytesIO()
            torch.save(saved, buffer)
            connection.send(("restore", buffer.getvalue()))
            connection.recv()

    def close(self):
        for connection in self.connections:
            try:
                connection.send(("stop",))
            except (BrokenPipeError, EOFError):
                pass
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
        for connection in self.connections:
            connection.close()


def train(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "checkpoint.pt"
    if checkpoint_path.exists() and not args.resume:
        raise FileExistsError("Output already has a checkpoint. Use --resume or a new output directory.")
    parallel = ParallelSelfPlay(args) if args.workers == 4 else None
    agents = [] if parallel else [SearchAgent(args.kind, args.seed + i, args.simulations, args.hidden, args.learning_rate, args.gamma, args.device) for i in range(4)]
    completed, previous_seconds = 0, 0.0
    if args.resume:
        checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=True)
        if checkpoint["format"] != FORMAT or checkpoint["kind"] != args.kind:
            raise ValueError("Checkpoint format or algorithm mismatch")
        if checkpoint["source_sha256"] != provenance():
            raise ValueError("Source changed since the saved checkpoint; use a new experiment directory")
        for key in ("seed", "simulations", "hidden", "learning_rate", "gamma", "device"):
            if checkpoint["arguments"][key] != getattr(args, key):
                raise ValueError(f"Resume must preserve {key}")
        if parallel:
            parallel.restore(checkpoint["agents"])
        else:
            for agent, saved in zip(agents, checkpoint["agents"]):
                agent.model.load_state_dict(saved["model"])
                agent.optimizer.load_state_dict(saved["optimizer"])
        completed, previous_seconds = checkpoint["games"], checkpoint["seconds"]
        if args.games <= completed:
            raise ValueError("Requested game count must exceed checkpoint count")
        log_path = out / "training.jsonl"
        if log_path.exists():
            kept = [line for line in log_path.read_text(encoding="utf-8").splitlines() if json.loads(line)["game"] <= completed]
            log_path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    started = time.perf_counter()
    protocol = vars(args).copy()
    protocol.update({"format": FORMAT, "paper": REPORT_URL, "source_sha256": provenance(),
                     "runtime": {"python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__, "logical_cpus": os.cpu_count()},
                     "status": "method reimplementation; not original ETH weights"})
    dump_json(out / "protocol.json", protocol)
    with (out / "training.jsonl").open("a", encoding="utf-8") as log:
        for game in range(completed, args.games):
            deal_seed, action_seed = args.seed + game * 104729, args.seed + 1_000_000_000 + game * 8191
            scores, losses = parallel.game(deal_seed, action_seed) if parallel else play_game(agents, deal_seed, action_seed, training=True)
            seconds = previous_seconds + time.perf_counter() - started
            record = {"game": game + 1, "seconds": seconds, "scores": scores.tolist(), "losses": losses}
            log.write(json.dumps(record) + "\n")
            log.flush()
            if (game + 1) % args.save_every == 0 or game + 1 == args.games:
                temporary = checkpoint_path.with_suffix(".tmp")
                torch.save({"format": FORMAT, "kind": args.kind, "games": game + 1,
                            "seconds": seconds, "arguments": vars(args), "source_sha256": provenance(),
                            "agents": parallel.checkpoint() if parallel else [agent.checkpoint() for agent in agents]}, temporary)
                temporary.replace(checkpoint_path)
                print(json.dumps({"kind": args.kind, "games": game + 1, "seconds": round(seconds, 2), "seconds_per_game": round(seconds / (game + 1), 3), "losses": losses[0]}), flush=True)
    if args.games <= completed:
        raise ValueError("Requested game count must exceed checkpoint count")
    summary = {"format": FORMAT, "kind": args.kind, "games": args.games,
               "seconds": seconds, "seconds_per_game": seconds / args.games,
               "device": args.device, "checkpoint": str(checkpoint_path.resolve()),
               "selection": "Fixed final checkpoint; agent index 0, no best-of-four selection"}
    dump_json(out / "summary.json", summary)
    if parallel:
        parallel.close()


def load_agent(path, simulations, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["format"] != FORMAT:
        raise ValueError("Unknown checkpoint")
    saved = checkpoint["agents"][0]
    agent = SearchAgent(saved["kind"], 0, simulations, saved["hidden"], saved["learning_rate"], saved["gamma"], device)
    agent.model.load_state_dict(saved["model"])
    agent.model.eval()
    return agent, checkpoint


def bootstrap_interval(values, seed, resamples=10000):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    boot = np.empty(resamples)
    for start in range(0, resamples, 100):
        stop = min(resamples, start + 100)
        boot[start:stop] = values[rng.integers(0, len(values), size=(stop - start, len(values)))].mean(axis=1)
    return np.quantile(boot, [0.025, 0.975]).tolist()


def evaluate(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "games.jsonl").exists():
        raise FileExistsError("Use a fresh evaluation output directory")
    alpha, alpha_checkpoint = load_agent(args.alpha, args.alpha_simulations, args.device)
    dirv, dirv_checkpoint = load_agent(args.dirv, args.dirv_simulations, args.device)
    if alpha.kind != "alpha" or dirv.kind != "dirv":
        raise ValueError("Reversed algorithms")
    agents = [alpha, dirv, None, None]
    names = ["alpha", "dirv", "random1", "random2"]
    blocks, block_scores = [], []
    strict_blocks, legacy_blocks = [], []
    started = time.perf_counter()
    protocol = vars(args).copy()
    protocol.update({"format": FORMAT, "paper": REPORT_URL, "source_sha256": provenance(),
                     "checkpoint_sha256": {k: hashlib.sha256(Path(v).read_bytes()).hexdigest() for k, v in [("alpha", args.alpha), ("dirv", args.dirv)]},
                     "training_games": {"alpha": alpha_checkpoint["games"], "dirv": dirv_checkpoint["games"]},
                     "independent_unit": "deal seed; all four seat rotations stay in one bootstrap block"})
    dump_json(out / "protocol.json", protocol)
    with (out / "games.jsonl").open("w", encoding="utf-8") as log:
        for block in range(args.deals):
            shares, penalties, stricts, legacies = [], [], [], []
            for rotation in range(4):
                order = [(i + rotation) % 4 for i in range(4)]
                deal_seed = args.seed + block * 104729
                scores, _ = play_game([agents[i] for i in order], deal_seed, args.seed + 2_000_000_000 + block * 65537 + rotation * 8191)
                wins = (scores == max(scores)).astype(float)
                share = wins / wins.sum()
                strict = wins if wins.sum() == 1 else np.zeros(4)
                legacy = np.eye(4)[int(np.argmax(scores))]
                inverse = np.argsort(order)
                shares.append(share[inverse])
                penalties.append(-scores[inverse])
                stricts.append(strict[inverse])
                legacies.append(legacy[inverse])
                log.write(json.dumps({"block": block, "deal_seed": deal_seed, "rotation": rotation,
                                      "seats": [names[i] for i in order], "negative_scores": scores.tolist(),
                                      "win_share": share.tolist()}) + "\n")
                log.flush()
            blocks.append(np.mean(shares, axis=0))
            block_scores.append(np.mean(penalties, axis=0))
            strict_blocks.append(np.mean(stricts, axis=0))
            legacy_blocks.append(np.mean(legacies, axis=0))
            if (block + 1) % 5 == 0 or block + 1 == args.deals:
                print(json.dumps({"evaluated_games": (block + 1) * 4, "seconds": round(time.perf_counter() - started, 2)}), flush=True)
    block_array = np.array(blocks)
    deltas = block_array[:, 1] - block_array[:, 0]
    ci = bootstrap_interval(deltas, args.seed + 999)
    summary = {"format": FORMAT, "games": args.deals * 4, "independent_deals": args.deals,
               "training_games": protocol["training_games"], "seconds": time.perf_counter() - started,
               "win_share": dict(zip(names, block_array.mean(axis=0).tolist())),
               "strict_win_rate": dict(zip(names, np.mean(strict_blocks, axis=0).tolist())),
               "upstream_first_seat_tiebreak_win_rate": dict(zip(names, np.mean(legacy_blocks, axis=0).tolist())),
               "mean_bullheads": dict(zip(names, np.mean(block_scores, axis=0).tolist())),
               "dirv_minus_alpha_pp": float(deltas.mean() * 100),
               "ci95_pp": [v * 100 for v in ci] if ci else None,
               "decision_seconds": {name: {"mean": float(np.mean(agent.decision_seconds)), "p95": float(np.quantile(agent.decision_seconds, 0.95))} for name, agent in zip(names[:2], agents[:2])},
               "interpretation": "Exploratory reimplementation result; timing equality and paper-scale training must be checked separately."}
    dump_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    if Path(sys.prefix).name.lower() != "ntw-ai":
        raise RuntimeError("Use the project ntw-ai Python environment")
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    training = subparsers.add_parser("train")
    training.add_argument("--kind", choices=["alpha", "dirv"], required=True)
    training.add_argument("--games", type=int, required=True)
    training.add_argument("--simulations", type=int, default=50)
    training.add_argument("--hidden", type=int, default=100)
    training.add_argument("--learning-rate", type=float, default=0.001)
    training.add_argument("--gamma", type=float, default=0.99)
    training.add_argument("--save-every", type=int, default=10)
    training.add_argument("--resume", action="store_true")
    training.add_argument("--workers", type=int, choices=[1, 4], default=1, help="Four independent players may search simultaneously; learning schedule is unchanged")
    evaluation = subparsers.add_parser("evaluate")
    evaluation.add_argument("--alpha", required=True)
    evaluation.add_argument("--dirv", required=True)
    evaluation.add_argument("--deals", type=int, default=100, help="Four seat rotations per deal; 100 deals = 400 games")
    evaluation.add_argument("--alpha-simulations", type=int, default=50)
    evaluation.add_argument("--dirv-simulations", type=int, default=200)
    for command in (training, evaluation):
        command.add_argument("--output", required=True)
        command.add_argument("--seed", type=int, required=True)
        command.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
        command.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    for key in ("games", "simulations", "hidden", "save_every", "deals", "alpha_simulations", "dirv_simulations", "threads"):
        if hasattr(args, key) and getattr(args, key) <= 0:
            parser.error(f"{key} must be positive")
    if hasattr(args, "gamma") and not 0 < args.gamma <= 1:
        parser.error("gamma must be in (0, 1]")
    torch.set_num_threads(args.threads)
    train(args) if args.command == "train" else evaluate(args)


if __name__ == "__main__":
    main()
