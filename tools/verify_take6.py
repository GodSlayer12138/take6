"""Independent upstream encoding/rule checks and NumPy versus Torch equations."""
from __future__ import annotations
import __future__
import ast
import json
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import numpy as np
import torch
from take6_adapter import ROOT, UPSTREAM, Take6Actor, encode, sha


def upstream_classes():
    # Extract only these reviewed, dependency-free classes. The complete upstream
    # environment/entrypoints are not imported (Ray/Gym/Azure are unnecessary).
    source = (UPSTREAM / 'src/take6/env.py').read_text()
    old = 'idx = np.nonzero(highest_cards == highest)[0]'
    assert source.count(old) == 1
    source = source.replace(old, 'idx = int(np.nonzero(highest_cards == highest)[0][0])')
    tree = ast.parse(source)
    names = {'Table', 'Hand', 'RuleState', 'Scoreboard'}
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    namespace = {'np': np}
    exec(compile(tree, 'audited_take6_classes', 'exec', flags=__future__.annotations.compiler_flag), namespace)
    return namespace


def audit_native(record, players, deck, classes=None):
    classes = classes or upstream_classes()
    cards = (np.random.default_rng(record['deal_seed']).permutation(deck) + 1).tolist()
    hands = [set(cards[i*10:i*10+10]) for i in range(players)]
    table = classes['Table']()
    table.reset(np.asarray([cards[-1-i] for i in range(4)]))
    scores = np.zeros(players, dtype=int)
    assert len(record['actions']) == 10
    for raw in record['actions']:
        actions = np.asarray(raw) + 1
        for i, card in enumerate(actions):
            assert card in hands[i]
            hands[i].remove(card)
        scores += table.play(actions)
    assert all(not h for h in hands)
    assert scores.tolist() == record['bullheads']
    minimum = min(scores)
    assert record['shares'] == [float(s == minimum) / sum(scores == minimum) for s in scores]


def main():
    torch.set_num_threads(1)
    classes = upstream_classes()
    max_logit_error = 0.
    max_probability_error = 0.
    count = 0
    models = []
    for players in (2, 4):
        actor = Take6Actor(players)
        models.append(actor.metadata)
        rng = np.random.default_rng(0x71A600 + players)
        for game in range(32):
            expert = players == 2 and game % 2 == 0
            deck = players * 10 + 4 if expert else 104
            cards = rng.permutation(deck) + 1
            hands = [classes['Hand'](cards[i*10:i*10+10].copy()) for i in range(players)]
            table = classes['Table']()
            table.reset(cards[-4:].copy())
            board = classes['Scoreboard']()
            board.reset(players)
            seen = cards[-4:].tolist()
            for turn in range(10):
                for seat in range(players):
                    hand = hands[seat].cards[hands[seat].cards > 0].tolist()
                    rows = [row[row > 0].tolist() for row in table.rows]
                    x, mask, padded = encode(hand, rows, seen, board.scores.tolist(), seat, expert)
                    history = np.zeros(104, np.float32)
                    history[np.asarray(seen) - 1] = 1
                    sc = board.encode()
                    scores = np.concatenate((sc[seat:seat+1], sc[np.arange(10) != seat]))
                    reference = np.concatenate((hands[seat].encode(), table.encode(), history, scores,
                                                classes['RuleState'](players, expert).encode()))
                    np.testing.assert_array_equal(x, reference)
                    np.testing.assert_array_equal(mask, hands[seat].mask())
                    np.testing.assert_array_equal(padded, hands[seat].cards)
                    y = reference.copy()
                    for i, (kernel, bias) in enumerate(actor.arrays):
                        y = y @ kernel + bias
                        if i < 3:
                            y = np.maximum(y, 0)
                    actual = actor.logits(x)
                    max_logit_error = max(max_logit_error, float(np.max(np.abs(actual - y))))
                    np.testing.assert_allclose(actual, y, rtol=2e-5, atol=5e-4)
                    expected_probs = actor.probabilities(y, mask)
                    actual_probs = actor.probabilities(actual, mask)
                    max_probability_error = max(max_probability_error, float(np.max(np.abs(actual_probs - expected_probs))))
                    np.testing.assert_allclose(actual_probs, expected_probs, rtol=2e-4, atol=3e-5)
                    seed = game * 1000 + turn * 10 + seat
                    chosen = actor.choose(hand, rows, seen, board.scores.tolist(), seat, expert, np.random.RandomState(seed))
                    assert chosen in hand and np.all(actual_probs[mask == 0] == 0)
                    count += 1
                actions = np.asarray([h.select(int(rng.choice(np.flatnonzero(h.mask())))) for h in hands])
                board += table.play(actions)
                seen.extend(actions.tolist())
    # Explicit public score clipping and absent-seat padding contract.
    x, _, _ = encode([1], [[2], [3], [4], [5]], [2, 3, 4, 5], [132, 33, 99, 0], 2, False)
    np.testing.assert_array_equal(x[624:634], [1, 1, .5, 0, -1, -1, -1, -1, -1, -1])
    result = dict(status='passed', public_state_cases=count, random_games=64,
        max_abs_logit_error=max_logit_error, max_abs_probability_error=max_probability_error,
        encoding='Exact array equality against extracted upstream Hand/Table/Scoreboard/RuleState classes; public history and ScoreWrapper seat order checked.',
        compatibility_change='Extracted Table.play tie index converted from length-one array to int for NumPy 2.4; original upstream file unchanged.',
        runtime_limit='Original TensorFlow/Ray process was not run; independent NumPy equations and PyTorch FP32 output compared against audited upstream architecture.',
        models=models, source_sha256={str(p.relative_to(ROOT)):sha(p) for p in
        [Path(__file__), ROOT/'tools/take6_adapter.py', UPSTREAM/'src/take6/env.py', UPSTREAM/'src/take6/model.py', ROOT/'external/take6-reference/ray-2.0.0-fcnet.py']})
    out = ROOT / 'artifacts/take6-evaluation'
    out.mkdir(parents=True, exist_ok=True)
    (out/'inference-verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
