"""Replay take6 decisions from traces using upstream encoders and NumPy Dense.

This never calls the adapter's encode/logits/choose methods. It checks the
actual recorded action against independently reconstructed public inputs.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from datetime import datetime
import json
import sys
import numpy as np
from take6_adapter import ROOT, Take6Actor, sha
from verify_take6 import upstream_classes
sys.path.insert(0, str(ROOT/'training'))
from report_local_tournament import bullheads

OUT = ROOT/'artifacts/take6-evaluation'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    protocol = json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    models = {p:Take6Actor(p) for p in (2,4)}
    classes = upstream_classes()
    # Snapshot bytes once so a growing live trace cannot create a partial record.
    raw = (OUT/'blocks.jsonl').read_bytes()
    if args.partial:
        raw = raw[:raw.rfind(b'\n')+1]
    cases, games, near_boundary = 0, 0, []
    for line in raw.splitlines():
        for record in json.loads(line):
            config = protocol['modes'][record['mode']]
            p, deck = config['players'], config['deck']
            cards = np.random.default_rng(record['deal_seed']).permutation(deck) + 1
            seat = record['seats'].index(f'take6-{p}')
            hand = classes['Hand'](cards[seat*10:seat*10+10].copy())
            table = classes['Table']()
            table.reset(np.asarray([cards[-1-i] for i in range(4)]))
            scoreboard = classes['Scoreboard']()
            scoreboard.reset(p)
            history = np.zeros(104, dtype=np.float32)
            history[table.rows[table.rows>0]-1] = 1
            rng = np.random.RandomState(record['take6_sampling_seed'])
            for turn, raw_actions in enumerate(record['actions']):
                scores = scoreboard.encode()
                relative = np.concatenate((scores[seat:seat+1], scores[np.arange(10) != seat]))
                x = np.concatenate((hand.encode(), table.encode(), history, relative,
                    classes['RuleState'](p, deck == 10*p+4).encode()))
                for i, (kernel, bias) in enumerate(models[p].arrays):
                    x = x @ kernel + bias
                    if i < 3:
                        x = np.maximum(x, 0)
                mask = hand.mask()
                x[mask == 0] = np.finfo(np.float32).min
                prob = np.exp(x - x.max())
                prob /= prob.sum()
                # RandomState.choice with p uses one uniform draw and the
                # normalized float64 cumulative distribution.
                cdf = np.cumsum(prob, dtype=np.float64)
                cdf /= cdf[-1]
                draw = rng.random_sample()
                index = int(np.searchsorted(cdf, draw, side='right'))
                actions = np.asarray(raw_actions) + 1
                chosen_index = int(np.flatnonzero(hand.cards == actions[seat])[0])
                if index != chosen_index:
                    # Keep numerical boundary cases explicit; never silently
                    # equate a different sampled action with exact agreement.
                    distance = float(np.min(np.abs(cdf - draw)))
                    assert distance < 0.0003, (record['mode'],record['block'],turn,'large decision mismatch')
                    near_boundary.append(dict(mode=record['mode'], block=record['block'],
                        rotation=record['rotation'], turn=turn, cdf_distance=distance))
                assert hand.select(chosen_index) == actions[seat]
                if config['rules'] == 'take6':
                    penalties = table.play(actions)
                else:
                    rows = [row[row>0].tolist() for row in table.rows]
                    penalties = np.zeros(p, dtype=int)
                    for player in sorted(range(p), key=lambda i:actions[i]):
                        card = int(actions[player])
                        eligible = [(r[-1],i) for i,r in enumerate(rows) if r[-1]<card]
                        forced = not eligible
                        target = max(eligible)[1] if eligible else min(range(4), key=lambda i:(sum(map(bullheads,rows[i])),len(rows[i]),i))
                        if forced or len(rows[target]) == 5:
                            penalties[player] += sum(map(bullheads,rows[target]))
                            rows[target] = [card]
                        else:
                            rows[target].append(card)
                    table.rows[:] = 0
                    for i,row in enumerate(rows):
                        table.rows[i,:len(row)] = row
                scoreboard += penalties
                history[actions-1] = 1
                cases += 1
            assert scoreboard.scores.tolist() == record['bullheads']
            games += 1
    if not args.partial:
        assert games == protocol['planned_games']
    result = dict(status='passed' if not near_boundary else 'numeric-boundary-cases-require-review',
        complete=not args.partial, games=games, decisions=cases,
        exact_numpy_action_matches=cases-len(near_boundary), numerical_boundary_cases=near_boundary,
        method='Upstream public encoders, independent NumPy dense/ReLU/masked softmax, reconstructed RandomState uniforms, independent rule replay; adapter encoding and action methods not called.',
        observed_at=datetime.now().astimezone().isoformat(timespec='seconds'),
        protocol_sha256=sha(OUT/'protocol.json'), script_sha256=sha(__file__),
        trace_snapshot_sha256=__import__('hashlib').sha256(raw).hexdigest())
    if not args.partial:
        (OUT/'decision-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
