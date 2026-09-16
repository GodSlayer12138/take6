"""Describe external-opponent scores in an existing certificate; no new games."""
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    folder = ROOT / 'artifacts/progressive-upgrades/certificate-004'
    trace = (folder / 'blocks.jsonl').read_bytes()
    results = load(folder / 'results.json')
    digest = hashlib.sha256(trace).hexdigest()
    assert digest == results['trace_sha256'], 'Certificate trace changed'
    policy = 'distill2048-specialist2048'
    rows = defaultdict(lambda: dict(games=0, wins=0, ties=0, losses=0,
                                    own_sum=0., opponent_sum=0., deals=set()))
    totals = defaultdict(lambda: dict(games=0, shares=0.))
    blocks_seen = set()
    for line in trace.splitlines():
        block = json.loads(line)
        n = block['players']
        key = (n, block['block'])
        assert key not in blocks_seen
        blocks_seen.add(key)
        selected = [r for r in block['records'] if r['candidate'] == policy]
        assert len(selected) == n and {r['rotation'] for r in selected} == set(range(n))
        for record in selected:
            seats, scores = record['seats'], record['bullheads']
            assert len(seats) == n and len(set(seats)) == n
            own = seats.index(policy)
            score = scores[own]
            first = min(scores)
            share = (1 / scores.count(first)) if score == first else 0
            assert abs(share - record['shares'][own]) < 1e-12
            totals[n]['games'] += 1
            totals[n]['shares'] += share
            for seat, opponent in enumerate(seats):
                if seat == own:
                    continue
                row = rows[n, opponent]
                row['games'] += 1
                row['wins'] += score < scores[seat]
                row['ties'] += score == scores[seat]
                row['losses'] += score > scores[seat]
                row['own_sum'] += score
                row['opponent_sum'] += scores[seat]
                row['deals'].add(block['deal_seed'])
    output_rows = []
    for n, total in totals.items():
        original = next(r for r in results['results'][str(n)] if r['version'] == policy)
        assert total['games'] == original['games']
        assert abs(total['shares'] / total['games'] - original['win_rate']) < 1e-12
        assert sum(row['games'] for (p, _), row in rows.items() if p == n) == total['games'] * (n - 1)
    for (n, opponent), row in sorted(rows.items()):
        if opponent not in ('mcs', 'alpha-2000', 'dirv-10000'):
            continue
        games = row['games']
        assert row['wins'] + row['ties'] + row['losses'] == games
        assert len(row['deals']) * n == games
        output_rows.append(dict(players=n, opponent=opponent, games=games,
                                independent_deals=len(row['deals']),
                                wins=row['wins'], ties=row['ties'], losses=row['losses'],
                                pair_score_share=(row['wins'] + .5 * row['ties']) / games,
                                champion_mean_bullheads=row['own_sum'] / games,
                                opponent_mean_bullheads=row['opponent_sum'] / games))
    report = dict(status='passed', observed_at=datetime.now().astimezone().isoformat(timespec='seconds'),
                  policy=policy, certificate='certificate-004', trace_sha256=digest,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='Post-hoc descriptive same-table score comparisons in existing 3/4-player games. A win means fewer bullheads than that opponent, ties count half. Not 2-player win rate, whole-table first-place share, equal-compute comparison, or a new formal significance claim. Seat rotations are correlated within deal groups.',
                  rows=output_rows)
    destination = ROOT / 'artifacts/web-integration/external-comparison.json'
    destination.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
