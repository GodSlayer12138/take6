"""Independently replay and rescore the completed four-player proxy diagnostic."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from datetime import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
from small_player_env import SmallGame

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/progressive-upgrades/four-player-proxy-diagnostic-001'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def seed(namespace, tag):
    return int.from_bytes(hashlib.sha256(f'{namespace}:{tag}'.encode()).digest()[:8], 'little') | (1 << 63)


def main():
    protocol, result = load(OUT / 'protocol.json'), load(OUT / 'results.json')
    assert protocol['players'] == 4 and protocol['deck_size'] == 104 and protocol['turns'] == 10 and protocol['focal_seat'] == 0
    assert protocol['games_per_model_per_pool'] == 4096 and protocol['batch_size'] == 128
    assert protocol['bootstrap_replicates'] == 20000 and protocol['total_games'] == result['games'] == 24576
    assert result['status'] == 'complete' and result['protocol_sha256'] == sha(OUT / 'protocol.json')
    assert result['trace_sha256'] == sha(OUT / 'blocks.jsonl')
    for path, digest in protocol['source_sha256'].items():
        assert sha(ROOT / path) == sha(OUT / 'source-snapshot' / path) == digest
    for path, digest in protocol['weights_sha256'].items():
        assert sha(path) == digest
    assert sha(ROOT / 'artifacts/small-player-exploration/evaluation-manifest.json') == protocol['evaluation_manifest_sha256']
    namespace = protocol['seed_namespace']
    records = {}
    with (OUT / 'blocks.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            record = json.loads(line)
            key = record['pool'], record['model'], record['block']
            assert key not in records and key[0] in protocol['proxy_pools'] and key[1] in protocol['models']
            assert 0 <= key[2] < 32 and record['games'] == 128
            assert record['deal_seed'] == seed(namespace, f'diagnostic-{key[2]}')
            assert record['action_seed'] == seed(namespace, f'actions-{key[2]}')
            assert record['deal_seed'] >= 1 << 63
            game = SmallGame(128, 4, record['deal_seed'])
            game.rows = game.rows[:, ::-1].copy()
            assert len(record['actions']) == 10
            for turn in record['actions']:
                cards = np.asarray(turn, dtype=np.int64)
                assert cards.shape == (128, 4)
                assert np.all((game.hands == cards[:, :, None]).sum(axis=-1) == 1)
                game.step(cards)
            np.testing.assert_array_equal(game.scores, np.asarray(record['scores']))
            records[key] = game.scores.copy()
    assert len(records) == 192
    reports = {}
    for pool in protocol['proxy_pools']:
        values = {}
        for model in protocol['models']:
            costs = np.concatenate([records[(pool, model, block)] for block in range(32)])
            winners = costs == costs.min(axis=1, keepdims=True)
            values[model] = (winners / winners.sum(axis=1, keepdims=True))[:, 0]
        published = {row['model']: row for row in result['results'][pool]}
        assert set(published) == set(values)
        rows = []
        for model, wins in values.items():
            difference = wins - values['original-ppo4']
            rng = np.random.default_rng(seed(namespace, 'bootstrap-' + pool))
            bootstrap = np.empty(20000)
            for start in range(0, 20000, 100):
                bootstrap[start:start + 100] = difference[rng.integers(0, 4096, (100, 4096))].mean(axis=1) * 100
            gain = float(difference.mean() * 100)
            interval = np.quantile(bootstrap, [.025, .975])
            row = published[model]
            assert row['games'] == 4096
            np.testing.assert_allclose(row['win_rate'], wins.mean(), rtol=0, atol=1e-12)
            costs = np.concatenate([records[(pool, model, block)] for block in range(32)])
            np.testing.assert_allclose(row['mean_bullheads'], costs[:, 0].mean(), rtol=0, atol=1e-12)
            np.testing.assert_allclose(row['delta_pp'], gain, rtol=0, atol=1e-10)
            np.testing.assert_allclose(row['development_delta_ci95_pp'], interval, rtol=0, atol=1e-9)
            rows.append(dict(model=model, delta_pp=gain, independent_ci95_pp=interval.tolist()))
        reports[pool] = rows
    value = dict(status='passed', observed_at=datetime.now().astimezone().isoformat(), verified_games=24576,
                 results=reports, all_sources_and_weights_checked=True, legal_actions_and_scores_replayed=True,
                 independent_paired_statistics=True, certificate_or_incomplete_development_outcomes_read=False,
                 auditor_sha256=sha(__file__), protocol_sha256=sha(OUT / 'protocol.json'), results_sha256=sha(OUT / 'results.json'),
                 scope='Development diagnostic against learned proxies only; does not accept an upgrade.')
    (OUT / 'independent-audit.json').write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(value), flush=True)


if __name__ == '__main__':
    main()
