"""Replay replication 008 and independently verify its frozen finalist selection."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from datetime import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
from report_local_tournament import replay

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/progressive-upgrades'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def seed(namespace, players, block):
    return int.from_bytes(hashlib.sha256(f'ntw-progressive-20260909:{namespace}:{players}:{block}'.encode()).digest()[:4], 'little')


def main():
    folder = OUT / 'development-008'
    manifest, result = load(folder / 'manifest.json'), load(folder / 'results.json')
    selection = load(OUT / 'selection-development-008.json')
    screen, screen_selection = load(OUT / 'development-007/manifest.json'), load(OUT / 'selection-development-007.json')
    certificate = load(OUT / 'certificate-004/manifest.json')
    protocol, state = load(OUT / 'protocol.json'), load(OUT / 'state.json')
    assert manifest['purpose'] == result['purpose'] == 'development' and manifest['deals'] == 512
    assert result['audit'] == 'passed' and result['games'] == 10752
    assert sha(folder / 'blocks.jsonl') == result['trace_sha256'] == selection['trace_sha256']
    assert sha(folder / 'manifest.json') == selection['manifest_sha256']
    assert sha(folder / 'results.json') == selection['results_sha256']
    assert sha(ROOT / 'training/advance_progressive_development7.py') == selection['selector_sha256']
    assert selection['frozen_master_protocol'] == screen['selection_protocol']
    assert manifest['selection_protocol']['minimum_observed_delta_pp'] == .5
    assert manifest['selection_protocol']['minimum_ordinary95_lower_pp'] == 0
    assert manifest['selection_protocol']['lower_bound_comparison'] == 'strictly greater than zero'
    assert manifest['versions'] == ['reference'] + screen_selection['selected']
    assert manifest['source_screen']['selection_sha256'] == sha(OUT / 'selection-development-007.json')
    assert manifest['source_screen']['manifest_sha256'] == sha(OUT / 'development-007/manifest.json')
    entries, original = {e['id']: e for e in manifest['entrants']}, {e['id']: e for e in screen['entrants']}
    for name in manifest['versions']:
        assert entries[name] == original[name]
    assert entries['reference'] == dict(state['accepted_upgrades'][1]['policy'], id='reference')
    for entry in entries.values():
        if 'sha256' in entry:
            assert sha(entry['path']) == entry['sha256']
        if 'runtime_sha256' in entry:
            assert sha(entry['runtime']) == entry['runtime_sha256']
        for path, digest in entry.get('dependency_sha256', {}).items():
            assert sha(path) == digest
    for name, digest in manifest['source_sha256'].items():
        assert sha(folder / 'source-snapshot' / name) == digest
    for name, digest in manifest.get('common_dependency_sha256', {}).items():
        assert sha(name) == digest
    expected, blocks, games = {}, {3: {}, 4: {}}, 0
    for p in (3, 4):
        rng = np.random.default_rng(seed(manifest['namespace'], p, -1))
        for block in range(512):
            expected[(p, block)] = rng.choice(protocol['opponents'], p - 1, replace=False).tolist()
    with (folder / 'blocks.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            block = json.loads(line)
            p, number = block['players'], block['block']
            assert number not in blocks[p] and block['opponents'] == expected[(p, number)]
            assert block['deal_seed'] == seed(manifest['namespace'], p, number)
            assert len(block['records']) == p * len(manifest['versions'])
            values = []
            for version in manifest['versions']:
                records = [r for r in block['records'] if r['candidate'] == version]
                assert len(records) == p
                shares = []
                for rotation, record in enumerate(records):
                    group = [version] + block['opponents']
                    assert record['rotation'] == rotation and record['seats'] == group[rotation:] + group[:rotation]
                    assert record['deal_seed'] == block['deal_seed']
                    replay(record, p, 104)
                    games += 1
                    costs = np.asarray(record['bullheads'])
                    winners = costs == costs.min()
                    shares.append(float(winners[record['seats'].index(version)] / winners.sum()))
                values.append(float(np.mean(shares)))
            blocks[p][number] = values
    assert games == result['games'] == 10752
    independent = {}
    for p, values in blocks.items():
        assert len(values) == 512
        matrix = np.asarray([values[i] for i in range(512)])
        published = {r['version']: r for r in result['results'][str(p)]}
        assert set(published) == set(manifest['versions'])
        independent[p] = {}
        for column, version in enumerate(manifest['versions']):
            difference = matrix[:, column] - matrix[:, 0]
            rng = np.random.default_rng(seed(manifest['namespace'] + '-bootstrap', p, -1))
            bootstrap = np.empty(20000)
            for start in range(0, 20000, 100):
                bootstrap[start:start + 100] = difference[rng.integers(0, 512, (100, 512))].mean(axis=1) * 100
            gain, interval = float(difference.mean() * 100), np.quantile(bootstrap, [.025, .975])
            row = published[version]
            assert row['games'] == 512 * p
            np.testing.assert_allclose(row['win_rate'], matrix[:, column].mean(), rtol=0, atol=1e-12)
            np.testing.assert_allclose(row['delta_pp'], gain, rtol=0, atol=1e-10)
            np.testing.assert_allclose(row['delta_ci95_pp'], interval, rtol=0, atol=1e-9)
            independent[p][version] = dict(delta_pp=gain, lower95_pp=float(interval[0]))
    ranking = []
    for version in manifest['versions'][1:]:
        gain = [independent[p][version]['delta_pp'] for p in (3, 4)]
        lower = [independent[p][version]['lower95_pp'] for p in (3, 4)]
        ranking.append(dict(candidate=version, delta_3_pp=gain[0], delta_4_pp=gain[1],
                            lower95_3_pp=lower[0], lower95_4_pp=lower[1],
                            minimum_delta_pp=min(gain), mean_delta_pp=sum(gain) / 2,
                            minimum_lower95_pp=min(lower), eligible=min(gain) >= .5 and min(lower) > 0))
    ranking.sort(key=lambda row: (-row['minimum_lower95_pp'], -row['minimum_delta_pp'], -row['mean_delta_pp'], row['candidate']))
    for a, b in zip(ranking, selection['ranking']):
        assert a['candidate'] == b['candidate'] and a['eligible'] == b['eligible']
        for key in set(a) - {'candidate', 'eligible'}:
            np.testing.assert_allclose(a[key], b[key], rtol=0, atol=1e-9)
    selected = [r['candidate'] for r in ranking if r['eligible']][:1]
    assert selected == selection['selected'] == ['distill2048-specialist2048']
    assert certificate['versions'] == ['reference'] + selected
    assert certificate['purpose'] == 'certificate' and certificate['deals'] == 2048
    assert certificate['certificate_attempt'] == 4 and certificate['alpha_per_mode'] == .00125
    assert certificate['predecessor'] == 'proxy-v2-512-lowprior'
    assert certificate['selection_record']['sha256'] == sha(OUT / 'selection-development-008.json')
    for entry in certificate['entrants']:
        assert entry == entries[entry['id']]
    report = dict(status='passed', observed_at=datetime.now().astimezone().isoformat(), verified_games=games,
                  independent_ranking=ranking, selected=selected, all_focal_policies_exact_screen_copies=True,
                  all_certificate_entrants_exact_replication_copies=True, source_and_weight_hashes_checked=True,
                  independent_paired_statistics=True, certificate_outcomes_read=False,
                  before_certificate_games=not (OUT / 'certificate-004/blocks.jsonl').exists(),
                  selector_sha256=selection['selector_sha256'], selection_sha256=sha(OUT / 'selection-development-008.json'),
                  certificate_manifest_sha256=sha(OUT / 'certificate-004/manifest.json'), auditor_sha256=sha(__file__))
    (OUT / 'selection-development-008-independent-audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
