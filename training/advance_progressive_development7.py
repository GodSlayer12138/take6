"""Apply the development-007 screen and replication rules, only after complete runs.

This prepares manifests and selection records; it never launches games, trains
weights, changes completed results, or relaxes certificate acceptance rules.
"""
import argparse
from datetime import datetime
from pathlib import Path

import progressive_campaign as campaign

OUT = campaign.OUT


def identity(entry):
    return (entry['sha256'], entry.get('runtime_sha256'), entry.get('dependency_sha256', {}))


def completed(name, deals):
    folder = OUT / name
    manifest = campaign.load(folder / 'manifest.json')
    result = campaign.load(folder / 'results.json')
    assert manifest['purpose'] == result['purpose'] == 'development'
    assert manifest['deals'] == deals
    assert result['audit'] == 'passed'
    assert result['trace_sha256'] == campaign.sha(folder / 'blocks.jsonl')
    assert result['games'] == deals * 7 * len(manifest['versions'])
    campaign.verify(manifest, folder)
    for p in (3, 4):
        rows = result['results'][str(p)]
        assert len(rows) == len(manifest['versions'])
        assert {r['version'] for r in rows} == set(manifest['versions'])
        reference = next(r for r in rows if r['version'] == 'reference')
        for r in rows:
            assert r['games'] == deals * p
            assert abs(r['delta_pp'] - (r['win_rate'] - reference['win_rate']) * 100) < 1e-9
    return manifest, result


def ranking(manifest, result, replication):
    rows = {p: {r['version']: r for r in result['results'][str(p)]} for p in (3, 4)}
    candidates = []
    for name in manifest['versions']:
        if name == 'reference':
            continue
        gain = [rows[p][name]['delta_pp'] for p in (3, 4)]
        lower = [rows[p][name]['delta_ci95_pp'][0] for p in (3, 4)]
        candidates.append(dict(candidate=name, delta_3_pp=gain[0], delta_4_pp=gain[1],
                               lower95_3_pp=lower[0], lower95_4_pp=lower[1],
                               minimum_delta_pp=min(gain), mean_delta_pp=sum(gain) / 2,
                               minimum_lower95_pp=min(lower),
                               eligible=min(gain) >= .5 and (not replication or min(lower) > 0)))
    if replication:
        candidates.sort(key=lambda r: (-r['minimum_lower95_pp'], -r['minimum_delta_pp'],
                                       -r['mean_delta_pp'], r['candidate']))
    else:
        candidates.sort(key=lambda r: (-r['minimum_delta_pp'], -r['mean_delta_pp'], r['candidate']))
    return candidates


def selection(name, manifest, result, rows, selected, master):
    folder = OUT / name
    path = OUT / f'selection-{name}.json'
    value = dict(development=name, predecessor='proxy-v2-512-lowprior',
                 frozen_master_protocol=master, ranking=rows, selected=selected,
                 manifest_sha256=campaign.sha(folder / 'manifest.json'),
                 results_sha256=campaign.sha(folder / 'results.json'),
                 trace_sha256=result['trace_sha256'], selector_sha256=campaign.sha(__file__),
                 basis='Complete development only; no incomplete run or certificate outcome is used.')
    if path.exists():
        previous = campaign.load(path)
        assert all(previous.get(k) == v for k, v in value.items()), 'Do not overwrite a prior selection'
    else:
        campaign.dump(path, dict(value, selected_at=datetime.now().astimezone().isoformat()))
    return path


def replicate(manifest, selected, record, master):
    entries = {e['id']: e for e in manifest['entrants']}
    reference = entries['reference']
    versions = ['reference'] + selected
    path = OUT / 'development-008/manifest.json'
    if path.exists():
        previous = campaign.load(path)
        assert previous['versions'] == versions and previous['deals'] == 512
        assert previous['source_screen']['selection_sha256'] == campaign.sha(record)
        assert previous['selection_protocol']['eligibility'] == master['finalist_eligibility']
        assert previous['selection_protocol']['ranking'] == master['finalist_ranking']
        for name in versions:
            assert identity(next(e for e in previous['entrants'] if e['id'] == name)) == identity(entries[name])
        campaign.verify(previous, path.parent)
        return path
    path = campaign.prepare_run('development-008', {}, 512, 'development', reference)
    target = campaign.load(path)
    target['versions'] = versions
    target['entrants'].extend(dict(entries[name]) for name in selected)
    target['source_screen'] = dict(run='development-007',
                                  manifest_sha256=campaign.sha(OUT / 'development-007/manifest.json'),
                                  selection_sha256=campaign.sha(record))
    target['selection_protocol'] = dict(
        stage='replication', deals_per_player_count=512,
        eligibility=master['finalist_eligibility'], ranking=master['finalist_ranking'],
        minimum_observed_delta_pp=.5, minimum_ordinary95_lower_pp=0,
        lower_bound_comparison='strictly greater than zero',
        no_interim_selection=True, if_none_eligible=master['if_none_eligible'],
        formal_certificate=master['formal_certificate'],
    )
    target['preflight_protocol'] = dict(
        script='training/verify_progressive_service.py',
        script_sha256=campaign.sha(campaign.ROOT / 'training/verify_progressive_service.py'),
        archive='development-007', policies=versions, decisions_per_policy=140,
        streams=2, clients=8,
        scope='Exactly copied policies must reproduce their completed screen actions before fresh replication games.',
    )
    target['policy_copy_rule'] = 'Original config, runtime and dependency entries copied exactly; no retraining or parameter changes.'
    campaign.dump(path, target)
    return path


def main(args):
    name = 'development-007' if args.stage == 'screen' else 'development-008'
    if not (OUT / name / 'results.json').exists():
        print(f'Not ready: {name} has no complete results; no selection or manifest was created.', flush=True)
        return
    state = campaign.load(OUT / 'state.json')
    assert state['active_baseline'] == 'proxy-v2-512-lowprior' and len(state['accepted_upgrades']) == 2
    assert state['certificate_attempts'][-1]['attempt'] == 3 and not state['certificate_attempts'][-1]['passed']
    master = campaign.load(OUT / 'development-007/manifest.json')['selection_protocol']
    assert master['screen_deals_per_player_count'] == 256 and master['shortlist_limit'] == 2
    manifest, result = completed(name, 256 if args.stage == 'screen' else 512)
    reference = next(e for e in manifest['entrants'] if e['id'] == 'reference')
    assert identity(reference) == identity(state['accepted_upgrades'][-1]['policy'])
    if args.stage == 'replication':
        screen_record = campaign.load(OUT / 'selection-development-007.json')
        screen = campaign.load(OUT / 'development-007/manifest.json')
        assert screen_record['manifest_sha256'] == campaign.sha(OUT / 'development-007/manifest.json')
        assert manifest['versions'] == ['reference'] + screen_record['selected']
        assert manifest['source_screen']['selection_sha256'] == campaign.sha(OUT / 'selection-development-007.json')
        assert manifest['selection_protocol']['eligibility'] == master['finalist_eligibility']
        assert manifest['selection_protocol']['ranking'] == master['finalist_ranking']
        original = {e['id']: e for e in screen['entrants']}
        for version in manifest['versions']:
            assert identity(next(e for e in manifest['entrants'] if e['id'] == version)) == identity(original[version])
    rows = ranking(manifest, result, args.stage == 'replication')
    limit = 2 if args.stage == 'screen' else 1
    selected = [r['candidate'] for r in rows if r['eligible']][:limit]
    record = selection(name, manifest, result, rows, selected, master)
    if not selected:
        print(f'No eligible candidate. Preserved {record}; continue development.', flush=True)
        return
    if args.stage == 'screen':
        assert not state.get('pending_certificate')
        path = replicate(manifest, selected, record, master)
    else:
        path = campaign.prepare_certificate(name, selected[0])
        certificate = campaign.load(path)
        assert certificate['certificate_attempt'] == 4 and certificate['alpha_per_mode'] == .05 / 40
    print(f'Selection: {record}\nPrepared: {path}\nGames have not been launched by this script.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['screen', 'replication'], required=True)
    main(parser.parse_args())
