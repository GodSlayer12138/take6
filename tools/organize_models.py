"""Build a model catalog and a conservative inventory without moving artifacts.

Only models/registry.json and, with --inventory, models/inventory.json are written. Historical
manifests, models, training data and evaluation records are never modified.
Uses the standard library; does not load PyTorch checkpoints or start a GPU.
"""
import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / 'models'
CAMPAIGN = ROOT / 'artifacts/progressive-upgrades'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def local(path):
    value = Path(str(path).replace('\\', '/'))
    value = (value if value.is_absolute() else ROOT / value).resolve()
    if not value.is_relative_to(ROOT):
        raise ValueError(f'Path outside repository: {value}')
    return value


def relative(path):
    return local(path).relative_to(ROOT).as_posix()


def sha(path):
    digest = hashlib.sha256()
    with local(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def file_record(path, expected=None):
    path = local(path)
    digest = sha(path)
    if expected is not None and digest != expected:
        raise ValueError(f'Frozen artifact changed: {relative(path)}')
    return dict(path=relative(path), bytes=path.stat().st_size, sha256=digest)


def entry_files(entry):
    files = {}
    for field, digest in (('path', 'sha256'), ('runtime', 'runtime_sha256')):
        if entry.get(field):
            record = file_record(entry[field], entry.get(digest))
            files[record['path']] = record
    for path, digest in entry.get('dependency_sha256', {}).items():
        record = file_record(path, digest)
        files[record['path']] = record
    return list(files.values())


def browser_files():
    # Preserve every static dependency of the current AI module, including
    # auxiliary networks used by its alternative strategies and evaluations.
    source = ROOT / 'src/game/ai-core.js'
    paths = re.findall(r"from\s+['\"]([^'\"]+\.json)['\"]", source.read_text(encoding='utf-8'))
    return [file_record(source.parent / path) for path in paths]


def build_registry():
    state = load(CAMPAIGN / 'state.json')
    champion = state['accepted_upgrades'][-1]
    exploration = load(ROOT / 'artifacts/small-player-exploration/evaluation-manifest.json')
    entries = {entry['id']: entry for entry in exploration['entrants']}
    two = next(row for row in load(ROOT / 'artifacts/small-player-exploration/classic2-results.json')['ranking'] if row['version'] == 'v6')
    catalog = [
        dict(id='champion-3p4p-104', tier='primary', policy_id=champion['candidate'], players=[3, 4], deck_size=104,
             reason='Latest independently accepted upgrade; preserve its exact archived runtime and all frozen dependencies.',
             files=entry_files(champion['policy']), evidence=f'artifacts/progressive-upgrades/{champion["run"]}/results.json',
             results=champion['results'], integrated_in_browser=True,
             web_entrypoint='src/web/ai-router.js', backend='server/champion_api.py'),
        dict(id='specialist-2p-104', tier='primary', policy_id='v6', players=[2], deck_size=104,
             reason='Two-player improvement survived candidate and player-count multiplicity checks; no corresponding 3/4-player claim.',
             files=entry_files(entries['v6']), runtime='scripts/small-strategy-runtime.mjs',
             evidence='artifacts/small-player-exploration/classic2-results.json', results=two, integrated_in_browser=True,
             web_entrypoint='src/web/ai-router.js'),
        dict(id='browser-legacy', tier='compatibility', policy_id='neural_hybrid',
             reason='Legacy browser fallback for rules without a certified specialist, and historical arena opponents; loaded on demand.',
             entrypoint='src/game/ai-core.js', files=browser_files(),
             historical_5p_recipe='artifacts/models/ntw-final-v1.json',
             evidence='docs/TECHNICAL_REPORT.md'),
        dict(id='fast-neural-baseline', tier='baseline', policy_id='ntw-adaptive5-v1',
             reason='Fast standalone baseline, historical 4-player league point leader and a required planner dependency; not a new strength claim.',
             files=entry_files(entries['baseline']) + [file_record('artifacts/models/ntw-adaptive5-v1.pt')],
             evidence='artifacts/small-player-exploration/classic2-results.json'),
        dict(id='starting-v5', tier='baseline', policy_id='v5',
             reason='Starting reference of the independently certified upgrade chain.',
             files=entry_files(entries['v5']), evidence='artifacts/progressive-upgrades/certificate-001/results.json'),
    ]
    for upgrade in state['accepted_upgrades'][:-1]:
        catalog.append(dict(id=f'accepted-upgrade-{upgrade["upgrade"]}', tier='baseline', policy_id=upgrade['candidate'],
                            reason='Required predecessor for replay, regression comparisons and the original acceptance chain.',
                            files=entry_files(upgrade['policy']), evidence=f'artifacts/progressive-upgrades/{upgrade["run"]}/results.json'))
    protocol = load(CAMPAIGN / 'protocol.json')
    opponent_files = {}
    for identifier in protocol['opponents']:
        for record in entry_files(entries[identifier]):
            opponent_files[record['path']] = record
    catalog.append(dict(id='frozen-opponent-pool', tier='baseline', policies=protocol['opponents'],
                        reason='Keep actual evaluation opponents as well as learned proxies; they are different artifacts.',
                        files=list(opponent_files.values()), evidence='artifacts/progressive-upgrades/protocol.json'))
    for folder in ('opponent-proxies-v3-devshift', 'continuation-v3-fourp'):
        paths = [p for p in (CAMPAIGN / folder).glob('*.pt') if p.name not in ('dataset.pt', 'last.pt') and not p.name.endswith('-last.pt')]
        paths += [p.with_suffix('.json') for p in paths if p.with_suffix('.json').exists()]
        catalog.append(dict(id=folder, tier='research',
                            reason='Training and independent numerical audits passed; no accepted win-rate improvement. Preserve for future fresh development.',
                            files=[file_record(path) for path in sorted(paths)],
                            evidence=f'artifacts/progressive-upgrades/{folder}/independent-audit.json',
                            training_directory=f'artifacts/progressive-upgrades/{folder}'))
    return dict(schema_version=1, generated_at=datetime.now().astimezone().isoformat(timespec='seconds'),
                scope='2-4 players research; existing web compatibility retained; comparisons are specific to their rules and opponent pools.',
                active_baseline=state['active_baseline'], accepted_upgrades=len(state['accepted_upgrades']),
                policy='Logical catalog only. Original absolute-path manifests remain unchanged. Files are not a standalone deployment bundle.',
                entries=catalog)


def build_inventory(registry):
    protected = defaultdict(set)
    for entry in registry['entries']:
        for record in entry['files']:
            protected[record['path']].add('catalog:' + entry['id'])
    # Legacy manifests pin entire model directories, even excluded candidates.
    # Mark both live paths and snapshots conservatively rather than inferring
    # that an unselected checkpoint is safe to delete.
    manifests = list((ROOT / 'artifacts').rglob('*manifest*.json'))
    def visit(value, source):
        if isinstance(value, dict):
            for key, child in value.items():
                if isinstance(child, str) and re.fullmatch(r'[0-9a-f]{64}', child):
                    for path in (key, source.parent / 'source-snapshot' / key):
                        try:
                            candidate = local(path)
                        except (ValueError, OSError):
                            continue
                        if candidate.is_file():
                            protected[relative(candidate)].add('manifest:' + relative(source))
                elif key in ('path', 'runtime', 'exported_from') and isinstance(child, str):
                    try:
                        candidate = local(child)
                    except (ValueError, OSError):
                        continue
                    if candidate.is_file():
                        protected[relative(candidate)].add('manifest:' + relative(source))
                visit(child, source)
        elif isinstance(value, list):
            for child in value:
                visit(child, source)
    for manifest in manifests:
        visit(load(manifest), manifest)
    groups = defaultdict(lambda: dict(files=0, bytes=0))
    datasets, review, files = [], [], []
    for path in sorted((ROOT / 'artifacts').rglob('*')):
        if not path.is_file():
            continue
        name, size = relative(path), path.stat().st_size
        group = '/'.join(path.relative_to(ROOT).parts[:2])
        groups[group]['files'] += 1
        groups[group]['bytes'] += size
        reasons = sorted(protected.get(name, ()))
        item = dict(path=name, bytes=size, protection_reasons=reasons)
        if path.name == 'dataset.pt' or path.suffix in ('.npy', '.npz'):
            item['role'] = 'training-data'
            datasets.append(item)
        elif path.suffix == '.pt' or path.parent == ROOT / 'artifacts/models' and path.suffix == '.json':
            item['role'] = 'weight-checkpoint-or-model-config'
            if any(word in name.lower() for word in ('smoke', 'profile', 'pilot')) or re.search(r'-(?:u|e)\d+\.(?:json|pt)$', path.name):
                review.append(dict(item, recommendation='archive-review; keep original path if pinned; not a deletion authorization'))
        elif path.suffix == '.jsonl':
            item['role'] = 'training-or-evaluation-trace'
        else:
            item['role'] = 'metadata-source-or-log'
        files.append(item)
    return dict(schema_version=1, generated_at=registry['generated_at'], artifact_bytes=sum(v['bytes'] for v in groups.values()),
                artifact_files=len(files), folders=dict(groups), scanned_manifests=len(manifests),
                warning='Static references are incomplete for computed paths, glob loading and training provenance. Unprotected never means safe to delete. No artifacts moved or deleted.',
                training_data=sorted(datasets, key=lambda row: -row['bytes']),
                archive_review_candidates=review, files=files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify catalog files and frozen digests without writing')
    parser.add_argument('--inventory', action='store_true', help='Also generate a local, ignored full artifact inventory')
    args = parser.parse_args()
    if args.check:
        registry = load(INDEX / 'registry.json')
        state = load(CAMPAIGN / 'state.json')
        assert registry['active_baseline'] == state['active_baseline'], 'Refresh stale registry'
        records = {record['path']: record for entry in registry['entries'] for record in entry['files']}
        for record in records.values():
            file_record(record['path'], record['sha256'])
        for entry in registry['entries']:
            assert local(entry['evidence']).is_file(), entry['evidence']
        print(json.dumps(dict(status='passed', entries=len(registry['entries']), unique_files=len(records))))
        return
    registry = build_registry()
    INDEX.mkdir(exist_ok=True)
    (INDEX / 'registry.json').write_text(json.dumps(registry, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    summary = dict(entries=len(registry['entries']), artifacts_changed=False)
    if args.inventory:
        inventory = build_inventory(registry)
        (INDEX / 'inventory.json').write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        summary.update(artifact_files=inventory['artifact_files'], artifact_bytes=inventory['artifact_bytes'])
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
