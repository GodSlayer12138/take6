"""Restore selected historical artifacts from the pre-cleanup Git revision.

No arguments lists the available groups. Use a repository-relative path prefix
or --all to restore. Existing files are never overwritten, and every restored
file is checked against the recorded SHA-256 before it is installed.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def destination(root, name):
    parts = PurePosixPath(name)
    if '\\' in name or parts.is_absolute() or '..' in parts.parts:
        raise ValueError(f'Invalid archive path: {name}')
    if not name.startswith('artifacts/') and name != 'models/inventory.json':
        raise ValueError(f'Not a research artifact: {name}')
    target = root.joinpath(*parts.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Path outside repository: {name}')
    return target


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def restore(root, revision, records):
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('Invalid archived Git revision')
    # Check all conflicts before writing any files.
    missing = []
    for record in records:
        target = destination(root, record['path'])
        if target.exists():
            if not target.is_file() or digest(target) != record['sha256']:
                raise ValueError(f'Refusing to overwrite changed file: {record["path"]}')
        else:
            missing.append((record, target))
    if not missing:
        return 0
    available = subprocess.run(['git', 'cat-file', '-e', revision + '^{commit}'],
                               cwd=root, capture_output=True)
    if available.returncode:
        raise ValueError(f'Historical commit is not available locally. Run: git fetch origin {revision}')
    for record, target in missing:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                result = subprocess.run(['git', 'show', revision + ':' + record['path']],
                                        cwd=root, stdout=stream, stderr=subprocess.PIPE)
            if result.returncode:
                raise ValueError(f'Git could not restore {record["path"]}: {result.stderr.decode(errors="replace")}')
            if temporary.stat().st_size != record['bytes'] or digest(temporary) != record['sha256']:
                raise ValueError(f'Archived checksum mismatch: {record["path"]}')
            # Atomically install verified bytes without replacing a file that
            # may have appeared after preflight (temporary is on the same disk).
            os.link(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return len(missing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='*', help='File or directory prefixes, e.g. artifacts/local-tournament')
    parser.add_argument('--all', action='store_true', help='Restore all tracked historical artifacts')
    args = parser.parse_args()
    archive = json.loads((ROOT / 'models/history.json').read_text(encoding='utf-8'))
    if not args.paths and not args.all:
        groups = defaultdict(lambda: [0, 0])
        for row in archive['files']:
            group = '/'.join(row['path'].split('/')[:2])
            groups[group][0] += 1
            groups[group][1] += row['bytes']
        for group, (count, size) in sorted(groups.items()):
            print(f'{group}: {count} files, {size / 1024**2:.1f} MiB')
        print('Restore selected paths or pass --all. Local untracked data is not included.')
        return
    selected = {row['path']: row for row in archive['files']} if args.all else {}
    for prefix in args.paths:
        prefix = prefix.replace('\\', '/').rstrip('/')
        matches = [row for row in archive['files'] if row['path'] == prefix or row['path'].startswith(prefix + '/')]
        if not matches:
            parser.error(f'No archived files match: {prefix}')
        selected.update((row['path'], row) for row in matches)
    restored = restore(ROOT, archive['revision'], list(selected.values()))
    print(f'Restored {restored} files; {len(selected) - restored} already present and verified.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as error:
        raise SystemExit(str(error))
