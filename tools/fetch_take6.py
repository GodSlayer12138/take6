"""Download a pinned upstream snapshot for local evaluation; execute no files."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'external/take6'
REPO = 'GiovanniGatti/take6'


def get(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'ntw-take6-evaluation'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    provenance_path = DEST / 'provenance.json'
    if provenance_path.exists():
        revision = json.loads(provenance_path.read_text())['commit']
    else:
        revision = json.loads(get(f'https://api.github.com/repos/{REPO}/commits/master'))['sha']
    tree = json.loads(get(f'https://api.github.com/repos/{REPO}/git/trees/{revision}?recursive=1'))
    assert not tree['truncated']
    selected = [row for row in tree['tree'] if row['type'] == 'blob' and
                (row['path'].endswith(('.py', '.txt', '.md', '.typed')) or row['path'] == '.gitignore' or
                 row['path'] in ('trained-anns/2-players-expert', 'trained-anns/4-players-standard'))]
    def download(row):
        relative = row['path']
        target = (DEST / relative).resolve()
        assert target.is_relative_to(DEST.resolve())
        data = target.read_bytes() if target.exists() else get(f'https://raw.githubusercontent.com/{REPO}/{revision}/{relative}')
        assert len(data) == row['size']
        git_hash = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
        assert git_hash == row['sha'], relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(data)
        return dict(path=relative, bytes=len(data), git_blob_sha1=git_hash, sha256=hashlib.sha256(data).hexdigest())
    with ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(download, selected))
    value = dict(repository=f'https://github.com/{REPO}', commit=revision, tree=tree['sha'],
                 retrieved_at=datetime.now().astimezone().isoformat(timespec='seconds'), files=files,
                 upstream_paths=[r['path'] for r in tree['tree'] if r['type'] == 'blob'],
                 scope='Local evaluation of original 2-player and 4-player weights. No upstream scripts executed by this downloader. No license file present in the retrieved tree; no redistribution license inferred.')
    provenance_path.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(dict(commit=revision, files=len(files), bytes=sum(r['bytes'] for r in files))))


if __name__ == '__main__':
    main()
