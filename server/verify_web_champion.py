"""Compare the web transport with archived decisions, without a new strength claim."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
from verify_progressive_config_service import observations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8765')
    parser.add_argument('--output', default='artifacts/web-integration/http-equivalence.json')
    args = parser.parse_args()
    cases, trace = observations('development-008', 'distill2048-specialist2048')
    fields = ['hand', 'rows', 'seenCards', 'playerCount', 'deckSize', 'scores', 'seed']
    start = time.perf_counter()
    def check(case):
        payload = {key: case['observation'][key] for key in fields}
        request = urllib.request.Request(args.url.rstrip('/') + '/api/ai/choose',
            data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.load(response)
        assert result['card'] == case['archived_card'], (case, result)
        assert result['model'] == 'distill2048-specialist2048' and result['worlds'] == 2048
        return case['observation']['playerCount']
    with ThreadPoolExecutor(max_workers=8) as pool:
        checked = list(pool.map(check, cases))
    report = dict(status='passed', observed_at=datetime.now().astimezone().isoformat(), url=args.url,
                  decisions=len(checked), by_players={p: checked.count(p) for p in (3, 4)}, concurrent_clients=8,
                  public_payload_fields=fields, archive='development-008', archive_trace_sha256=trace,
                  verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), seconds=time.perf_counter()-start,
                  scope='Transport equivalence on completed development states; not a new strength test.')
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
