"""Loopback-only web adapter for the unchanged, certified CUDA planner."""
import argparse
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import threading
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
MAX_BODY = 32768


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cards(value, minimum, maximum):
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise ValueError('Invalid card list length')
    if any(type(c) is not int or not 1 <= c <= 104 for c in value) or len(set(value)) != len(value):
        raise ValueError('Invalid or duplicate card')
    return list(value)


def public_observation(value):
    """Accept only the player's hand and public state, never game/player objects."""
    allowed = {'hand', 'rows', 'seenCards', 'playerCount', 'deckSize', 'scores', 'seed', 'history'}
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError('Only the documented public observation fields are accepted')
    p = value.get('playerCount')
    if type(p) is not int or p not in (3, 4) or value.get('deckSize') != 104:
        raise ValueError('Certified champion requires 3 or 4 players and 104 cards')
    hand = cards(value.get('hand'), 1, 10)
    rows = value.get('rows')
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError('Exactly four rows are required')
    rows = [cards(row, 1, 5) for row in rows]
    flat = sum(rows, [])
    if len(set(flat)) != len(flat) or any(row != sorted(row) for row in rows):
        raise ValueError('Invalid public rows')
    seen = cards(value.get('seenCards'), 4, 104)
    if set(hand) & set(seen) or not set(flat) <= set(seen):
        raise ValueError('Hand and public cards are inconsistent')
    if len(seen) != 4 + p * (10 - len(hand)):
        raise ValueError('Public card count does not match the turn')
    scores = value.get('scores')
    if not isinstance(scores, list) or len(scores) != p or any(type(s) not in (int, float) or not math.isfinite(s) or not 0 <= s <= 1000 for s in scores):
        raise ValueError('Scores must be in player-relative order')
    seed = value.get('seed')
    if type(seed) is not int or not 0 <= seed <= 2**53 - 1:
        raise ValueError('A non-negative safe-integer seed is required')
    history = value.get('history', [])
    if not isinstance(history, list) or len(history) > 10 - len(hand):
        raise ValueError('Invalid public history')
    # The accepted configuration does not use history reweighting. Validate any
    # supplied history nonetheless, and do not pass arbitrary nested keys.
    clean_history = []
    for round in history:
        if not isinstance(round, dict) or set(round) != {'rows', 'seenCards', 'ownCard', 'opponentCards'}:
            raise ValueError('Invalid history fields')
        old_rows = round['rows']
        if not isinstance(old_rows, list) or len(old_rows) != 4:
            raise ValueError('Invalid historical rows')
        old_rows = [cards(row, 1, 5) for row in old_rows]
        old_seen = cards(round['seenCards'], 4, 104)
        played = cards([round['ownCard']] + cards(round['opponentCards'], p - 1, p - 1), p, p)
        if not set(old_seen + played) <= set(seen) or set(old_seen) & set(played):
            raise ValueError('Historical cards must already be public')
        clean_history.append(dict(rows=old_rows, seenCards=old_seen, ownCard=played[0], opponentCards=played[1:]))
    return dict(hand=hand, rows=rows, seenCards=seen, playerCount=p, deckSize=104, scores=list(scores), seed=seed, history=clean_history)


class Champion:
    def __init__(self):
        import progressive_campaign as campaign
        import torch
        state = load(ROOT / 'artifacts/progressive-upgrades/state.json')
        upgrade = state['accepted_upgrades'][-1]
        folder = ROOT / 'artifacts/progressive-upgrades' / upgrade['run']
        manifest = load(folder / 'manifest.json')
        campaign.verify(manifest, folder)
        # Archived planner imports a few shared current Python modules. Freeze
        # those too rather than silently accepting an edited training runtime.
        for name, digest in manifest['source_sha256'].items():
            if sha(ROOT / name) != digest:
                raise RuntimeError(f'Frozen runtime dependency changed: {name}')
        entry = upgrade['policy']
        original = load(ROOT / 'artifacts/small-player-exploration/evaluation-manifest.json')
        for model in original['entrants']:
            if model['id'] in ('ntw-adaptive-rl-v3b', 'ntw-adaptive-arena-v8-distill', 'ntw-champion'):
                if sha(model['path']) != model['sha256']:
                    raise RuntimeError('Frozen opponent model changed')
        if not torch.cuda.is_available():
            raise RuntimeError('正式冠军需要可用的 CUDA GPU；请使用 ntw-ai 环境启动')
        name = 'web_certified_planner_' + entry['runtime_sha256'][:16]
        spec = importlib.util.spec_from_file_location(name, entry['runtime'])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        self.planner = module.Planner(load(entry['path'])['params'], 'cpu')
        self.lock = threading.Lock()
        self.calls = 0
        self.identity = dict(app='ntw-champion', status='ready', policy=entry['id'],
                             certificate=upgrade['run'], config_sha256=entry['sha256'],
                             runtime_sha256=entry['runtime_sha256'], device=str(self.planner.device),
                             worlds=2048, players=[3, 4], deckSize=104, pid=os.getpid())

    def choose(self, observation):
        clean = public_observation(observation)
        if not self.lock.acquire(timeout=45):
            raise TimeoutError('AI 请求排队超时，请稍后重试')
        try:
            result = self.planner.choose(clean)
            self.calls += 1
        finally:
            self.lock.release()
        if result['card'] not in clean['hand']:
            raise RuntimeError('Planner returned an illegal card')
        candidates = result.get('candidates', [result['card']])
        utilities = result.get('utility', [0.0])
        evaluations = sorted([dict(card=c, utility=u) for c, u in zip(candidates, utilities)], key=lambda row: (row['utility'], row['card']))
        return dict(card=result['card'], evaluations=evaluations, seconds=result['seconds'],
                    model=self.identity['policy'], worlds=result.get('worlds', 2048),
                    runtime_sha256=self.identity['runtime_sha256'])


def handler_for(champion, static_directory=None):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(static_directory or ROOT / 'dist'), **kwargs)

        def json_response(self, status, data):
            body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def local_request(self):
            host = self.headers.get('Host', '').split(':')[0].lower()
            origin = urlsplit(self.headers.get('Origin', ''))
            return host in ('localhost', '127.0.0.1') and (not origin.netloc or origin.hostname in ('localhost', '127.0.0.1'))

        def do_GET(self):
            if not self.local_request():
                return self.json_response(403, dict(error='Only local browser requests are supported'))
            if self.path == '/api/ai/health':
                return self.json_response(200, dict(champion.identity, calls=champion.calls))
            if static_directory is None or self.path.startswith('/api/'):
                return self.json_response(404, dict(error='Not found'))
            # Serve only the built website, with no directory listings.
            translated = Path(self.translate_path(self.path)).resolve()
            if not translated.is_relative_to(Path(static_directory).resolve()):
                return self.json_response(403, dict(error='Not found'))
            if translated.is_dir() and not (translated / 'index.html').is_file():
                return self.json_response(404, dict(error='Not found'))
            super().do_GET()

        def do_POST(self):
            # Consume a bounded request before rejecting it. On Windows, closing
            # a socket with unread body bytes can reset it before the client sees
            # the JSON error response.
            try:
                length = int(self.headers.get('Content-Length', '0'))
            except ValueError:
                return self.json_response(400, dict(error='Invalid request size'))
            if not 0 < length <= MAX_BODY:
                return self.json_response(413, dict(error='Invalid request size'))
            self.connection.settimeout(10)
            try:
                body = self.rfile.read(length)
            except TimeoutError:
                return self.json_response(408, dict(error='Request body timeout'))
            if not self.local_request():
                return self.json_response(403, dict(error='Only local browser requests are supported'))
            if self.path != '/api/ai/choose':
                return self.json_response(404, dict(error='Not found'))
            if self.headers.get_content_type() != 'application/json':
                return self.json_response(415, dict(error='Use application/json'))
            try:
                data = json.loads(body)
                result = champion.choose(data)
                self.json_response(200, result)
            except (ValueError, TypeError, KeyError) as error:
                self.json_response(400, dict(error=str(error)))
            except TimeoutError as error:
                self.json_response(503, dict(error=str(error)))
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception:
                self.json_response(500, dict(error='冠军推理失败，请检查本机服务日志'))

        def log_message(self, format, *args):
            # Do not log hands or request bodies.
            if args and '/api/ai/health' in str(args[0]):
                return
            super().log_message(format, *args)
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--static', action='store_true', help='Also serve the production dist/ website')
    args = parser.parse_args()
    if args.static and not (ROOT / 'dist/index.html').exists():
        parser.error('Run npm run build before npm start')
    os.chdir(ROOT)
    champion = Champion()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(champion, ROOT / 'dist' if args.static else None))
    print(json.dumps(dict(champion.identity, url=f'http://127.0.0.1:{args.port}')), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
