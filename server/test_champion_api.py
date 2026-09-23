"""Boundary and HTTP tests without loading GPU weights."""
import copy
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import json
import hashlib
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from champion_api import handler_for, public_observation, verify_runtime


def initial():
    return dict(playerCount=3, deckSize=104, hand=list(range(1, 11)), rows=[[20], [30], [40], [50]],
                seenCards=[20, 30, 40, 50], scores=[0, 0, 0], seed=123)


class FakeChampion:
    identity = dict(app='ntw-champion', status='ready')
    calls = 0

    def choose(self, obs):
        clean = public_observation(obs)
        self.calls += 1
        return dict(card=clean['hand'][0], model='test')


class Boundaries(unittest.TestCase):
    def test_public_state(self):
        self.assertEqual(public_observation(initial())['hand'], list(range(1, 11)))
        for key, value in [('opponentHands', [[12]]), ('playerCount', 5), ('scores', [0, float('nan'), 0]),
                           ('hand', [20]), ('seed', -1), ('seenCards', [20, 30, 40, 50, 60])]:
            obs = initial()
            obs[key] = value
            with self.assertRaises(ValueError):
                public_observation(obs)

    def test_duplicate_rows_and_history_fields(self):
        obs = initial()
        obs['rows'] = [[20], [20], [40], [50]]
        with self.assertRaises(ValueError): public_observation(obs)
        obs = initial()
        obs['history'] = [dict(opponentHands=[[99]])]
        with self.assertRaises(ValueError): public_observation(obs)

    def test_http_contract_and_origin(self):
        fake = FakeChampion()
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(fake))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            def request(method, path, data=None, headers=None):
                conn = HTTPConnection('127.0.0.1', server.server_port, timeout=5)
                conn.request(method, path, body=json.dumps(data) if data else None,
                             headers={'Content-Type': 'application/json', **(headers or {})})
                response = conn.getresponse()
                result = response.status, json.loads(response.read())
                conn.close()
                return result
            self.assertEqual(request('GET', '/api/ai/health')[0], 200)
            self.assertEqual(request('POST', '/api/ai/choose', initial())[1]['card'], 1)
            self.assertEqual(request('POST', '/api/ai/choose', initial(), {'Origin': 'https://example.com'})[0], 403)
            self.assertEqual(request('POST', '/api/ai/choose', dict(initial(), opponentHands=[[80]]))[0], 400)
            self.assertEqual(request('GET', '/artifacts/models/ntw-final-v1.json')[0], 404)
            self.assertEqual(fake.calls, 1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


class RuntimeFiles(unittest.TestCase):
    def test_history_is_optional_but_runtime_and_opponents_remain_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'certificate'
            def record(name):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'original')
                return str(path), hashlib.sha256(b'original').hexdigest()
            config, digest = record('policy.json')
            runtime, _ = record('planner.py')
            checkpoint, _ = record('network.pt')
            source, _ = record('training/helper.py')
            snapshot, _ = record('certificate/source-snapshot/training/helper.py')
            shared, _ = record('training/shared.py')
            opponents = []
            for identifier in ('ntw-adaptive-rl-v3b', 'ntw-adaptive-arena-v8-distill', 'ntw-champion'):
                path, _ = record(identifier + '.json')
                opponents.append(dict(id=identifier, path=path, sha256=digest))
            manifest = dict(source_sha256={'training/helper.py': digest},
                            common_dependency_sha256={shared: digest, str(root / 'removed-history.json'): digest})
            policy = dict(path=config, sha256=digest, runtime=runtime, runtime_sha256=digest,
                          dependency_sha256={checkpoint: digest})
            exploration = dict(entrants=opponents)
            with patch('champion_api.ROOT', root):
                verify_runtime(manifest, folder, policy, exploration)
                for name in [config, runtime, checkpoint, source, snapshot, shared, *[m['path'] for m in opponents]]:
                    with self.subTest(file=name):
                        Path(name).write_bytes(b'changed')
                        with self.assertRaisesRegex(RuntimeError, 'Frozen runtime dependency changed'):
                            verify_runtime(manifest, folder, policy, exploration)
                        Path(name).write_bytes(b'original')


if __name__ == '__main__':
    unittest.main()
