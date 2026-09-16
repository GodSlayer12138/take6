"""Boundary and HTTP tests without loading GPU weights."""
import copy
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import json
import threading
import unittest
from champion_api import handler_for, public_observation


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


if __name__ == '__main__':
    unittest.main()
