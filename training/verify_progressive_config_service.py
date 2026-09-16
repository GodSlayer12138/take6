"""Compare new frozen configurations with their serial direct inference on old development states.

Expected outputs and service checks run in separate processes, so the direct
inference allocator releases GPU memory before the shared service is started.
This checks execution equivalence, not playing strength.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import gc
import hashlib
import importlib.util
import json
from multiprocessing.connection import Client
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def observations(archive, policy):
    import numpy as np
    from small_player_env import SmallGame
    folder = ROOT / 'artifacts/progressive-upgrades' / archive
    assert load(folder / 'manifest.json')['purpose'] == 'development'
    assert (folder / 'results.json').exists()
    seen = set()
    cases = []
    for line in (folder / 'blocks.jsonl').read_text(encoding='utf-8').splitlines():
        block = json.loads(line)
        p = block['players']
        if p in seen:
            continue
        assert p in (3, 4)
        seen.add(p)
        records = [r for r in block['records'] if r['candidate'] == policy]
        assert len(records) == p
        for r in records:
            seat = r['seats'].index(policy)
            order = [(seat + i) % p for i in range(p)]
            game = SmallGame(1, p, r['deal_seed'])
            game.rows = game.rows[:, ::-1].copy()
            history = []
            for turn in range(10):
                obs = game.record(0, seat)
                obs.update(scores=game.scores[0, order].tolist(), history=list(history),
                           seed=int(r['deal_seed'] + r['rotation'] * 1000003 + turn * 104729 + seat * 8191))
                played = np.asarray(r['actions'][turn]) + 1
                cases.append(dict(observation=obs, archived_card=int(played[seat])))
                history.append(dict(rows=obs['rows'], seenCards=obs['seenCards'],
                                    ownCard=int(played[seat]), opponentCards=played[order[1:]].tolist()))
                game.step(played[None])
            np.testing.assert_array_equal(game.scores[0], r['bullheads'])
    assert len(cases) == 70 and seen == {3, 4}
    return cases, sha(folder / 'blocks.jsonl')


def prepare(args):
    import torch
    import progressive_campaign as campaign
    path = Path(args.manifest).resolve()
    manifest = load(path)
    campaign.verify(manifest, path.parent)
    protocol = manifest['preflight_protocol']
    assert protocol['script_sha256'] == sha(__file__)
    assert args.archive == protocol['archive'] and args.archive_policy == protocol['archive_policy']
    assert not Path(args.expected).exists(), 'Expected decisions are immutable'
    cases, trace_sha = observations(args.archive, args.archive_policy)
    outputs = {}
    for entry in manifest['entrants']:
        if entry['id'] not in manifest['versions']:
            continue
        name = 'direct_config_' + sha(entry['runtime'])[:16]
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, entry['runtime'])
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        model = sys.modules[name].Planner(load(entry['path'])['params'], 'cpu')
        cards = []
        for case in cases:
            card = int(model.choose(case['observation'])['card'])
            assert card in case['observation']['hand']
            if entry['id'] == 'reference':
                assert card == case['archived_card']
            cards.append(card)
        outputs[entry['id']] = cards
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps(dict(policy=entry['id'],direct_decisions=len(cards))), flush=True)
    assert set(outputs) == set(manifest['versions'])
    result = dict(manifest_sha256=sha(path), verifier_sha256=sha(__file__),
                  archive=args.archive, archive_policy=args.archive_policy,
                  archive_trace_sha256=trace_sha, cases=cases, expected_cards=outputs)
    dump(args.expected, result)
    print(json.dumps(dict(status='prepared',decisions=sum(map(len,outputs.values())),
                          expected_sha256=sha(args.expected))), flush=True)


def verify(args):
    manifest = load(args.manifest)
    expected = load(args.expected)
    state = load(args.state)
    digest = sha(args.manifest)
    protocol = manifest['preflight_protocol']
    assert protocol['script_sha256'] == sha(__file__) and args.clients == protocol['clients']
    assert len(expected['cases']) == 70 and set(expected['expected_cards']) == set(manifest['versions'])
    assert all(len(cards) == 70 for cards in expected['expected_cards'].values())
    assert expected['manifest_sha256'] == digest and digest in state['manifests']
    assert expected['verifier_sha256'] == sha(__file__)
    assert state['source_sha256'] == manifest['source_sha256']['training/progressive_inference_service.py']
    assert state['status'] == 'ready' and state['streams'] == manifest['preflight_protocol']['streams']
    jobs = [(policy, i, case['observation'], cards[i])
            for policy, cards in expected['expected_cards'].items()
            for i, case in enumerate(expected['cases'])]
    def check(job):
        policy, index, obs, card = job
        with Client(state['address'], family='AF_PIPE', authkey=bytes.fromhex(state['authkey'])) as conn:
            conn.send(dict(type='choose',manifest=digest,policy=policy,observation=obs))
            response = conn.recv()
        assert 'error' not in response, response.get('error')
        assert int(response['result']['card']) == card, (policy,index,card,response)
        return policy
    with ThreadPoolExecutor(max_workers=args.clients) as executor:
        checked = list(executor.map(check, jobs))
    result = dict(status='passed',observed_at=datetime.now().astimezone().isoformat(),
                  manifest_sha256=digest,expected_sha256=sha(args.expected),
                  verifier_sha256=sha(__file__),service_source_sha256=state['source_sha256'],
                  streams=state['streams'],clients=args.clients,decisions=len(checked),
                  decisions_by_policy={p:checked.count(p) for p in manifest['versions']},
                  scope='Every shared-service card matches serial direct inference on archived development observations; no new development or certificate outcomes are used.')
    dump(Path(args.expected).with_name('validation-result.json'), result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['prepare','verify'],required=True)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--expected',required=True)
    parser.add_argument('--archive',default='development-006')
    parser.add_argument('--archive-policy',default='reference')
    parser.add_argument('--state')
    parser.add_argument('--clients',type=int,default=8)
    args = parser.parse_args()
    if args.mode == 'verify' and not args.state:
        parser.error('--state is required for service verification')
    (prepare if args.mode == 'prepare' else verify)(args)
