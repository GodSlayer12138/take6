"""Predeclared, resumable take6 evaluation. Never edits training or model files."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
import urllib.request
import numpy as np
import torch
from take6_adapter import ROOT, UPSTREAM, Take6Actor, sha
from verify_take6 import audit_native, upstream_classes
sys.path.insert(0, str(ROOT/'training'))
import run_local_tournament as arena
import reproduce_eth_dirv as eth
from evaluate_small_exploration import ExplorationBridge
from report_local_tournament import replay

OUT = ROOT/'artifacts/take6-evaluation'
URL = 'http://127.0.0.1:8765/api/ai'
CHAMPION = 'distill2048-specialist2048'
MODES = {
    'four104_project': dict(players=4, deck=104, rules='project', deals=128,
        entrants=[CHAMPION, 'take6-4', 'dirv-10000', 'mcs'], primary=True),
    'four104_take6_rules': dict(players=4, deck=104, rules='take6', deals=64,
        entrants=[CHAMPION, 'take6-4', 'dirv-10000', 'mcs'], primary=False),
    'two24_native': dict(players=2, deck=24, rules='take6', deals=128,
        entrants=['web-adaptive2', 'take6-2'], primary=False),
    'two104_transfer': dict(players=2, deck=104, rules='project', deals=128,
        entrants=['v6', 'take6-2'], primary=False),
}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def http(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(URL+path, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=90) as response:
        return json.load(response)


def seed_for(mode, block):
    # The sensitivity cohort deliberately reuses the primary cohort's first 64 deals.
    domain = 'four104_project' if mode == 'four104_take6_rules' else mode
    text = f'ntw-take6-20260915-v1:{domain}:{block}'
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], 'little')


class NativeTable(arena.ArenaTable):
    def step(self, actions):
        if len(actions) != len(self.hands) or len(set(actions)) != len(actions) or any(c not in h for c,h in zip(actions,self.hands)):
            raise ValueError('Illegal simultaneous actions')
        rewards = np.zeros(len(actions), dtype=np.float32)
        for seat in sorted(range(len(actions)), key=lambda i: actions[i]):
            card = int(actions[seat])
            eligible = [i for i,row in enumerate(self.rows) if row[-1] < card]
            index = max(eligible, key=lambda i:self.rows[i][-1]) if eligible else min(range(4), key=lambda i:(int(eth.POINTS[self.rows[i]].sum()), -self.rows[i][-1]))
            if not eligible or len(self.rows[index]) == 5:
                rewards[seat] = -eth.POINTS[self.rows[index]].sum()
                self.rows[index] = [card]
            else:
                self.rows[index].append(card)
            self.hands[seat].remove(card)
        return rewards


def verify(protocol):
    for name, digest in protocol['source_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Frozen source/weight changed: '+name)
    health = http('/health')
    for key in ('policy', 'config_sha256', 'runtime_sha256', 'worlds', 'device'):
        assert health[key] == protocol['champion_health'][key], key


def prepare():
    path = OUT/'protocol.json'
    if path.exists():
        protocol = load(path)
        verify(protocol)
        return protocol
    verification = load(OUT/'inference-verification.json')
    assert verification['status'] == 'passed'
    old = load(ROOT/'artifacts/small-player-exploration/evaluation-manifest.json')
    by_id = {e['id']: e for e in old['entrants']}
    entries = [by_id[e] for e in ('v6', 'dirv-10000', 'mcs')]
    entries.append(dict(id='web-adaptive2', kind='strategy', strategy='neural_hybrid'))
    state = load(ROOT/'artifacts/progressive-upgrades/state.json')
    upgrade = state['accepted_upgrades'][-1]
    cert = ROOT/'artifacts/progressive-upgrades'/upgrade['run']/'manifest.json'
    frozen = {**old['source_sha256'], **load(cert)['source_sha256']}
    extra = [ROOT/name for name in frozen]
    extra += [Path(e['path']) for e in entries if 'path' in e]
    extra += [Path(upgrade['policy']['path']), Path(upgrade['policy']['runtime']), cert,
        ROOT/'artifacts/progressive-upgrades/state.json', ROOT/'server/champion_api.py',
        ROOT/'tools/evaluate_take6.py', ROOT/'tools/take6_adapter.py', ROOT/'tools/verify_take6.py',
        OUT/'inference-verification.json', UPSTREAM/'provenance.json',
        ROOT/'external/take6-reference/ray-2.0.0-fcnet.py']
    extra += [UPSTREAM/f['path'] for f in load(UPSTREAM/'provenance.json')['files']]
    health = http('/health')
    assert health['policy'] == CHAMPION and health['worlds'] == 2048 and health['device'].startswith('cuda')
    protocol = dict(version=1, frozen_at=datetime.now().astimezone().isoformat(timespec='seconds'),
        modes=MODES, planned_games=sum(m['players']*m['deals'] for m in MODES.values()), entrants=entries,
        take6_models=verification['models'], champion_health=health,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(extra))},
        inference='take6 original weights, FP32 PyTorch CPU, categorical sampling as upstream interactive script. No fine-tuning or model selection on these games.',
        seeds='SHA256 first 4 bytes little endian of ntw-take6-20260915-v1:MODE:BLOCK; sensitivity shares first 64 primary deals. All seat rotations of each deal.',
        rng='NumPy default_rng for uniform deals; separate per-game RandomState for take6 categorical choices; frozen integer seeds for other policies.',
        rules='Ten simultaneous turns, positive bullheads, fractional first-place ties. Project forced row: min cost, shortest, lowest index. take6 forced row: min cost, highest tail.',
        native_sensitivity='Only the host forced-row tie convention changes. NTW/MCS internal planning remains frozen project code; DirV simulations use the host convention. This tests deployment under the alternative convention, not retrained/native NTW.',
        two_player_scope='24 cards compares take6 expert with web fallback neural_hybrid at samples=18. 104 cards compares with V6; expert input is false and take6 is transfer. No 3-player or 5-player weights evaluated.',
        budgets={'champion_worlds_per_candidate':2048,'dirv_simulations':200,'mcs_total_rollouts':100,'web_adaptive2_samples':18,'take6_actor_forwards_per_decision':1},
        statistics='Primary contrast: champion minus take6 fractional first-place share in four104_project. 95% percentile paired bootstrap over independent deals (all rotations averaged), 20000 resamples. Other cohorts and same-table scores descriptive/supplementary with unadjusted 95% intervals. No optional stopping or pooled cross-mode ranking.',
        timing='take6 and DirV measured in worker; JS worker reports compute time; champion uses returned server seconds excluding client queue. Concurrent heterogeneous execution is not a controlled latency or equal-compute benchmark.',
        resume='One flushed JSONL line per complete deal block. Fail on frozen hash changes; completed blocks never rerun or selected by score.')
    dump(path, protocol)
    return protocol


def initialize(path):
    global PROTOCOL, BRIDGE, ACTORS, DIRV, CLASSES
    torch.set_num_threads(1)
    PROTOCOL = load(path)
    BRIDGE = ExplorationBridge(path)
    ACTORS = {2:Take6Actor(2), 4:Take6Actor(4)}
    e = next(e for e in PROTOCOL['entrants'] if e['id'] == 'dirv-10000')
    DIRV = eth.load_agent(e['path'], e['simulations'], 'cpu')[0]
    CLASSES = upstream_classes()


def run_block(task):
    mode, block = task
    config = PROTOCOL['modes'][mode]
    p, deck, ids = config['players'], config['deck'], config['entrants']
    seed = seed_for(mode, block)
    cards = np.random.default_rng(seed).permutation(deck).tolist()
    initial_hands = [sorted(cards[i*10:i*10+10]) for i in range(p)]
    initial_rows = [[cards[-1-i]] for i in range(4)]
    table_type = NativeTable if config['rules'] == 'take6' else arena.ArenaTable
    eth.Table = table_type
    records = []
    for rotation in range(p):
        seats = ids[rotation:] + ids[:rotation]
        table = table_type(initial_rows, initial_hands)
        scores = np.zeros(p, dtype=np.float32)
        seen = [r[0] for r in initial_rows]
        history, trace = [], []
        timings = np.zeros(p)
        client_times = np.zeros(p)
        take6_seed = (seed + rotation*1000003 + seats.index(f'take6-{p}')*65537) % 2**32
        take6_rng = np.random.RandomState(take6_seed)
        if 'dirv-10000' in seats:
            seat = seats.index('dirv-10000')
            DIRV.begin_game((seed + rotation*1000003 + seat*65537) % 2**32)
            DIRV.available = set(range(deck))
            DIRV.decision_seconds.clear()
        for turn in range(10):
            actions = [None]*p
            js_items, js_seats = [], []
            states = table.states()
            for seat, eid in enumerate(seats):
                action_seed = seed + rotation*1000003 + turn*104729 + seat*8191
                started = time.perf_counter()
                if eid.startswith('take6-'):
                    actions[seat] = ACTORS[p].choose([c+1 for c in table.hands[seat]],
                        [[c+1 for c in r] for r in table.rows], [c+1 for c in seen],
                        (-scores).tolist(), seat, deck == p*10+4, take6_rng)-1
                    timings[seat] += time.perf_counter()-started
                elif eid == 'dirv-10000':
                    actions[seat] = int(DIRV.select(states[seat], table.hands[seat].copy()))
                    timings[seat] += time.perf_counter()-started
                else:
                    observation = arena.make_observation(table, scores, history, seen, seat, action_seed)
                    observation['deckSize'] = deck
                    if eid == CHAMPION:
                        payload = {k:observation[k] for k in ('hand','rows','seenCards','playerCount','deckSize','scores','seed')}
                        result = http('/choose', payload)
                        assert result['model'] == CHAMPION and result['worlds'] == 2048
                        assert result['runtime_sha256'] == PROTOCOL['champion_health']['runtime_sha256']
                        actions[seat] = result['card']-1
                        timings[seat] += result['seconds']
                        client_times[seat] += time.perf_counter()-started
                    else:
                        js_items.append(dict(id=eid, observation=observation))
                        js_seats.append(seat)
            if js_items:
                for seat, result in zip(js_seats, BRIDGE.call(dict(type='actions',items=js_items))):
                    actions[seat] = result['card']-1
                    timings[seat] += result['seconds']
            history.append(dict(rows=[[c+1 for c in r] for r in table.rows], seen=[c+1 for c in seen], cards=[c+1 for c in actions]))
            scores += table.step(actions)
            seen.extend(actions)
            trace.append(actions)
        penalties = (-scores).tolist()
        minimum = min(penalties)
        record = dict(mode=mode, block=block, deal_seed=seed, rotation=rotation, seats=seats,
            bullheads=penalties, shares=[1/penalties.count(minimum) if s == minimum else 0 for s in penalties],
            decision_seconds=timings.tolist(), champion_client_seconds=client_times.tolist(),
            take6_sampling_seed=take6_seed, actions=trace)
        if config['rules'] == 'take6':
            audit_native(record, p, deck, CLASSES)
        else:
            replay(record, p, deck)
        records.append(record)
    return records


def summarize(protocol):
    verify(protocol)
    grouped = {mode:{} for mode in protocol['modes']}
    classes = upstream_classes()
    total = 0
    for line in (OUT/'blocks.jsonl').read_text().splitlines():
        records = json.loads(line)
        mode, block = records[0]['mode'], records[0]['block']
        config = protocol['modes'][mode]
        p, deck, ids = config['players'], config['deck'], config['entrants']
        assert 0 <= block < config['deals'] and block not in grouped[mode] and len(records) == p
        for rotation, record in enumerate(records):
            assert record['rotation'] == rotation and record['mode'] == mode and record['block'] == block
            assert record['deal_seed'] == seed_for(mode, block)
            assert record['seats'] == ids[rotation:]+ids[:rotation]
            if config['rules'] == 'take6':
                audit_native(record, p, deck, classes)
            else:
                replay(record, p, deck)
            total += 1
        grouped[mode][block] = records
    summaries = {}
    for mode, config in protocol['modes'].items():
        p, ids = config['players'], config['entrants']
        blocks = [grouped[mode][i] for i in range(config['deals'])]
        shares = np.asarray([[np.mean([r['shares'][r['seats'].index(eid)] for r in records]) for eid in ids] for records in blocks])
        n = len(blocks)
        rng = np.random.default_rng(seed_for(mode, 999999))
        weights = rng.multinomial(n, np.full(n, 1/n), size=20000)/n
        boot = weights @ shares
        ci = np.quantile(boot, [.025, .975], axis=0)
        rankings = []
        for j, eid in enumerate(ids):
            penalties = [r['bullheads'][r['seats'].index(eid)] for b in blocks for r in b]
            seconds = [r['decision_seconds'][r['seats'].index(eid)] for b in blocks for r in b]
            rankings.append(dict(id=eid, games=n*p, first_share=float(shares[:,j].mean()), ci95=ci[:,j].tolist(),
                mean_bullheads=float(np.mean(penalties)), mean_decision_ms=float(np.mean(seconds)*100),
                seat_counts=[sum(r['seats'][i] == eid for b in blocks for r in b) for i in range(p)]))
        take6_id = f'take6-{p}'
        ours_index, theirs_index = 0, ids.index(take6_id)
        pair = [[float(r['bullheads'][r['seats'].index(ids[0])] < r['bullheads'][r['seats'].index(take6_id)])
            + .5*float(r['bullheads'][r['seats'].index(ids[0])] == r['bullheads'][r['seats'].index(take6_id)]) for r in b] for b in blocks]
        pairs = np.mean(pair, axis=1)
        summaries[mode] = dict(independent_deals=n, games=n*p, ranking=sorted(rankings,key=lambda r:-r['first_share']),
            ours=ids[0], take6=take6_id, delta_first_share_pp=float((shares[:,ours_index]-shares[:,theirs_index]).mean()*100),
            delta_ci95_pp=(np.quantile(boot[:,ours_index]-boot[:,theirs_index],[.025,.975])*100).tolist(),
            ours_same_table_score_share=float(pairs.mean()), same_table_score_ci95=np.quantile(weights@pairs,[.025,.975]).tolist())
    assert total == protocol['planned_games']
    result = dict(completed_at=datetime.now().astimezone().isoformat(timespec='seconds'), audited_games=total,
        protocol_sha256=sha(OUT/'protocol.json'), trace_sha256=sha(OUT/'blocks.jsonl'),
        independent_deals_total=384, paired_sensitivity_deals=64, summaries=summaries,
        audit='Every action legal, all scores and first shares replayed independently, every scheduled deal and complete seat rotation present, frozen dependencies verified.')
    dump(OUT/'results.json', result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare','smoke','run','summarize'])
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    protocol = prepare()
    if args.command == 'prepare':
        print(json.dumps(dict(status='frozen', games=protocol['planned_games'], protocol_sha256=sha(OUT/'protocol.json'))))
        return
    if args.command == 'smoke':
        initialize(str(OUT/'protocol.json'))
        started = time.perf_counter()
        for mode in protocol['modes']:
            records = run_block((mode, 1000000))
            print(json.dumps(dict(mode=mode, legal_replayed_games=len(records), seconds=round(time.perf_counter()-started,2))), flush=True)
        BRIDGE.close()
        dump(OUT/'smoke.json',dict(status='passed', games=12, no_heldout_seeds=True, protocol_sha256=sha(OUT/'protocol.json')))
        return
    if args.command == 'summarize':
        summarize(protocol)
        return
    assert load(OUT/'smoke.json')['status'] == 'passed'
    completed = set()
    trace = OUT/'blocks.jsonl'
    if trace.exists():
        for line in trace.read_text().splitlines():
            records = json.loads(line)
            key = records[0]['mode'], records[0]['block']
            assert key not in completed and len(records) == protocol['modes'][key[0]]['players']
            completed.add(key)
    tasks = [(mode,i) for mode,c in protocol['modes'].items() for i in range(c['deals']) if (mode,i) not in completed]
    games = sum(protocol['modes'][mode]['players'] for mode,_ in completed)
    started = time.perf_counter()
    print(json.dumps(dict(status='running', completed_games=games, planned_games=protocol['planned_games'], remaining_blocks=len(tasks))), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'), initializer=initialize, initargs=(str(OUT/'protocol.json'),)) as pool:
        with trace.open('a',encoding='utf-8') as log:
            for records in pool.map(run_block, tasks, chunksize=1):
                log.write(json.dumps(records)+'\n')
                log.flush()
                games += len(records)
                if games % 16 == 0:
                    print(json.dumps(dict(mode=records[0]['mode'], completed_games=games, planned_games=protocol['planned_games'], elapsed_seconds=round(time.perf_counter()-started,1))), flush=True)
    summarize(protocol)


if __name__ == '__main__':
    main()
