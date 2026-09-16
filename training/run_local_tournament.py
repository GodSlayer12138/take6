"""Frozen local-model tournament; balanced random groups and paired seat rotations."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import atexit
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import re
import subprocess
import sys
import time
import numpy as np
import torch
import reproduce_eth_dirv as eth

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/local-tournament'
NODE = r'D:\Programs\nodejs\node.exe'

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def layers(state, prefix):
    indices = sorted({int(k.split('.')[1]) for k in state if k.startswith(prefix + '.')})
    return [{'weight': state[f'{prefix}.{i}.weight'].tolist(), 'bias': state[f'{prefix}.{i}.bias'].tolist()} for i in indices]

def prepare():
    if (OUT / 'manifest.json').exists():
        raise FileExistsError('Manifest already frozen')
    entrants, excluded, aliases = [], [], []
    formats = {'ntw-neural-v1', 'ntw-policy-v2', 'ntw-policy-v2-compact', 'ntw-compact-opponent-v1',
               'ntw-contextual-opponent-v1', 'ntw-contextual-opponent-v2', 'ntw-counterfactual-ensemble', 'ntw-counterfactual-gated'}
    paths = sorted((ROOT / 'src/game/models').glob('*.json')) + sorted((ROOT / 'artifacts/models').glob('*.json'))
    # Export final PyTorch-only candidates without changing any existing model.
    for path in sorted((ROOT / 'artifacts/models').glob('*.pt')):
        if path.with_suffix('.json').exists() or re.search(r'-(?:u\d+|e\d+)$', path.stem) or any(x in path.stem for x in ('smoke', 'profile', 'export')):
            continue
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        state = checkpoint['state_dict']
        if 'layers.0.weight' in state:
            model = dict(format='ntw-neural-v1', featureVersion=1, featureSize=270, targetScale=10.0, activation='relu', layers=layers(state, 'layers'))
        elif 'state_layers.0.weight' in state:
            model = dict(format='ntw-policy-v2', featureVersion=2, stateSize=420, actionSize=17, cards=54,
                         stateLayers=layers(state, 'state_layers'), actionLayers=layers(state, 'action_layers'), valueLayers=layers(state, 'value_layers'))
        else:
            excluded.append({'path': str(path.relative_to(ROOT)), 'reason': 'No supported standalone inference export'})
            continue
        model['exportedFrom'] = str(path.relative_to(ROOT))
        model['sourceSha256'] = sha(path)
        target = OUT / 'exports' / (path.stem + '.json')
        dump(target, model)
        paths.append(target)
    fingerprints = {}
    inference_keys = ['format','featureVersion','featureSize','targetScale','activation','stateSize','actionSize','cards',
                      'layers','stateLayers','actionLayers','valueLayers','models','modelPaths','weights','gateLayers']
    def canonical(model):
        def strip_source(value):
            if isinstance(value,list):return [strip_source(x) for x in value]
            if isinstance(value,dict):return {k:strip_source(v) for k,v in value.items() if k!='source'}
            return value
        result=strip_source({k:model[k] for k in inference_keys if k in model})
        if 'models' in model:result['models']=[canonical(m) for m in model['models']]
        return result
    for path in paths:
        reason = None
        if re.search(r'-(?:u\d+|e\d+)$', path.stem): reason = 'Historical intermediate training checkpoint'
        elif any(word in path.stem for word in ('smoke', 'profile', 'pilot')): reason = 'Smoke/profiling/pilot artifact'
        model = json.loads(path.read_text(encoding='utf-8'))
        if model.get('format') not in formats: reason = 'Configuration or auxiliary model without a standalone card policy'
        if path.stem.startswith(('ntw-value-', 'ntw-winvalue-')): reason = 'Value-only training; policy head is not the resulting trained agent'
        if reason:
            excluded.append({'path': str(path.relative_to(ROOT)), 'reason': reason})
            continue
        fingerprint = hashlib.sha256(json.dumps(canonical(model), sort_keys=True).encode()).hexdigest()
        if fingerprint in fingerprints:
            aliases.append({'path': str(path.relative_to(ROOT)), 'same_as': fingerprints[fingerprint]})
            continue
        entry_id = path.stem
        if any(e['id'] == entry_id for e in entrants): raise ValueError('Duplicate model name')
        fingerprints[fingerprint] = entry_id
        entrants.append(dict(id=entry_id, name=entry_id, kind='model', path=str(path), sha256=sha(path),
                             format=model['format'], modes=['adaptive5','classic4'] if model['format']=='ntw-neural-v1' else ['adaptive5'],
                             inference='Greedy standalone card policy; no search', exported_from=model.get('exportedFrom')))
    strategies = [('final-agent','牧场终极冠军（54 张主版本）','final',['adaptive5']),
                  ('default-classic','当前默认 AI（104 张路由）','neural_hybrid',['classic4']),
                  ('hybrid-rl','终局强化＋搜索','neural_hybrid_rl',['adaptive5','classic4']),
                  ('hybrid-ensemble','双模型共识＋搜索','neural_hybrid_ensemble',['adaptive5','classic4']),
                  ('hybrid-specialist','五人专用＋搜索','neural_hybrid_spec',['adaptive5']),
                  ('champion-search','牧场冠军 v2 搜索','champion',['adaptive5','classic4']),
                  ('mcs','公开 MCS 适配版','external_mcs',['adaptive5','classic4']),
                  ('random','随机策略（校准基线）','random',['adaptive5','classic4'])]
    for eid, name, strategy, modes in strategies:
        entrants.append(dict(id=eid,name=name,kind='strategy',strategy=strategy,modes=modes))
    for eid,name,folder,sims in [('alpha-2000','Alpha 复现版 2,000 局','paper-alpha',50),
                                ('dirv-10000','DirV 复现版 10,000 局','paper-dirv',200),
                                ('dirv-6200','DirV 等时版 6,200 局','paper-dirv-time-matched',200)]:
        path = ROOT / 'artifacts/eth-reproduction' / folder / 'checkpoint.pt'
        entrants.append(dict(id=eid,name=name,kind='eth',path=str(path),sha256=sha(path),simulations=sims,
                             modes=['adaptive5','classic4'],inference='Frozen agent index 0; adaptive5 is transfer without retraining'))
    source_paths = list((ROOT/'src/game').glob('*.js')) + list((ROOT/'src/game/models').glob('*.json')) + list((ROOT/'artifacts/models').glob('*.json')) + [Path(__file__),ROOT/'scripts/local-tournament-worker.mjs',ROOT/'training/reproduce_eth_dirv.py']
    manifest = dict(version=1, entrants=entrants,excluded=excluded,aliases=aliases,
                    source_sha256={str(p.relative_to(ROOT)):sha(p) for p in source_paths},
                    rules='10 turns, four rows, lowest bullheads wins. Forced row: least cost, shortest on cost tie, first index on remaining tie.',
                    information='Own hand, public rows/history/scores only; opposing policy identities never supplied.',
                    budgets={'project_search_samples':18,'mcs_samples_per_card':18,'alpha_root_simulations':50,'dirv_root_simulations':200},
                    ranking='Fractional first-place share, ties divided equally. Fixed heterogeneous deployed budgets; not an equal-compute comparison.',
                    sampling='Every epoch: shuffled cyclic groups; every entrant appears in P groups, each with P seat rotations of one deal. Equal games and exact seat balance.',
                    uncertainty='95% percentile bootstrap over independent deal blocks for each entrant, conditional on frozen models and opponent schedule. Close point ranks are not proven differences.',
                    planned_epochs={'adaptive5':80,'classic4':128}, planned_minimum_games=2000,
                    selection='Final exports/branch versions and supported standalone policies; historical numbered snapshots and smoke artifacts excluded, identical weights deduplicated.')
    dump(OUT/'manifest.json',manifest)
    print(json.dumps({'entrants':len(entrants),'modes':{m:sum(m in e['modes'] for e in entrants) for m in ('adaptive5','classic4')},'aliases':len(aliases),'excluded':len(excluded)}),flush=True)

class ArenaTable(eth.Table):
    def step(self, actions):
        if len(actions) != len(self.hands) or len(set(actions)) != len(actions) or any(c not in h for c,h in zip(actions,self.hands)):
            raise ValueError('Illegal simultaneous actions')
        rewards=np.zeros(len(actions),dtype=np.float32)
        for p in sorted(range(len(actions)), key=lambda p: actions[p]):
            card=int(actions[p]); eligible=[i for i,r in enumerate(self.rows) if r[-1]<card]
            row_id=max(eligible,key=lambda i:self.rows[i][-1]) if eligible else min(range(4),key=lambda i:(int(eth.POINTS[self.rows[i]].sum()),len(self.rows[i]),i))
            if not eligible or len(self.rows[row_id])==5:
                rewards[p]=-eth.POINTS[self.rows[row_id]].sum();self.rows[row_id]=[card]
            else:self.rows[row_id].append(card)
            self.hands[p].remove(card)
        return rewards

class Bridge:
    def __init__(self, manifest):
        self.process=subprocess.Popen([NODE,str(ROOT/'scripts/local-tournament-worker.mjs'),str(manifest)],cwd=ROOT,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8',bufsize=1,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        atexit.register(self.close)
        assert self.call({'type':'ping'})['ready']
    def call(self, request):
        self.process.stdin.write(json.dumps(request)+'\n');self.process.stdin.flush()
        line=self.process.stdout.readline()
        if not line:raise RuntimeError('Node worker exited: '+self.process.stderr.read())
        result=json.loads(line)
        if 'error' in result:raise RuntimeError(result['error'])
        return result['result']
    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:self.process.terminate()

def initialize(manifest_path):
    global MANIFEST, ENTRIES, BRIDGE, ETH_AGENTS
    torch.set_num_threads(1)
    MANIFEST=json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    ENTRIES={e['id']:e for e in MANIFEST['entrants']}
    BRIDGE=Bridge(manifest_path)
    ETH_AGENTS={}
    eth.Table=ArenaTable  # Uniform forced-row convention inside ETH simulations too.
    for e in MANIFEST['entrants']:
        if e['kind']=='eth': ETH_AGENTS[e['id']]=eth.load_agent(e['path'],e['simulations'],'cpu')[0]

def make_observation(table, scores, history, seen, seat, seed):
    p=len(table.hands);order=[(seat+i)%p for i in range(p)]
    return dict(rows=[[c+1 for c in r] for r in table.rows],hand=[c+1 for c in table.hands[seat]],seenCards=[c+1 for c in seen],
                deckSize=54 if p==5 else 104,playerCount=p,playerSeat=seat,scores=[float(-scores[s]) for s in order],
                history=[dict(rows=h['rows'],seenCards=h['seen'],ownCard=h['cards'][seat],opponentCards=[h['cards'][s] for s in order[1:]]) for h in history],
                playedCards=[[h['cards'][s] for h in history] for s in order],samples=18,seed=int(seed))

def run_block(task):
    mode,block,epoch,ids,seed=task
    p=len(ids);deck=54 if mode=='adaptive5' else 104
    rng=np.random.default_rng(seed);cards=rng.permutation(deck).tolist()
    initial_hands=[sorted(cards[s*10:(s+1)*10]) for s in range(p)]
    initial_rows=[[cards[-1-i]] for i in range(4)]
    records=[]
    for rotation in range(p):
        seats=ids[rotation:]+ids[:rotation]
        table=ArenaTable(initial_rows,initial_hands);scores=np.zeros(p,dtype=np.float32)
        seen=[r[0] for r in initial_rows];history=[];timings=np.zeros(p)
        for seat,eid in enumerate(seats):
            if eid in ETH_AGENTS:
                ETH_AGENTS[eid].begin_game(seed+rotation*1000003+seat*65537)
                ETH_AGENTS[eid].available=set(range(deck))
                ETH_AGENTS[eid].decision_seconds.clear()
        trace=[]
        for turn in range(10):
            states=table.states();actions=[None]*p;js_items=[];js_seats=[]
            for seat,eid in enumerate(seats):
                if eid in ETH_AGENTS:
                    started=time.perf_counter()
                    actions[seat]=int(ETH_AGENTS[eid].select(states[seat],table.hands[seat].copy()))
                    timings[seat]+=time.perf_counter()-started
                else:
                    js_items.append({'id':eid,'observation':make_observation(table,scores,history,seen,seat,seed+rotation*1000003+turn*104729+seat*8191)})
                    js_seats.append(seat)
            if js_items:
                for seat,result in zip(js_seats,BRIDGE.call({'type':'actions','items':js_items})):
                    actions[seat]=result['card']-1;timings[seat]+=result['seconds']
            history.append({'rows':[[c+1 for c in r] for r in table.rows],'seen':[c+1 for c in seen],'cards':[c+1 for c in actions]})
            scores+=table.step(actions);seen.extend(actions);trace.append(actions)
        bullheads=(-scores).tolist();minimum=min(bullheads);winners=bullheads.count(minimum)
        records.append(dict(mode=mode,block=block,epoch=epoch,deal_seed=seed,rotation=rotation,seats=seats,bullheads=bullheads,
                            shares=[1/winners if s==minimum else 0 for s in bullheads],decision_seconds=timings.tolist(),actions=trace))
    return records

def schedule(mode, epochs, seed, entries):
    ids=[e['id'] for e in entries if mode in e['modes']];p=5 if mode=='adaptive5' else 4
    rng=np.random.default_rng(seed)
    for epoch in range(epochs):
        order=rng.permutation(ids).tolist()
        for i in range(len(ids)):
            group=[order[(i+j)%len(ids)] for j in range(p)]
            block=epoch*len(ids)+i
            yield mode,block,epoch,group,seed+100000000+block*104729

def verify_frozen(manifest):
    for path,expected in manifest['source_sha256'].items():
        if sha(ROOT/path)!=expected:raise ValueError('Changed frozen source: '+path)
    for entry in manifest['entrants']:
        if 'sha256' in entry and sha(entry['path'])!=entry['sha256']:raise ValueError('Changed frozen weights: '+entry['id'])

def run(args):
    path=OUT/'manifest.json';manifest=json.loads(path.read_text(encoding='utf-8'));verify_frozen(manifest)
    dest=OUT/args.output;dest.mkdir(parents=True,exist_ok=True)
    protocol=dict(mode=args.mode,epochs=args.epochs,seed=args.seed,workers=args.workers,manifest_sha256=sha(path))
    if (dest/'protocol.json').exists() and json.loads((dest/'protocol.json').read_text())!=protocol:raise ValueError('Resume protocol mismatch')
    dump(dest/'protocol.json',protocol)
    completed=set()
    if (dest/'blocks.jsonl').exists():
        for line in (dest/'blocks.jsonl').read_text(encoding='utf-8').splitlines():
            records=json.loads(line);completed.add(records[0]['block'])
    tasks=[t for t in schedule(args.mode,args.epochs,args.seed,manifest['entrants']) if t[1] not in completed]
    started=time.perf_counter();count=len(completed)
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(str(path),)) as executor:
        with (dest/'blocks.jsonl').open('a',encoding='utf-8') as log:
            for records in executor.map(run_block,tasks,chunksize=1):
                log.write(json.dumps(records,ensure_ascii=False)+'\n');log.flush();count+=1
                if count%10==0:
                    print(json.dumps({'mode':args.mode,'blocks':count,'total_blocks':len(tasks)+len(completed),'games':count*len(records),'seconds_this_run':round(time.perf_counter()-started,1)}),flush=True)
    summarize(dest,manifest)

def summarize(dest,manifest):
    protocol=json.loads((dest/'protocol.json').read_text());mode=protocol['mode'];p=5 if mode=='adaptive5' else 4
    entries={e['id']:e for e in manifest['entrants'] if mode in e['modes']}
    values={eid:[] for eid in entries};stats={eid:dict(games=0,wins=0.,bullheads=0.,rank=0.,seconds=0.,seats=[0]*p) for eid in entries}
    pair={eid:{} for eid in entries};blocks=set();games=0
    for line in (dest/'blocks.jsonl').read_text(encoding='utf-8').splitlines():
        records=json.loads(line);block=records[0]['block']
        if block in blocks or len(records)!=p:raise ValueError('Duplicate/incomplete block')
        blocks.add(block);group={eid:[] for eid in records[0]['seats']}
        for record in records:
            scores=record['bullheads'];minimum=min(scores);n=scores.count(minimum)
            expected=[1/n if s==minimum else 0 for s in scores]
            if expected!=record['shares']:raise ValueError('Incorrect stored victory share')
            for seat,eid in enumerate(record['seats']):
                s=stats[eid];s['games']+=1;s['wins']+=expected[seat];s['bullheads']+=scores[seat]
                s['rank']+=1+sum(x<scores[seat] for x in scores)+(scores.count(scores[seat])-1)/2
                s['seconds']+=record['decision_seconds'][seat];s['seats'][seat]+=1;group[eid].append(expected[seat])
                for j,other in enumerate(record['seats']):
                    if j==seat:continue
                    t=pair[eid].setdefault(other,dict(games=0,better=0,tied=0))
                    t['games']+=1;t['better']+=int(scores[seat]<scores[j]);t['tied']+=int(scores[seat]==scores[j])
            games+=1
        for eid,x in group.items():values[eid].append(float(np.mean(x)))
    rng=np.random.default_rng(protocol['seed']+999);ranking=[]
    for eid,s in stats.items():
        x=np.array(values[eid]);boot=np.empty(10000)
        for begin in range(0,10000,100):boot[begin:begin+100]=x[rng.integers(0,len(x),(100,len(x)))].mean(axis=1)
        ranking.append(dict(id=eid,name=entries[eid]['name'],games=s['games'],independent_blocks=len(x),win_rate=s['wins']/s['games'],
                            ci95=np.quantile(boot,[.025,.975]).tolist(),mean_bullheads=s['bullheads']/s['games'],mean_rank=s['rank']/s['games'],
                            mean_decision_ms=s['seconds']/s['games']/10*1000,seat_counts=s['seats']))
    ranking.sort(key=lambda e:(-e['win_rate'],e['mean_bullheads'],e['id']))
    for i,row in enumerate(ranking):row['rank']=i+1
    expected_games=protocol['epochs']*p*p
    assert len(blocks)==protocol['epochs']*len(entries)
    assert all(e['games']==expected_games and len(set(e['seat_counts']))==1 for e in ranking)
    summary=dict(mode=mode,total_games=games,entrants=len(entries),games_per_entrant=expected_games,independent_blocks=len(blocks),ranking=ranking,
                 audit='Victory shares recalculated; complete unique blocks; exact equal appearances and seat balance.',protocol=protocol)
    dump(dest/'summary.json',summary);dump(dest/'head-to-head.json',pair)
    title='五人 / 54 张' if p==5 else '四人 / 104 张'
    lines=[f'# 本地模型随机赛：{title}','',f'共 {len(entries)} 个参赛版本、{games:,} 局，每个版本 {expected_games:,} 局。并列第一平分胜利份额。','',
           '名次按夺冠份额点估计排列；95% 区间按发牌组重采样。接近的名次不代表已证明强弱差异。每组相同发牌轮换全部座位；各版本参赛次数完全相同。','',
           '搜索使用冻结的现有预算，纯策略直接出牌；这不是等算力比较。不给任何策略提供对手身份或真实手牌。ETH 模型在 54 张模式属于未重新训练的迁移测试。','',
           '| 名次 | 模型 | 胜率 | 95% 区间 | 平均牛头↓ | 每步 ms |','|---:|---|---:|---|---:|---:|']
    for row in ranking:
        lo,hi=row['ci95'];lines.append(f"| {row['rank']} | {row['name']} | {row['win_rate']*100:.2f}% | {lo*100:.2f}–{hi*100:.2f}% | {row['mean_bullheads']:.2f} | {row['mean_decision_ms']:.2f} |")
    (dest/'ranking.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'completed':mode,'games':games,'top10':ranking[:10]},ensure_ascii=False),flush=True)

def validate():
    path=OUT/'manifest.json';initialize(path)
    # Compare the unified Python resolver with the actual project engine, including row-cost ties.
    rng=np.random.default_rng(60816001);turns=0
    for p in (4,5):
        deck=104 if p==4 else 54
        for _ in range(15):
            cards=rng.permutation(deck).tolist();table=ArenaTable([[cards[-i-1]] for i in range(4)],[sorted(cards[i*10:(i+1)*10]) for i in range(p)])
            for turn in range(10):
                actions=[int(rng.choice(h)) for h in table.hands]
                js=BRIDGE.call({'type':'resolve','rows':[[c+1 for c in r] for r in table.rows],
                               'actions':[{'playerId':i,'card':c+1} for i,c in enumerate(actions)]})
                rewards=table.step(actions)
                assert [[c+1 for c in r] for r in table.rows]==js['rows']
                assert [-int(r) for r in rewards]==[js['penalties'][str(i)] for i in range(p)]
                turns+=1
    tie=ArenaTable([[0,1],[2],[4],[9]],[[3],[5],[6],[7]])
    # Explicit equal-cost rows of different lengths: row [1,2] costs 2, row [5] costs 2.
    tie=ArenaTable([[5,6],[4],[9],[10]],[[0],[1],[2],[3]])
    js=BRIDGE.call({'type':'resolve','rows':[[c+1 for c in r] for r in tie.rows],'actions':[{'playerId':i,'card':i+1} for i in range(4)]})
    tie.step([0,1,2,3]);assert [[c+1 for c in r] for r in tie.rows]==js['rows']
    # Every entrant receives at least a full legal game; no failed candidate may be silently skipped.
    for mode in ('adaptive5','classic4'):
        ids=[e['id'] for e in MANIFEST['entrants'] if mode in e['modes']];p=5 if mode=='adaptive5' else 4
        for i in range(0,len(ids),p):
            group=ids[i:i+p]
            group+= [eid for eid in ids if eid not in group][:p-len(group)]
            result=run_block((mode,i,0,group,60817001+i*104729))
            assert len(result)==p
            print(json.dumps({'validation':mode,'tested':min(i+p,len(ids)),'total':len(ids)}),flush=True)
    # Frozen randomness and weights: repeat a mixed ETH/project block exactly.
    task=('classic4',0,0,['alpha-2000','dirv-10000','ntw-champion','mcs'],60818001)
    first,second=run_block(task),run_block(task)
    for a,b in zip(first,second):
        for key in ('bullheads','actions','shares'):assert a[key]==b[key]
    verify_frozen(MANIFEST)
    dump(OUT/'validation.json',{'status':'passed','engine_equivalence_turns':turns,'forced_row_tie':True,
                              'every_entrant_legal_games':True,'deterministic_mixed_eth_project':True,'frozen_hashes':True})
    BRIDGE.close()

def main():
    if Path(sys.prefix).name.lower()!='ntw-ai':raise RuntimeError('Use ntw-ai environment')
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('prepare');sub.add_parser('validate')
    p=sub.add_parser('run');p.add_argument('--mode',choices=['adaptive5','classic4'],required=True);p.add_argument('--epochs',type=int,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--workers',type=int,default=6);p.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='validate':validate()
    else:run(args)

if __name__=='__main__':main()
