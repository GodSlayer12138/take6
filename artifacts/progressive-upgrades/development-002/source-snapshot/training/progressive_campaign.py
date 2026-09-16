"""Persistent campaign: development screening and independently certified consecutive upgrades."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
import numpy as np
import run_local_tournament as arena
from evaluate_small_exploration import ExplorationBridge
from evaluate_small_iterations import OPPONENTS
from report_local_tournament import replay
from explore_small_strategies import ROOT,dump,sha

OUT=ROOT/'artifacts/progressive-upgrades'
PLAN=dict(scope='Only 3 and 4 players, fixed 104 cards; win means fractional first-place share.',
    target='Three consecutive accepted upgrades, each significantly better than its accepted predecessor at both player counts.',
    starting_baseline='Existing V5 symmetry strategy, the highest point estimate in prior 3/4-player paired tests.',
    alpha_total=.05,alpha_spending='For certificate attempt j>=1, alpha_j=.05/(j*(j+1)); split equally between two player counts. Sum across unlimited attempts <= .05.',
    certificate_deals_per_player_count=2048,bootstrap_replicates=200000,
    acceptance='Both one-sided multiplicity-adjusted paired deal-block bootstrap lower bounds must exceed zero and both observed improvements must be at least 0.5 percentage point.',
    minimum_observed_delta_pp=.5,
    independence='Checkpoint/config selection uses development games only. Each certificate freezes both compared policies and its complete sample size before fresh games; no early significance peeking or test reuse.',
    opponents=OPPONENTS,
    failed_attempts='A failure consumes its alpha allocation and never counts as an upgrade. Keep all records; the accepted predecessor remains unchanged.',
    scope_of_claim='Conditional on this opponent pool and these rules; search computation is reported, not forced equal.')

def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def initialize_campaign():
    plan=OUT/'protocol.json'
    if not plan.exists():dump(plan,PLAN)
    else:assert load(plan)==PLAN,'Do not silently change the campaign acceptance protocol'
    state=OUT/'state.json'
    if not state.exists():dump(state,dict(accepted_upgrades=[],certificate_attempts=[],active_baseline='v5',goal_complete=False))

class CampaignBridge(ExplorationBridge):
    def __init__(self,path):
        manifest=load(path);self.planners={}
        for entry in manifest['entrants']:
            if entry['kind']=='planner':
                name='planner_'+sha(entry['runtime'])[:16]
                if name not in sys.modules:
                    spec=importlib.util.spec_from_file_location(name,entry['runtime']);module=importlib.util.module_from_spec(spec)
                    sys.modules[name]=module;spec.loader.exec_module(module)
                self.planners[entry['id']]=sys.modules[name].Planner(load(entry['path'])['params'],'cpu')
        super().__init__(path)
    def call(self,request):
        if request['type']!='actions':return super().call(request)
        results=[None]*len(request['items']);remote=[];indices=[]
        for i,item in enumerate(request['items']):
            if item['id'] in self.planners:
                r=self.planners[item['id']].choose(item['observation']);results[i]={'card':r['card'],'seconds':r['seconds']}
            else:remote.append(item);indices.append(i)
        if remote:
            for i,result in zip(indices,super().call(dict(type='actions',items=remote))):results[i]=result
        return results

def initialize(path):
    arena.Bridge=CampaignBridge;arena.initialize(path)

def seed_for(namespace,p,block):
    return int.from_bytes(hashlib.sha256(f'ntw-progressive-20260909:{namespace}:{p}:{block}'.encode()).digest()[:4],'little')

def schedule(manifest):
    for p in (3,4):
        rng=np.random.default_rng(seed_for(manifest['namespace'],p,-1))
        for block in range(manifest['deals']):
            opponents=rng.choice(OPPONENTS,p-1,replace=False).tolist()
            yield p,block,opponents,seed_for(manifest['namespace'],p,block),manifest['versions']

def task_work(task):
    p,block,opponents,seed,versions=task;records=[]
    for version in versions:
        group=arena.run_block((f'classic{p}',block,0,[version]+opponents,seed))
        for record in group:record['candidate']=version
        records.extend(group)
    return dict(players=p,block=block,opponents=opponents,deal_seed=seed,records=records)

def prepare_run(name,configs,deals,purpose,reference=None):
    initialize_campaign();folder=OUT/name;manifest_path=folder/'manifest.json'
    if manifest_path.exists():return manifest_path
    prior=load(ROOT/'artifacts/small-player-exploration/evaluation-manifest.json')
    entries=[dict(e) for e in prior['entrants'] if e['id'] in OPPONENTS]
    if reference is None:
        reference=dict(next(e for e in prior['entrants'] if e['id']=='v5'))
        reference['id']='reference'
    else:reference=dict(reference,id='reference')
    entries.append(reference)
    snapshot=folder/'source-snapshot';snapshot.mkdir(parents=True,exist_ok=True)
    sources=['training/progressive_planner.py','training/progressive_torch_env.py','training/progressive_campaign.py','training/explore_small_strategies.py',
        'training/small_player_env.py','training/train_policy.py','training/run_local_tournament.py','training/report_local_tournament.py',
        'training/evaluate_small_exploration.py','training/evaluate_small_iterations.py',
        'scripts/small-exploration-worker.mjs','scripts/small-strategy-runtime.mjs']
    fingerprints={}
    for relative in sources:
        source=ROOT/relative;target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
        fingerprints[relative]=sha(source)
    runtime=str(snapshot/'training/progressive_planner.py')
    for candidate,params in configs.items():
        path=folder/(candidate+'.json');dump(path,dict(format='ntw-progressive-planner-v1',name=candidate,params=params))
        dependencies=[ROOT/'artifacts/models/ntw-adaptive5-v1.pt',ROOT/'artifacts/models/ntw-adaptive-rl-v3b.pt',
            ROOT/'artifacts/models/ntw-champion-v2.pt',ROOT/'artifacts/small-player-iterations/v4/model.pt']
        if params.get('opponents')=='actual-proxy':
            directory=ROOT/params.get('proxy_directory','artifacts/progressive-upgrades/opponent-proxies-v1')
            dependencies.extend(directory/(k+'.pt') for k in ('dirv-10000','alpha-2000','mcs','champion-search'))
        if params.get('continuation_checkpoint'):dependencies.append(ROOT/params['continuation_checkpoint'])
        if params.get('engine')=='torch':dependencies.append(snapshot/'training/progressive_torch_env.py')
        entries.append(dict(id=candidate,name=candidate,kind='planner',path=str(path),sha256=sha(path),runtime=runtime,runtime_sha256=sha(runtime),
            dependency_sha256={str(p):sha(p) for p in dependencies}))
    used=set()
    for archive in ('small-player-iterations','small-player-exploration'):
        for line in (ROOT/'artifacts'/archive/'heldout-blocks.jsonl').read_text().splitlines():
            b=json.loads(line)
            if b['mode'] in ('classic3','classic4'):used.add(b['deal_seed'])
    for other in OUT.glob('*/manifest.json'):
        prior_run=load(other)
        used.update(t[3] for t in schedule(prior_run))
    namespace=name;nonce=0
    while True:
        seeds=[seed_for(namespace,p,i) for p in (3,4) for i in range(deals)]
        if len(set(seeds))==len(seeds) and not used.intersection(seeds):break
        nonce+=1;namespace=f'{name}-seed-reservation-{nonce}'
    manifest=dict(name=name,purpose=purpose,namespace=name,deals=deals,versions=['reference']+list(configs),entrants=entries,
        source_sha256=fingerprints,protocol_sha256=sha(OUT/'protocol.json'),policy='Frozen settings; no changes during this run.',
        common_dependency_sha256={str(ROOT/path):value for path,value in prior['source_sha256'].items()},
        new_deal_seeds_disjoint_from_prior_tests_and_campaign_runs=True)
    manifest['namespace']=namespace
    dump(manifest_path,manifest);return manifest_path

def verify(manifest,folder):
    for e in manifest['entrants']:
        if 'sha256' in e:assert sha(e['path'])==e['sha256'],e['id']
        if 'runtime_sha256' in e:assert sha(e['runtime'])==e['runtime_sha256'],e['id']
        for path,digest in e.get('dependency_sha256',{}).items():assert sha(path)==digest,path
    for path,digest in manifest['source_sha256'].items():assert sha(folder/'source-snapshot'/path)==digest,path
    for path,digest in manifest.get('common_dependency_sha256',{}).items():assert sha(path)==digest,path

def summarize(path):
    manifest=load(path);folder=path.parent;verify(manifest,folder)
    versions=manifest['versions'];results={};expected={(t[0],t[1]):t for t in schedule(manifest)};blocks={3:{},4:{}};games=0
    for line in (folder/'blocks.jsonl').read_text().splitlines():
        block=json.loads(line);p=block['players'];bid=block['block'];task=expected[(p,bid)]
        assert block['opponents']==task[2] and block['deal_seed']==task[3] and bid not in blocks[p]
        assert len(block['records'])==p*len(versions)
        shares=[];costs=[];times=[]
        for v in versions:
            records=[r for r in block['records'] if r['candidate']==v];assert len(records)==p
            a=b=c=0.
            for rotation,r in enumerate(records):
                group=[v]+task[2]
                assert r['rotation']==rotation and r['deal_seed']==task[3] and r['seats']==group[rotation:]+group[:rotation]
                replay(r,p,104);games+=1;seat=(-rotation)%p
                a+=r['shares'][seat]/p;b+=r['bullheads'][seat]/p;c+=r['decision_seconds'][seat]/p/10*1000
            shares.append(a);costs.append(b);times.append(c)
        blocks[p][bid]=(shares,costs,times)
    certificate=manifest['purpose']=='certificate'
    repetitions=PLAN['bootstrap_replicates'] if certificate else 20000
    tail=manifest.get('alpha_per_mode',.025)
    for p,data in blocks.items():
        assert len(data)==manifest['deals'];n=len(data)
        x=np.array([data[i][0] for i in range(n)]);costs=np.array([data[i][1] for i in range(n)]);times=np.array([data[i][2] for i in range(n)])
        rng=np.random.default_rng(seed_for(manifest['namespace']+'-bootstrap',p,-1));boot=np.empty((repetitions,len(versions)))
        for start in range(0,repetitions,100):boot[start:start+100]=x[rng.integers(0,n,(100,n))].mean(axis=1)
        ranking=[]
        for i,v in enumerate(versions):
            delta=(boot[:,i]-boot[:,0])*100
            ranking.append(dict(version=v,games=n*p,win_rate=float(x[:,i].mean()),delta_pp=float((x[:,i]-x[:,0]).mean()*100),
                delta_ci95_pp=np.quantile(delta,[.025,.975]).tolist(),adjusted_one_sided_lower_pp=float(np.quantile(delta,tail)),
                mean_bullheads=float(costs[:,i].mean()),mean_decision_ms=float(times[:,i].mean())))
        results[str(p)]=ranking
    result=dict(purpose=manifest['purpose'],games=games,results=results,audit='passed',trace_sha256=sha(folder/'blocks.jsonl'))
    dump(folder/'results.json',result);print(json.dumps(result),flush=True)
    if certificate:record_certificate(manifest,result)
    return result

def record_certificate(manifest,result):
    state=load(OUT/'state.json');attempt=manifest['certificate_attempt'];candidate=manifest['versions'][1]
    passed=all(next(r for r in result['results'][str(p)] if r['version']==candidate)['adjusted_one_sided_lower_pp']>0 and
        next(r for r in result['results'][str(p)] if r['version']==candidate)['delta_pp']>=PLAN['minimum_observed_delta_pp'] for p in (3,4))
    entry=dict(attempt=attempt,candidate=candidate,run=manifest['name'],passed=passed,alpha=manifest['alpha_attempt'],results=result['results'])
    existing=[r for r in state['certificate_attempts'] if r['attempt']==attempt]
    if existing:assert existing[0]==entry;return
    assert attempt==len(state['certificate_attempts'])+1
    state['certificate_attempts'].append(entry)
    assert state.get('pending_certificate',{}).get('attempt')==attempt
    state.pop('pending_certificate')
    if passed:
        upgrade=len(state['accepted_upgrades'])+1
        policy=next(e for e in manifest['entrants'] if e['id']==candidate)
        state['accepted_upgrades'].append(dict(upgrade=upgrade,predecessor=state['active_baseline'],**entry,policy=policy))
        state['active_baseline']=candidate
    state['goal_complete']=len(state['accepted_upgrades'])>=3
    dump(OUT/'state.json',state)

def run(path,workers):
    manifest=load(path);folder=path.parent;verify(manifest,folder)
    completed=set();trace=folder/'blocks.jsonl'
    if trace.exists():
        for line in trace.read_text().splitlines():
            b= json.loads(line);completed.add((b['players'],b['block']))
    tasks=[t for t in schedule(manifest) if (t[0],t[1]) not in completed];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(str(path),)) as executor:
        with trace.open('a',encoding='utf-8') as file:
            for i,block in enumerate(executor.map(task_work,tasks,chunksize=1),1):
                file.write(json.dumps(block)+'\n');file.flush()
                if i%8==0:print(json.dumps({'run':manifest['name'],'blocks':i+len(completed),'total':2*manifest['deals'],'players':block['players'],'seconds':round(time.perf_counter()-start,1)}),flush=True)
    return summarize(path)

def pilot():
    return prepare_run('development-001',{
        'rollout32-mixed':dict(worlds=32,opponents='mixed',prior_weight=.005,bullhead_weight=.002),
        'rollout64-neural':dict(worlds=64,opponents='neural',prior_weight=.01,bullhead_weight=.002),
        'rollout64-risk':dict(worlds=64,opponents='neural',prior_weight=.015,bullhead_weight=.02,rank_weight=.05),
        'rollout64-base':dict(worlds=64,opponents='base',prior_weight=.01,bullhead_weight=.01)
    },64,'development')

def pilot2():
    common=dict(engine='torch',device='cuda',prior_weight=.01,bullhead_weight=.002)
    return prepare_run('development-002',{
        'gpu64-neural':dict(common,worlds=64,opponents='neural'),
        'gpu128-neural':dict(common,worlds=128,opponents='neural'),
        'gpu64-proxy':dict(common,worlds=64,opponents='actual-proxy'),
        'gpu128-proxy':dict(common,worlds=128,opponents='actual-proxy')
    },128,'development')

def prepare_certificate(development,candidate):
    initialize_campaign();state=load(OUT/'state.json');source=OUT/development/'manifest.json';dev=load(source)
    assert dev['purpose']=='development' and (source.parent/'results.json').exists()
    verify(dev,source.parent)
    selected=next(e for e in dev['entrants'] if e['id']==candidate)
    reference=next(e for e in dev['entrants'] if e['id']=='reference')
    if state.get('pending_certificate'):
        pending=state['pending_certificate'];assert pending['candidate']==candidate and pending['development']==development
        return OUT/pending['run']/'manifest.json'
    if state['accepted_upgrades']:
        predecessor=state['accepted_upgrades'][-1]['policy']
        assert reference['sha256']==predecessor['sha256'] and reference.get('runtime_sha256')==predecessor.get('runtime_sha256'),'Development must compare the current accepted predecessor'
    else:assert reference['sha256']==sha(ROOT/'artifacts/small-player-exploration/v5/model.json')
    attempt=len(state['certificate_attempts'])+1;name=f'certificate-{attempt:03d}'
    path=prepare_run(name,{},PLAN['certificate_deals_per_player_count'],'certificate',reference=reference)
    manifest=load(path);manifest['versions'].append(candidate);manifest['entrants'].append(dict(selected))
    manifest.update(certificate_attempt=attempt,alpha_attempt=PLAN['alpha_total']/(attempt*(attempt+1)),
        alpha_per_mode=PLAN['alpha_total']/(2*attempt*(attempt+1)),selection_run=development,predecessor=state['active_baseline'])
    dump(path,manifest)
    state['pending_certificate']=dict(attempt=attempt,run=name,candidate=candidate,development=development)
    dump(OUT/'state.json',state)
    return path

def main():
    if Path(sys.prefix).name.lower()!='ntw-ai':raise RuntimeError('Use ntw-ai')
    parser=argparse.ArgumentParser();parser.add_argument('--manifest');parser.add_argument('--pilot',action='store_true');parser.add_argument('--pilot2',action='store_true')
    parser.add_argument('--certify-development');parser.add_argument('--candidate');parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    initialize_campaign()
    if args.certify_development:path=prepare_certificate(args.certify_development,args.candidate)
    else:path=Path(args.manifest) if args.manifest else pilot2() if args.pilot2 else pilot()
    run(path,args.workers)

if __name__=='__main__':main()
