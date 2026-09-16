"""Learn relative action values from high-budget public-information search.

Unlike hard action imitation, every candidate card gets a soft winning-value
target. The label source is completed development games, never certificates.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import explore_small_strategies as e
from progressive_planner import Planner
from small_player_env import SmallGame
from train_policy import encode_candidate

OUT=e.ROOT/'artifacts/progressive-upgrades/qstudent-v1'
CONFIG=dict(source_run='development-002',source_candidate='gpu128-proxy',states_per_player_count=512,
    selection_seed=90931001,epochs=40,lr=.0001,temperature=.075,
    teacher=dict(engine='torch',device='cuda',worlds=512,opponents='actual-proxy',
        proxy_directory='artifacts/progressive-upgrades/opponent-proxies-v2',prior_weight=.003,bullhead_weight=.002),
    selection='Minimum development validation regret under teacher values; not a game-strength certificate',
    split='Whole source deal grouped across rotations and turns')


def states():
    path=OUT/'states.json'
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    source=e.ROOT/'artifacts/progressive-upgrades'/CONFIG['source_run']
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['purpose']=='development' and (source/'results.json').exists()
    records={3:[],4:[]}
    for line in (source/'blocks.jsonl').read_text(encoding='utf-8').splitlines():
        block=json.loads(line)
        records[block['players']].extend(r for r in block['records'] if r['candidate']==CONFIG['source_candidate'])
    selected=[];rng=np.random.default_rng(CONFIG['selection_seed'])
    for p,group in records.items():
        keys=rng.choice(len(group)*9,CONFIG['states_per_player_count'],replace=False)
        wanted=set((int(k)//9,int(k)%9) for k in keys)
        for idx,r in enumerate(group):
            if not any((idx,t) in wanted for t in range(9)):continue
            table=SmallGame(1,p,r['deal_seed']);table.rows=table.rows[:,::-1].copy()
            seat=r['seats'].index(r['candidate']);order=[(seat+i)%p for i in range(p)];history=[]
            for turn in range(10):
                obs=table.record(0,seat);obs['scores']=table.scores[0,order].tolist()
                if (idx,turn) in wanted:
                    obs.update(seed=int(r['deal_seed']+r['rotation']*1000003+turn*104729+seat*8191),history=list(history))
                    selected.append(dict(id=f'{p}:{r["deal_seed"]}:{r["rotation"]}:{turn}',deal_seed=r['deal_seed'],observation=obs))
                played=np.array(r['actions'][turn])+1
                history.append(dict(rows=obs['rows'],seenCards=obs['seenCards'],ownCard=int(played[seat]),opponentCards=played[order[1:]].tolist()))
                table.step(played[None])
            np.testing.assert_array_equal(table.scores[0],r['bullheads'])
    e.dump(path,selected)
    e.dump(OUT/'data-provenance.json',dict(source_trace_sha256=e.sha(source/'blocks.jsonl'),states=len(selected),
        information='Own current hand, public rows/history/scores only; hypothetical remaining hands sampled by teacher'))
    return selected


def labels():
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/'labels.jsonl';protocol=OUT/'protocol.json'
    dependencies=[Path(__file__),e.ROOT/'training/progressive_planner.py',e.ROOT/'training/progressive_torch_env.py',
        *[e.ROOT/CONFIG['teacher']['proxy_directory']/(name+'.pt') for name in ('dirv-10000','alpha-2000','mcs','champion-search')]]
    frozen=dict(**CONFIG,source_sha256={str(p):e.sha(p) for p in dependencies})
    if protocol.exists():assert json.loads(protocol.read_text(encoding='utf-8'))==frozen
    else:e.dump(protocol,frozen)
    jobs=states();completed={}
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            row=json.loads(line);assert row['id'] not in completed;completed[row['id']]=row
    planner=Planner(CONFIG['teacher']);start=time.perf_counter()
    with path.open('a',encoding='utf-8') as file:
        for state in jobs:
            if state['id'] in completed:continue
            result=planner.choose(state['observation'])
            assert result['candidates']==sorted(state['observation']['hand']) and np.isfinite(result['values']).all()
            row=dict(id=state['id'],values=result['values'],seconds=result['seconds'])
            file.write(json.dumps(row)+'\n');file.flush();completed[row['id']]=row
            if len(completed)%32==0:print(json.dumps(dict(phase='labels',states=len(completed),total=len(jobs),seconds=time.perf_counter()-start)),flush=True)
    return jobs,completed


@torch.inference_mode()
def evaluate(model,features,values,mask,ids):
    predicted=(-model(features[ids])*10).masked_fill(~mask[ids],-1e9)
    chosen=predicted.argmax(-1);target=values[ids].masked_fill(~mask[ids],-1e9)
    regret=target.max(-1).values-target.gather(1,chosen[:,None]).squeeze(-1)
    return dict(regret=float(regret.mean()),accuracy=float((chosen==target.argmax(-1)).float().mean()),states=len(ids))


def train():
    if (OUT/'complete.json').exists():print('Already complete');return
    jobs,targets=labels();torch.set_num_threads(1);torch.manual_seed(90931013);e.DEVICE=torch.device('cuda')
    f=np.zeros((len(jobs),10,270),dtype=np.float32);q=np.zeros((len(jobs),10),dtype=np.float32);m=np.zeros((len(jobs),10),dtype=bool)
    validation=[]
    for i,state in enumerate(jobs):
        obs=state['observation'];h=len(obs['hand']);f[i,:h]=[encode_candidate(obs,c) for c in sorted(obs['hand'])]
        q[i,:h]=targets[state['id']]['values'];m[i,:h]=True
        validation.append(int.from_bytes(hashlib.sha256(f'qstudent-v1:{state["deal_seed"]}'.encode()).digest()[:4],'little')%5==0)
    train_ids=np.flatnonzero(~np.array(validation));val_ids=np.flatnonzero(validation)
    assert {jobs[i]['deal_seed'] for i in train_ids}.isdisjoint({jobs[i]['deal_seed'] for i in val_ids})
    features=torch.tensor(f,device=e.DEVICE);values=torch.tensor(q,device=e.DEVICE);mask=torch.tensor(m,device=e.DEVICE)
    model=e.load(e.BASE)
    with torch.no_grad():model.layers[-1].weight.mul_(.02);model.layers[-1].bias.mul_(.02).sub_(.05)
    optimizer=torch.optim.AdamW(model.parameters(),lr=CONFIG['lr'],weight_decay=.001)
    initial=evaluate(model,features,values,mask,val_ids);e.dump(OUT/'initial-validation.json',initial);best=float('inf')
    for epoch in range(1,CONFIG['epochs']+1):
        losses=[]
        for ix in np.array_split(np.random.default_rng(90931100+epoch).permutation(train_ids),max(1,(len(train_ids)+127)//128)):
            permutation=e.PERMS[int(torch.randint(len(e.PERMS),(1,)))]
            augmented=torch.tensor(e.permute_features(f[ix],permutation),device=e.DEVICE)
            predicted=-model(augmented)*10;legal=mask[ix];n=legal.sum(-1,keepdim=True)
            centered=(predicted-(predicted*legal).sum(-1,keepdim=True)/n)*legal
            target=(values[ix]-(values[ix]*legal).sum(-1,keepdim=True)/n)*legal
            target_probability=torch.softmax((values[ix]/CONFIG['temperature']).masked_fill(~legal,-1e9),-1)
            log_probability=torch.log_softmax((predicted/CONFIG['temperature']).masked_fill(~legal,-1e9),-1)
            loss=torch.nn.functional.kl_div(log_probability,target_probability,reduction='batchmean')+5*(centered-target).square().sum()/legal.sum()
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite Q distillation loss')
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
        metrics=evaluate(model,features,values,mask,val_ids);row=dict(epoch=epoch,loss=float(np.mean(losses)),validation=metrics)
        with (OUT/'training.jsonl').open('a',encoding='utf-8') as file:file.write(json.dumps(row)+'\n')
        if metrics['regret']<best:
            best=metrics['regret'];torch.save(dict(state_dict={k:v.cpu().clone() for k,v in model.state_dict().items()},epoch=epoch,validation=metrics),OUT/'model.pt')
            e.dump(OUT/'model.json',e.serial(model))
        if epoch%5==0:print(json.dumps(row),flush=True)
    e.dump(OUT/'complete.json',dict(states=len(train_ids),validation_states=len(val_ids),initial=initial,best_validation_regret=best,
        role='Counterfactual soft action-value distillation from development states'))


if __name__=='__main__':train()
