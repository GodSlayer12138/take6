"""Distill completed development planners into a cheap continuation policy.

Only explicit completed development runs are allowed. Certificate games are never
used here. Whole deal seeds, across every rotation and teacher, share a split.
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
from small_player_env import SmallGame

CAMPAIGN=e.ROOT/'artifacts/progressive-upgrades'


def build_data(sources):
    records=[];provenance={}
    for run,teachers in sources.items():
        folder=CAMPAIGN/run;manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        assert manifest['purpose']=='development' and (folder/'results.json').exists()
        trace=folder/'blocks.jsonl';provenance[run]=dict(trace_sha256=e.sha(trace),teachers=teachers)
        for line in trace.read_text(encoding='utf-8').splitlines():
            block=json.loads(line)
            assert block['players'] in (3,4)
            records.extend(r for r in block['records'] if r['candidate'] in teachers)
    values={k:[] for k in ('features','mask','action','deal_seed','players')}
    for p in (3,4):
        group=[r for r in records if len(r['seats'])==p]
        for offset in range(0,len(group),64):
            batch=group[offset:offset+64];b=len(batch)
            cards=np.stack([np.random.default_rng(r['deal_seed']).permutation(104)+1 for r in batch])
            game=object.__new__(SmallGame);game.batch=b;game.players=p
            game.hands=np.sort(cards[:,:p*10].reshape(b,p,10),axis=-1)
            game.rows=np.zeros((b,4,6),dtype=np.int64);game.rows[:,:,0]=cards[:,-1:-5:-1]
            game.lengths=np.ones((b,4),dtype=np.int64);game.scores=np.zeros((b,p),dtype=np.float32)
            game.seen=np.zeros((b,104),dtype=bool);game.seen[np.arange(b)[:,None],game.rows[:,:,0]-1]=True
            seats=np.array([r['seats'].index(r['candidate']) for r in batch])
            for turn in range(10):
                played=np.array([r['actions'][turn] for r in batch])+1
                if turn<9:
                    features=game.features()[np.arange(b),seats];h=10-turn
                    padded=np.zeros((b,10,270),dtype=np.float32);padded[:,:h]=features
                    choice=(game.hands[np.arange(b),seats]==played[np.arange(b),seats,None]).argmax(axis=-1)
                    values['features'].append(padded);values['action'].append(choice)
                    values['mask'].append(np.broadcast_to(np.arange(10)<h,(b,10)).copy())
                    values['deal_seed'].append(np.array([r['deal_seed'] for r in batch]))
                    values['players'].append(np.full(b,p))
                game.step(played)
            np.testing.assert_array_equal(game.scores,np.array([r['bullheads'] for r in batch]))
    data={k:torch.from_numpy(np.concatenate(v)) for k,v in values.items()}
    unique=data['deal_seed'].unique().tolist()
    validation={seed:int.from_bytes(hashlib.sha256(f'continuation-v1:{seed}'.encode()).digest()[:4],'little')%5==0 for seed in unique}
    data['validation']=torch.tensor([validation[seed] for seed in data['deal_seed'].tolist()])
    return data,dict(sources=provenance,unique_deals=len(unique),states=len(data['action']),
        validation_states=int(data['validation'].sum()),information='Own current hand and public state; imitate development planner action',
        split='Grouped by original deal seed; no certificate games')


@torch.inference_mode()
def evaluate(model,data,ids):
    n=correct=0;loss=0.
    for ix in ids.split(512):
        features=data['features'][ix].to(e.DEVICE);mask=data['mask'][ix].to(e.DEVICE);target=data['action'][ix].to(e.DEVICE)
        logits=(-model(features).squeeze(-1)*10).masked_fill(~mask,-1e9)
        loss+=float(torch.nn.functional.cross_entropy(logits,target,reduction='sum'))
        correct+=int((logits.argmax(-1)==target).sum());n+=len(ix)
    return dict(states=n,nll=loss/n,accuracy=correct/n)


def train(args):
    out=CAMPAIGN/args.output
    if (out/'complete.json').exists():print('Already complete');return
    config=json.loads(Path(args.sources).read_text(encoding='utf-8'));out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.manual_seed(90925011);e.DEVICE=torch.device(args.device)
    e.dump(out/'protocol.json',dict(sources=config,epochs=20,lr=.0001,temperature=1.,
        loss='cross entropy against selected development planner actions',source_sha256=e.sha(Path(__file__))))
    path=out/'dataset.pt'
    if not path.exists():
        data,provenance=build_data(config);torch.save(data,path);e.dump(out/'data-provenance.json',provenance)
    else:data=torch.load(path,weights_only=True)
    model=e.load(e.BASE);optimizer=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.001)
    train_ids=torch.nonzero(~data['validation'],as_tuple=True)[0];val_ids=torch.nonzero(data['validation'],as_tuple=True)[0]
    best=float('inf');initial=evaluate(model,data,val_ids);e.dump(out/'initial-validation.json',initial);start=time.perf_counter()
    for epoch in range(1,21):
        losses=[]
        for ix in train_ids[torch.randperm(len(train_ids))].split(256):
            features=data['features'][ix].to(e.DEVICE);mask=data['mask'][ix].to(e.DEVICE);target=data['action'][ix].to(e.DEVICE)
            # Permuting row slots is a game symmetry and preserves the target card index.
            perm=e.PERMS[int(torch.randint(len(e.PERMS),(1,)))]
            features=torch.from_numpy(e.permute_features(features.cpu().numpy(),perm)).to(e.DEVICE)
            logits=(-model(features).squeeze(-1)*10).masked_fill(~mask,-1e9)
            loss=torch.nn.functional.cross_entropy(logits,target)
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite continuation loss')
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            losses.append(float(loss.detach()))
        metrics=evaluate(model,data,val_ids)
        row=dict(epoch=epoch,loss=float(np.mean(losses)),validation=metrics,seconds=time.perf_counter()-start)
        with (out/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if metrics['nll']<best:
            best=metrics['nll'];torch.save(dict(state_dict={k:v.cpu().clone() for k,v in model.state_dict().items()},
                epoch=epoch,validation=metrics,role='development-search continuation distillation'),out/'model.pt')
            e.dump(out/'model.json',e.serial(model))
        print(json.dumps(row),flush=True)
    e.dump(out/'complete.json',dict(initial=initial,best_validation_nll=best,states=len(train_ids),validation_states=len(val_ids)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--sources',required=True);parser.add_argument('--output',default='continuation-v1')
    parser.add_argument('--device',default='cuda');train(parser.parse_args())
