"""Distill four real search opponents from archived, now-development-only public traces."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json
import hashlib
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch import nn
import explore_small_strategies as e
from small_player_env import SmallGame

OUT=e.ROOT/'artifacts/progressive-upgrades/opponent-proxies-v1'
KINDS=['dirv-10000','alpha-2000','mcs','champion-search']
CONFIG=dict(seed=90912001,games_per_player_count=4096,epochs=16,batch=256,lr=.0001,validation_deal_fraction=.2,
    information='Acting opponent own hand plus public rows/seen/scores only; target is its observed action. No future features or opposing hands.',
    source_role='Archived earlier experiment tests explicitly repurposed as development/training data. Future campaign certificates use new seeds.',
    split='Whole original deal seed grouped across seat rotations and substituted variants; no same deal in both training and validation.')

def dump(path,value):e.dump(path,value)

def dataset():
    sources=[e.ROOT/'artifacts/small-player-iterations/heldout-blocks.jsonl',e.ROOT/'artifacts/small-player-exploration/heldout-blocks.jsonl']
    path=OUT/'dataset.pt'
    if path.exists():return torch.load(path,map_location='cpu',weights_only=True)
    selected={3:[],4:[]}
    for source in sources:
        for line in source.read_text().splitlines():
            block=json.loads(line);p=int(block['mode'][-1])
            if p not in selected:continue
            selected[p].extend(r for r in block['records'] if any(k in r['seats'] for k in KINDS))
    rng=np.random.default_rng(CONFIG['seed']);features=[];masks=[];actions=[];types=[];seeds=[]
    for p,records in selected.items():
        order=rng.permutation(len(records))[:CONFIG['games_per_player_count']];records=[records[i] for i in order]
        for offset in range(0,len(records),64):
            group=records[offset:offset+64];b=len(group)
            cards=np.stack([np.random.default_rng(r['deal_seed']).permutation(104)+1 for r in group])
            game=object.__new__(SmallGame);game.batch=b;game.players=p
            game.hands=np.sort(cards[:,:p*10].reshape(b,p,10),axis=-1)
            game.rows=np.zeros((b,4,6),dtype=np.int64);game.rows[:,:,0]=cards[:,-1:-5:-1]
            game.lengths=np.ones((b,4),dtype=np.int64);game.scores=np.zeros((b,p),dtype=np.float32)
            game.seen=np.zeros((b,104),dtype=bool);game.seen[np.arange(b)[:,None],game.rows[:,:,0]-1]=True
            owners=np.array([[KINDS.index(name) if name in KINDS else -1 for name in r['seats']] for r in group])
            take=owners>=0;deal_ids=np.broadcast_to(np.array([r['deal_seed'] for r in group])[:,None],(b,p))[take]
            for turn in range(10):
                played=np.array([r['actions'][turn] for r in group])+1
                if turn<9:
                    f=game.features()[take];h=10-turn;n=len(f)
                    pad=np.zeros((n,10,270),dtype=np.float32);pad[:,:h]=f
                    choice=(game.hands==played[:,:,None]).argmax(axis=-1)[take]
                    features.append(pad);masks.append(np.broadcast_to(np.arange(10)<h,(n,10)).copy())
                    actions.append(choice);types.append(owners[take]);seeds.append(deal_ids)
                game.step(played)
            for i,r in enumerate(group):np.testing.assert_array_equal(game.scores[i],r['bullheads'])
            if offset%512==0:print(json.dumps({'phase':'data','players':p,'games':min(offset+64,len(records))}),flush=True)
    values=dict(features=torch.from_numpy(np.concatenate(features)),mask=torch.from_numpy(np.concatenate(masks)),
        action=torch.from_numpy(np.concatenate(actions)).long(),kind=torch.from_numpy(np.concatenate(types)).long(),deal_seed=torch.from_numpy(np.concatenate(seeds)))
    # Stable split keyed by original deal, not by individual action or copied game.
    unique=np.unique(values['deal_seed'].numpy())
    is_validation={int(seed):int.from_bytes(hashlib.sha256(f'proxy-v1:{seed}'.encode()).digest()[:4],'little')%5==0 for seed in unique}
    values['validation']=torch.tensor([is_validation[int(seed)] for seed in values['deal_seed']],dtype=torch.bool)
    torch.save(values,path)
    dump(OUT/'data-provenance.json',dict(source_sha256={str(s.relative_to(e.ROOT)):e.sha(s) for s in sources},states=len(values['action']),
        unique_deals=len(unique),counts_by_type={k:int((values['kind']==i).sum()) for i,k in enumerate(KINDS)},
        validation_states=int(values['validation'].sum()),training_validation_deals_disjoint=True))
    return values

class MultiPolicy(nn.Module):
    def __init__(self,base):
        super().__init__();self.layers=nn.ModuleList([nn.Linear(270,256),nn.Linear(256,128),nn.Linear(128,64),nn.Linear(64,4)])
        with torch.no_grad():
            for a,b in zip(self.layers[:3],base.layers[:3]):a.load_state_dict(b.state_dict())
            self.layers[-1].weight.copy_(base.layers[-1].weight.repeat(4,1));self.layers[-1].bias.copy_(base.layers[-1].bias.repeat(4))
    def forward(self,x):
        for layer in self.layers[:-1]:x=torch.relu(layer(x))
        return self.layers[-1](x)

@torch.inference_mode()
def evaluate(model,data,indices):
    result={k:dict(n=0,correct=0,nll=0.) for k in KINDS}
    for ids in indices.split(512):
        f=data['features'][ids].to(e.DEVICE);kind=data['kind'][ids].to(e.DEVICE);mask=data['mask'][ids].to(e.DEVICE);chosen=data['action'][ids].to(e.DEVICE)
        raw=model(f);raw=raw.gather(-1,kind[:,None,None].expand(-1,10,1)).squeeze(-1)
        logp=torch.log_softmax((-raw*10).masked_fill(~mask,-1e9),dim=-1)
        correct=(logp.argmax(-1)==chosen).cpu();loss=-logp.gather(1,chosen[:,None]).squeeze(1).cpu()
        for i,k in enumerate(KINDS):
            take=data['kind'][ids]==i;result[k]['n']+=int(take.sum());result[k]['correct']+=int(correct[take].sum());result[k]['nll']+=float(loss[take].sum())
    return {k:dict(states=r['n'],accuracy=r['correct']/r['n'],nll=r['nll']/r['n']) for k,r in result.items()}

def main():
    if Path(sys.prefix).name.lower()!='ntw-ai' or not torch.cuda.is_available():raise RuntimeError('Use ntw-ai CUDA')
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'complete.json').exists():print('Opponent proxies already complete');return
    torch.set_num_threads(1);torch.manual_seed(CONFIG['seed']);e.DEVICE=torch.device('cuda')
    dump(OUT/'protocol.json',dict(**CONFIG,source_sha256=e.sha(Path(__file__))))
    data=dataset();base=e.load(e.BASE);model=MultiPolicy(base).to(e.DEVICE)
    optimizer=torch.optim.AdamW(model.parameters(),lr=CONFIG['lr'],weight_decay=.0001)
    train=torch.nonzero(~data['validation'],as_tuple=True)[0];validation=torch.nonzero(data['validation'],as_tuple=True)[0]
    initial=evaluate(model,data,validation);dump(OUT/'initial-validation.json',initial)
    best={k:float('inf') for k in KINDS};history=[];start=time.perf_counter()
    for epoch in range(1,CONFIG['epochs']+1):
        losses=[]
        for ids in train[torch.randperm(len(train))].split(CONFIG['batch']):
            f=data['features'][ids].to(e.DEVICE);mask=data['mask'][ids].to(e.DEVICE);kind=data['kind'][ids].to(e.DEVICE)
            raw=model(f).gather(-1,kind[:,None,None].expand(-1,10,1)).squeeze(-1)
            logits=(-raw*10).masked_fill(~mask,-1e9)
            loss=nn.functional.cross_entropy(logits,data['action'][ids].to(e.DEVICE))
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite opponent loss')
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
        metrics=evaluate(model,data,validation)
        row=dict(epoch=epoch,loss=float(np.mean(losses)),validation=metrics,seconds=time.perf_counter()-start);history.append(row)
        with (OUT/'training.jsonl').open('a',encoding='utf-8') as file:file.write(json.dumps(row)+'\n')
        for i,k in enumerate(KINDS):
            if metrics[k]['nll']<best[k]:
                best[k]=metrics[k]['nll'];student=e.load(e.BASE)
                with torch.no_grad():
                    for a,b in zip(student.layers[:3],model.layers[:3]):a.load_state_dict(b.state_dict())
                    student.layers[-1].weight.copy_(model.layers[-1].weight[i:i+1]);student.layers[-1].bias.copy_(model.layers[-1].bias[i:i+1])
                torch.save(dict(state_dict={n:v.detach().cpu().clone() for n,v in student.state_dict().items()},epoch=epoch,opponent=k,validation=metrics[k]),OUT/(k+'.pt'))
                dump(OUT/(k+'.json'),e.serial(student))
        print(json.dumps(row),flush=True)
    dump(OUT/'complete.json',dict(status='complete',initial=initial,best_validation_nll=best,epochs=CONFIG['epochs'],
        states=len(train),validation_states=len(validation),source_sha256=e.sha(Path(__file__))))

if __name__=='__main__':main()
