"""Three trained revisions of the local 270-feature policy; only 2-4 players."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch import nn
from train_policy import CandidateValueNet
from small_player_env import SmallGame

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/small-player-iterations'
BASE=ROOT/'artifacts/models/ntw-adaptive5-v1.pt'
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
STAGES={
 'v2':dict(method='advantage-weighted-regression',updates=64,batch=256,lr=3e-5,temperature=.9,anchor=.3,seed=60831001,save_every=16,players=[4]),
 'v3':dict(method='ppo-terminal-outcome',updates=160,batch=256,lr=1e-5,temperature=.85,anchor=.12,seed=60832001,save_every=40,players=[4]),
 'v4':dict(method='ppo-mixed-player-league',updates=192,batch=256,lr=8e-6,temperature=.85,anchor=.15,seed=60833001,save_every=48,players=[4,2,4,3]),
}

def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    model=CandidateValueNet().to(DEVICE);model.load_state_dict(checkpoint['state_dict']);model.eval()
    return model

def export(model,path,version,metadata):
    model_layers=[{'weight':l.weight.detach().cpu().tolist(),'bias':l.bias.detach().cpu().tolist()} for l in model.layers]
    dump(path,dict(format='ntw-neural-v1',featureVersion=1,featureSize=270,targetScale=10.,activation='relu',
                   name=f'NTW small-player {version}',layers=model_layers,training=metadata))

class Critic(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(252,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,x):return self.net(x).squeeze(-1)

def sample(logits,uniform):
    value=logits-logits.max(axis=-1,keepdims=True);prob=np.exp(value);prob/=prob.sum(axis=-1,keepdims=True)
    return np.minimum((uniform[:,:,None]>prob.cumsum(axis=-1)).sum(axis=-1),logits.shape[-1]-1)

@torch.inference_mode()
def rollout(policy,opponents,players,batch,seed,temperature=.85,collect=True):
    game=SmallGame(batch,players,seed);rng=np.random.default_rng(seed+900000000)
    # Focal seat varies independently; no opponent identity enters the policy features.
    focal=np.arange(batch)%players
    owner=rng.integers(0,len(opponents),(batch,players))
    random_seat=rng.random((batch,players))<.10
    features=[];masks=[];chosen=[];old_log=[];rewards=[]
    for turn in range(10):
        feats=game.features();h=game.hands.shape[-1]
        logits=np.empty((batch,players,h),dtype=np.float32)
        for i,opponent in enumerate(opponents):
            take=owner==i
            if take.any():logits[take]=(-opponent(torch.from_numpy(feats[take]).to(DEVICE))*10/.8).cpu().numpy()
        logits[random_seat]=0
        own=feats[np.arange(batch),focal]
        own_logits=(-policy(torch.from_numpy(own).to(DEVICE))*10/temperature).cpu().numpy()
        logits[np.arange(batch),focal]=own_logits
        selected=sample(logits,rng.random((batch,players)))
        own_selected=selected[np.arange(batch),focal]
        if collect:
            padded=np.zeros((batch,10,270),dtype=np.float32);padded[:,:h]=own
            features.append(padded);masks.append(np.broadcast_to(np.arange(10)<h,(batch,10)).copy())
            chosen.append(own_selected)
            stable=own_logits-own_logits.max(axis=-1,keepdims=True)
            logprob=stable-np.log(np.exp(stable).sum(axis=-1,keepdims=True))
            old_log.append(logprob[np.arange(batch),own_selected])
        cards=np.take_along_axis(game.hands,selected[:,:,None],axis=-1)[:,:,0]
        penalties=game.step(cards)
        rewards.append(-.003*penalties[np.arange(batch),focal])
    share,rank=game.outcome()
    rewards[-1]+=share[np.arange(batch),focal]-.10*(rank[np.arange(batch),focal]-1)/(players-1)
    reward=np.stack(rewards,axis=1);returns=np.flip(np.cumsum(np.flip(reward,axis=1),axis=1),axis=1).copy()
    metrics=dict(games=batch,players=players,win_share=float(share[np.arange(batch),focal].mean()),
                 mean_bullheads=float(game.scores[np.arange(batch),focal].mean()),mean_reward=float(reward.sum(axis=1).mean()))
    if not collect:return metrics
    # Flatten in turn-major order, consistent across all tensors.
    dataset=dict(features=torch.from_numpy(np.concatenate(features)),mask=torch.from_numpy(np.concatenate(masks)),
                 chosen=torch.from_numpy(np.concatenate(chosen)).long(),old_log=torch.from_numpy(np.concatenate(old_log)),
                 returns=torch.from_numpy(returns.T.reshape(-1)),episode_reward=torch.from_numpy(np.tile(reward.sum(axis=1),10)))
    return dataset,metrics

@torch.inference_mode()
def deterministic_scores(policy,opponents,players,seed,games=1024):
    result=[]
    for offset in range(0,games,256):
        batch=min(256,games-offset);game=SmallGame(batch,players,seed+offset*65537)
        rng=np.random.default_rng(seed+offset*65537+888)
        owners=rng.integers(0,len(opponents),(batch,players));focal=np.arange(batch)%players
        for turn in range(10):
            feats=game.features();indices=np.zeros((batch,players),dtype=np.int64)
            for i,opponent in enumerate(opponents):
                take=owners==i
                if take.any():indices[take]=opponent(torch.from_numpy(feats[take]).to(DEVICE)).argmin(dim=-1).cpu().numpy()
            indices[np.arange(batch),focal]=policy(torch.from_numpy(feats[np.arange(batch),focal]).to(DEVICE)).argmin(dim=-1).cpu().numpy()
            cards=np.take_along_axis(game.hands,indices[:,:,None],axis=-1)[:,:,0];game.step(cards)
        share,_=game.outcome();result.extend(share[np.arange(batch),focal].tolist())
    return np.array(result)

def log_probs(model,features,mask,temperature):
    logits=(-model(features)*10/temperature).masked_fill(~mask,-1e9)
    return torch.log_softmax(logits,dim=-1)

def update_awr(model,anchor,dataset,optimizer,config):
    reward=dataset['episode_reward'];adv=(reward-reward.mean())/reward.std(unbiased=False).clamp_min(.15)
    weights=torch.exp(adv/.8).clamp(max=4);weights/=weights.mean()
    totals=[]
    for epoch in range(2):
        for ids in torch.randperm(len(reward)).split(512):
            f=dataset['features'][ids].to(DEVICE);m=dataset['mask'][ids].to(DEVICE);a=dataset['chosen'][ids].to(DEVICE)
            logp=log_probs(model,f,m,config['temperature']);prob=logp.exp()
            with torch.no_grad():reference=log_probs(anchor,f,m,config['temperature'])
            kl=(prob*(logp-reference)).sum(dim=-1).mean()
            objective=-(logp.gather(1,a[:,None])[:,0]*weights[ids].to(DEVICE)).mean()
            loss=objective+config['anchor']*kl
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            totals.append([float(loss.detach()),float(kl.detach())])
    return dict(loss=float(np.mean(totals,axis=0)[0]),anchor_kl=float(np.mean(totals,axis=0)[1]))

def update_ppo(model,anchor,critic,dataset,optimizer,critic_optimizer,config):
    n=len(dataset['chosen'])
    with torch.no_grad():
        values=torch.cat([critic(f[:,0,:252].to(DEVICE)).cpu() for f in dataset['features'].split(512)])
        adv=dataset['returns']-values
        adv=(adv-adv.mean())/adv.std(unbiased=False).clamp_min(.1)
    totals=[]
    for epoch in range(3):
        for ids in torch.randperm(n).split(512):
            f=dataset['features'][ids].to(DEVICE);m=dataset['mask'][ids].to(DEVICE);a=dataset['chosen'][ids].to(DEVICE)
            logp=log_probs(model,f,m,config['temperature']);prob=logp.exp()
            selected=logp.gather(1,a[:,None])[:,0];old=dataset['old_log'][ids].to(DEVICE)
            ratio=torch.exp(selected-old);advantages=adv[ids].to(DEVICE)
            policy=-torch.minimum(ratio*advantages,ratio.clamp(.85,1.15)*advantages).mean()
            with torch.no_grad():ref=log_probs(anchor,f,m,config['temperature'])
            anchor_kl=(prob*(logp-ref)).sum(dim=-1).mean()
            entropy=-(prob*logp).sum(dim=-1).mean()
            loss=policy+config['anchor']*anchor_kl-.004*entropy
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite loss')
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            predicted=critic(f[:,0,:252]);value_loss=nn.functional.mse_loss(predicted,dataset['returns'][ids].to(DEVICE))
            critic_optimizer.zero_grad();value_loss.backward();nn.utils.clip_grad_norm_(critic.parameters(),1.);critic_optimizer.step()
            approx_kl=((ratio-1)-(selected-old)).mean().detach()
            totals.append([float(loss.detach()),float(anchor_kl.detach()),float(value_loss.detach()),float(approx_kl)])
        if np.mean(totals[-math_ceil(n,512):],axis=0)[3]>.04:break
    means=np.mean(totals,axis=0)
    return dict(loss=float(means[0]),anchor_kl=float(means[1]),value_loss=float(means[2]),approx_kl=float(means[3]))

def math_ceil(n,k):return (n+k-1)//k

def train_stage(version,initial,opponent_paths,smoke=False):
    config=dict(STAGES[version])
    if smoke:config.update(updates=2,batch=16,save_every=1)
    dest=OUT/('smoke-'+version if smoke else version)
    if (dest/'summary.json').exists():return dest/'model.pt'
    dest.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
    model=load(initial);anchor=copy.deepcopy(model).eval()
    for parameter in anchor.parameters():parameter.requires_grad_(False)
    opponents=[load(p) for p in opponent_paths];dev_opponents=[load(p) for p in [BASE,ROOT/'artifacts/models/ntw-adaptive-rl-v3b.pt',ROOT/'artifacts/models/ntw-champion-v2.pt']]
    base=load(BASE);critic=Critic().to(DEVICE)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=1e-5)
    critic_optimizer=torch.optim.AdamW(critic.parameters(),lr=3e-4)
    protocol=dict(version=version,config=config,initial=str(initial),initial_sha256=digest(initial),opponents=[str(p) for p in opponent_paths],
        reward='win_share - .10*(rank-1)/(P-1) - .003*bullheads; gamma=1',
        selection='Best prespecified checkpoint on development seeds only; independent final seeds never used for selection.',
        device=str(DEVICE),gpu=torch.cuda.get_device_name(0),torch=str(torch.__version__),
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'training/small_player_env.py',ROOT/'training/train_policy.py']})
    dump(dest/'protocol.json',protocol)
    best=-float('inf');best_update=None;history=[];start=time.perf_counter()
    dev_counts=[4] if version!='v4' else [2,3,4]
    dev_games=64 if smoke else 1024
    base_dev={p:deterministic_scores(base,dev_opponents,p,60841001+p*100000,dev_games).mean() for p in dev_counts}
    with (dest/'training.jsonl').open('w',encoding='utf-8') as log:
        for step in range(1,config['updates']+1):
            players=config['players'][(step-1)%len(config['players'])]
            behavior=anchor if version=='v2' else model
            dataset,metrics=rollout(behavior,opponents,players,config['batch'],config['seed']+step*104729,config['temperature'])
            model.train()
            result=update_awr(model,anchor,dataset,optimizer,config) if version=='v2' else update_ppo(model,anchor,critic,dataset,optimizer,critic_optimizer,config)
            if not all(torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError('Non-finite weights')
            model.eval()
            record=dict(update=step,**metrics,**result,seconds=round(time.perf_counter()-start,3))
            if step%config['save_every']==0:
                validation={p:float(deterministic_scores(model,dev_opponents,p,60841001+p*100000,dev_games).mean()) for p in dev_counts}
                score=validation[4]-base_dev[4] if version!='v4' else .6*(validation[4]-base_dev[4])+.2*(validation[2]-base_dev[2])+.2*(validation[3]-base_dev[3])
                record['development']=validation;record['development_delta']=score
                payload=dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},featureVersion=1,featureSize=270,
                    version=version,update=step,training=protocol,optimizer=optimizer.state_dict(),critic=critic.state_dict(),critic_optimizer=critic_optimizer.state_dict())
                checkpoint=dest/f'checkpoint-{step:04d}.pt';torch.save(payload,checkpoint)
                if score>best:
                    best=score;best_update=step;torch.save(payload,dest/'model.pt')
                if version=='v4':
                    opponents.append(copy.deepcopy(model).eval())
                    if len(opponents)>7:opponents.pop(3)
            log.write(json.dumps(record)+'\n');log.flush();history.append(record)
            if step%4==0 or 'development' in record:print(json.dumps({'stage':version,**record}),flush=True)
    selected=load(dest/'model.pt')
    metadata=dict(version=version,selected_update=best_update,total_training_games=config['updates']*config['batch'],
        initial=str(initial),initial_sha256=digest(initial),development_baseline=base_dev,development_delta=best,
        seconds=time.perf_counter()-start,protocol=str(dest/'protocol.json'),training_players=config['players'])
    export(selected,dest/'model.json',version,metadata)
    metadata.update(checkpoint_sha256=digest(dest/'model.pt'),json_sha256=digest(dest/'model.json'))
    dump(dest/'summary.json',metadata)
    print(json.dumps({'completed':version,**metadata}),flush=True)
    return dest/'model.pt'

def plan():
    path=OUT/'experiment-plan.json'
    if path.exists():return
    dump(path,dict(stages=STAGES,baseline=str(BASE),baseline_sha256=digest(BASE),scope='Only 2,3,4 players; fixed 104-card deck.',
        baseline_identity='Existing adaptive5-v1 checkpoint; not the current hybrid game route.',
        development_seed_base=60841001,heldout_seeds={'classic4':60851001,'classic3':60852001,'classic2':60853001},
        heldout_deals={'classic4':512,'classic3':256,'classic2':256},
        evaluation='Paired candidate substitution against identical frozen opponent groups and deal/seat seeds. Four versions: baseline,v2,v3,v4. Separate results per player count.',
        policy='Do not alter deployed UI defaults during training. Report regressions and uncertain differences honestly.'))

def main():
    if Path(sys.prefix).name.lower()!='ntw-ai' or DEVICE.type!='cuda':raise RuntimeError('Use ntw-ai CUDA environment')
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args();plan()
    initial=BASE
    fixed=[BASE,ROOT/'artifacts/models/ntw-adaptive-rl-v3b.pt',ROOT/'artifacts/models/ntw-adaptive-arena-v8-distill.pt']
    trained=[]
    for version in STAGES:
        initial=train_stage(version,initial,fixed+trained,args.smoke)
        trained.append(initial)
    dump(OUT/('smoke-complete.json' if args.smoke else 'training-complete.json'),{'versions':[str(p) for p in trained],'status':'complete'})

if __name__=='__main__':main()
