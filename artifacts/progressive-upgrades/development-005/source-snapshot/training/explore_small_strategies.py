"""Independent small-player branches: symmetry, evolutionary residuals, counterfactual distillation."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from small_player_env import SmallGame,POINTS
from train_policy import CandidateValueNet

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/small-player-exploration'
BASE=ROOT/'artifacts/models/ntw-adaptive5-v1.pt'
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PERMS=list(itertools.permutations(range(4)))
BASIS_INDICES=[264,266,265,267,269,255,254,257,256]
PLAN=dict(scope='Only 2,3,4 players; fixed 104 cards.',
    branches=['v5-symmetry','v6-evolution','v7-counterfactual'],
    symmetry_blends=[.25,.5,.75,1.],
    evolution=dict(population=24,generations=16,games=256,elites=6,initial_std=1.2,save_every=4,seed=60861001),
    counterfactual=dict(root_games=128,turns=[0,3,6,8],worlds=32,epochs=24,save_every=6,lr=1e-5,temperature=.75,advantage_scale=6.,seed=60862001),
    development_games=1024,development_seed=60871001,
    heldout_deals={'classic4':512,'classic3':256,'classic2':256},
    heldout_seeds={'classic4':60881001,'classic3':60882001,'classic2':60883001},
    selection='Choose each player-count setting on development games only; freeze all branches before any heldout games.',
    interpretation='V5 is an inference ensemble, V6 trains 27 residual coefficients per player count, V7 trains a separate policy per player count. These are experiments, not established improvements.')

def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path):
    m=CandidateValueNet().to(DEVICE)
    m.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['state_dict'])
    return m.eval()

def serial(model):
    return dict(format='ntw-neural-v1',featureVersion=1,featureSize=270,targetScale=10.,activation='relu',
        layers=[dict(weight=l.weight.detach().cpu().tolist(),bias=l.bias.detach().cpu().tolist()) for l in model.layers])

@torch.inference_mode()
def utilities(model,features):
    shape=features.shape[:-1];flat=features.reshape(-1,270)
    values=[model(torch.from_numpy(f).to(DEVICE)).cpu().numpy()*10 for f in np.array_split(flat,max(1,(len(flat)+16383)//16384))]
    return np.concatenate(values).reshape(shape)

def permute_features(features,permutation):
    f=features.copy();p=list(permutation)
    f[...,208:232]=features[...,208:232].reshape(*features.shape[:-1],4,6)[...,p,:].reshape(*features.shape[:-1],24)
    f[...,232:244]=features[...,232:244].reshape(*features.shape[:-1],4,3)[...,p,:].reshape(*features.shape[:-1],12)
    # Preserve the physical forced row selected under the actual game's tie convention.
    f[...,258:262]=features[...,258:262][...,p]
    return f

def basis(features):
    x=features[...,BASIS_INDICES]
    phase=features[...,251:252]
    deficit=np.clip((features[...,244:245]-features[...,245:246])*3,-1,1)
    return np.concatenate((x,x*phase,x*deficit),axis=-1)

class Policy:
    def __init__(self,model,kind='base',setting=None):self.model=model;self.kind=kind;self.setting=setting
    def score(self,features):
        raw=utilities(self.model,features)
        if self.kind=='symmetry':
            average=sum(utilities(self.model,permute_features(features,p)) for p in PERMS)/len(PERMS)
            return (1-self.setting)*raw+self.setting*average
        if self.kind=='residual':
            w=np.asarray(self.setting,dtype=np.float32)
            if w.ndim==2:return raw+np.einsum('bhk,bk->bh',basis(features),w)
            return raw+basis(features)@w
        return raw

def opponents():
    paths=[BASE,ROOT/'artifacts/models/ntw-adaptive-rl-v3b.pt',ROOT/'artifacts/models/ntw-champion-v2.pt',
        ROOT/'artifacts/small-player-iterations/v4/model.pt']
    return [load(p) for p in paths]

def opponent_actions(features,owners,pool,rng):
    b,p,h,_=features.shape;idx=np.zeros((b,p),dtype=np.int64)
    for i,m in enumerate(pool):
        take=owners==i
        if take.any():idx[take]=utilities(m,features[take]).argmin(axis=-1)
    # Four deliberately different, public-information-only heuristics.
    heuristics=[features[...,264]*25+features[...,252]*.01,
        features[...,252],
        features[...,264]*25+features[...,267]*8-features[...,263]*.4+features[...,269]*.2,
        rng.random((b,p,h))]
    for i,scores in enumerate(heuristics,len(pool)):
        take=owners==i
        if take.any():idx[take]=scores[take].argmin(axis=-1)
    return idx

def take_games(game,indices):
    result=object.__new__(SmallGame);result.players=game.players;result.batch=len(indices)
    for name in ('hands','rows','lengths','scores','seen'):setattr(result,name,getattr(game,name)[indices].copy())
    return result

@torch.inference_mode()
def games(policy,pool,players,batch,seed,copies=1):
    root=SmallGame(batch,players,seed);rng=np.random.default_rng(seed+222000000)
    game=take_games(root,np.tile(np.arange(batch),copies))
    owners0=rng.integers(0,len(pool)+4,(batch,players));owners=np.tile(owners0,(copies,1))
    focal=np.tile(np.arange(batch)%players,copies);b=game.batch
    # Reset random streams per turn and tile to give all population members common randomness.
    for turn in range(10):
        f=game.features()
        idx=opponent_actions(f,owners,pool,np.random.default_rng(seed+333000000+turn))
        random_owner=owners==len(pool)+3
        random_scores=np.tile(np.random.default_rng(seed+444000000+turn).random((batch,players,10-turn)),(copies,1,1))
        idx[random_owner]=random_scores[random_owner].argmin(axis=-1)
        idx[np.arange(b),focal]=policy.score(f[np.arange(b),focal]).argmin(axis=-1)
        cards=np.take_along_axis(game.hands,idx[...,None],axis=-1)[...,0];game.step(cards)
    share,_=game.outcome()
    return share[np.arange(b),focal].reshape(copies,batch),game.scores[np.arange(b),focal].reshape(copies,batch)

def dev(policy,pool,players,smoke=False):
    count=32 if smoke else PLAN['development_games'];results=[]
    for offset in range(0,count,256):
        shares,_=games(policy,pool,players,min(256,count-offset),PLAN['development_seed']+players*100000+offset*65537)
        results.extend(shares[0].tolist())
    return float(np.mean(results))

def symmetry(base,pool,dest,smoke):
    settings={};history=[]
    for p in (2,3,4):
        base_score=dev(Policy(base),pool,p,smoke)
        for blend in PLAN['symmetry_blends']:
            score=dev(Policy(base,'symmetry',blend),pool,p,smoke)
            row=dict(players=p,blend=blend,win_rate=score,baseline=base_score);history.append(row)
            print(json.dumps({'branch':'v5',**row}),flush=True)
        settings[str(p)]=max(history[-4:],key=lambda r:r['win_rate'])['blend']
    dump(dest/'model.json',dict(format='ntw-small-player-v1',kind='symmetry',base=serial(base),settings=settings))
    dump(dest/'summary.json',dict(branch='v5-symmetry',settings=settings,development=history,training_games=0,
        note='Inference-time average over all 24 row permutations; baseline blend selected separately per player count. No neural weights trained.'))

def evolution(base,pool,dest,smoke):
    cfg=dict(PLAN['evolution'])
    if smoke:cfg.update(population=8,generations=2,games=16,elites=2,save_every=1)
    settings={};summaries=[]
    for p in (2,3,4):
        rng=np.random.default_rng(cfg['seed']+p*100000)
        mean=np.zeros(27);std=np.full(27,cfg['initial_std']);best=-np.inf;selected=None
        baseline=dev(Policy(base),pool,p,smoke);history=[]
        for generation in range(1,cfg['generations']+1):
            population=rng.normal(mean,std,(cfg['population'],27)).clip(-6,6)
            population[0]=0;population[1]=mean
            fitness=[];penalties=[]
            seed=cfg['seed']+p*100000+generation*104729
            for offset in range(0,len(population),4):
                group=population[offset:offset+4]
                policy=Policy(base,'residual',np.repeat(group,cfg['games'],axis=0))
                wins,costs=games(policy,pool,p,cfg['games'],seed,len(group))
                fitness.extend(wins.mean(axis=1));penalties.extend(costs.mean(axis=1))
            fit=np.array(fitness)-.0002*(population**2).mean(axis=1)
            elite=population[np.argsort(fit)[-cfg['elites']:]]
            mean=.3*mean+.7*elite.mean(axis=0);std=np.maximum(.15,.7*std+.3*elite.std(axis=0))
            row=dict(generation=generation,players=p,games=cfg['population']*cfg['games'],
                population_mean=float(np.mean(fitness)),population_best=float(np.max(fitness)),
                mean=mean.tolist(),std=std.tolist(),fitness=np.asarray(fitness).tolist())
            if generation%cfg['save_every']==0:
                score=dev(Policy(base,'residual',mean),pool,p,smoke);row['development']=score
                dump(dest/f'p{p}-g{generation:02d}.json',dict(weights=mean.tolist(),development=score))
                if score>best:best=score;selected=mean.copy()
            history.append(row)
            with (dest/f'p{p}-training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps({'branch':'v6','players':p,'generation':generation,'development':row.get('development'),'seconds':time.perf_counter()-START}),flush=True)
        settings[str(p)]=selected.tolist()
        summaries.append(dict(players=p,baseline_development=baseline,selected_development=best,
            training_games=cfg['population']*cfg['games']*cfg['generations']))
    dump(dest/'model.json',dict(format='ntw-small-player-v1',kind='residual',base=serial(base),settings=settings,basis_indices=BASIS_INDICES))
    dump(dest/'summary.json',dict(branch='v6-evolution',config=cfg,players=summaries,training_games=sum(s['training_games'] for s in summaries)))

def public_worlds(game,seat,worlds,seed):
    """Discard every true opposing hand. Draw disjoint hands from public unseen cards."""
    b,p,h=game.hands.shape;indices=np.repeat(np.arange(b),worlds)
    result=take_games(game,indices);rng=np.random.default_rng(seed)
    for i in range(b):
        own=game.hands[i,seat];unseen=np.flatnonzero(~game.seen[i])+1
        available=unseen[~np.isin(unseen,own)]
        for j in range(worlds):
            draw=rng.choice(available,(p-1)*h,replace=False).reshape(p-1,h)
            result.hands[i*worlds+j,seat]=own
            for k,s in enumerate(s for s in range(p) if s!=seat):result.hands[i*worlds+j,s]=np.sort(draw[k])
    return result

@torch.inference_mode()
def counterfactual_values(root,pool,base,worlds,seed):
    b,p,h=root.hands.shape
    sampled=public_worlds(root,0,worlds,seed)
    # Candidate-major replication preserves identical hidden worlds and opponent identities.
    game=take_games(sampled,np.tile(np.arange(sampled.batch),h));n=game.batch
    owners0=np.random.default_rng(seed+11).integers(0,len(pool)+4,(sampled.batch,p))
    owners=np.tile(owners0,(h,1))
    root_cards=np.repeat(root.hands[:,:, :],worlds,axis=0)[:,0,:]
    candidate_cards=root_cards.T.reshape(-1)
    for turn in range(h):
        f=game.features();idx=opponent_actions(f,owners,pool,np.random.default_rng(seed+100+turn))
        # Uniform heuristic opponents share a random stream across candidate branches.
        random_owner=owners==len(pool)+3
        random_scores=np.tile(np.random.default_rng(seed+200+turn).random((sampled.batch,p,h-turn)),(h,1,1))
        idx[random_owner]=random_scores[random_owner].argmin(axis=-1)
        idx[:,0]=utilities(base,f[:,0]).argmin(axis=-1)
        cards=np.take_along_axis(game.hands,idx[...,None],axis=-1)[...,0]
        if turn==0:cards[:,0]=candidate_cards
        game.step(cards)
    share,_=game.outcome()
    q=share[:,0].reshape(h,b,worlds).mean(axis=2).T.copy()
    return q,n

@torch.inference_mode()
def roots(pool,base,players,batch,seed,turns):
    game=SmallGame(batch,players,seed);rng=np.random.default_rng(seed+333)
    owners=rng.integers(0,len(pool)+4,(batch,players));saved=[]
    for turn in range(10):
        if turn in turns:saved.append((turn,take_games(game,np.arange(batch))))
        f=game.features();idx=opponent_actions(f,owners,pool,rng)
        idx[:,0]=utilities(base,f[:,0]).argmin(axis=-1)
        cards=np.take_along_axis(game.hands,idx[...,None],axis=-1)[...,0];game.step(cards)
    return saved

def distill(base,pool,dest,smoke):
    cfg=dict(PLAN['counterfactual'])
    if smoke:cfg.update(root_games=8,turns=[6,8],worlds=4,epochs=2,save_every=1)
    models={};summaries=[]
    for p in (2,3,4):
        seed=cfg['seed']+p*100000;torch.manual_seed(seed)
        baseline=dev(Policy(base),pool,p,smoke)
        fs=[];qs=[];ms=[];simulations=0
        for turn,root in roots(pool,base,p,cfg['root_games'],seed,cfg['turns']):
            for offset in range(0,root.batch,8):
                subset=take_games(root,np.arange(offset,min(root.batch,offset+8)))
                q,n=counterfactual_values(subset,pool,base,cfg['worlds'],seed+turn*104729+offset*65537)
                h=subset.hands.shape[-1];f=subset.features()[:,0]
                padded=np.zeros((len(q),10,270),dtype=np.float32);padded[:,:h]=f
                target=np.zeros((len(q),10),dtype=np.float32);target[:,:h]=q
                mask=np.broadcast_to(np.arange(10)<h,(len(q),10)).copy()
                fs.append(padded);qs.append(target);ms.append(mask);simulations+=n
            print(json.dumps({'branch':'v7-data','players':p,'turn':turn,'completion_rollouts':simulations,'seconds':time.perf_counter()-START}),flush=True)
        features=torch.from_numpy(np.concatenate(fs));q=torch.from_numpy(np.concatenate(qs));mask=torch.from_numpy(np.concatenate(ms))
        torch.save(dict(features=features,win_values=q,mask=mask,config=cfg),dest/f'p{p}-counterfactual-data.pt')
        model=copy.deepcopy(base);optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=1e-5)
        with torch.no_grad():
            reference=torch.cat([(-base(f.to(DEVICE))*10/cfg['temperature']).cpu() for f in features.split(128)])
            reference=reference.masked_fill(~mask,-1e9)
            targets=torch.softmax(reference+q*cfg['advantage_scale'],dim=-1)
        best=-np.inf;selected=None;selected_epoch=None
        for epoch in range(1,cfg['epochs']+1):
            losses=[]
            for indices in torch.randperm(len(features)).split(128):
                f=features[indices].to(DEVICE);m=mask[indices].to(DEVICE)
                logits=(-model(f)*10/cfg['temperature']).masked_fill(~m,-1e9)
                logp=torch.log_softmax(logits,dim=-1)
                loss=-(targets[indices].to(DEVICE)*logp).sum(dim=-1).mean()
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite distillation loss')
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                losses.append(float(loss.detach()))
            row=dict(epoch=epoch,loss=float(np.mean(losses)))
            if epoch%cfg['save_every']==0:
                score=dev(Policy(model),pool,p,smoke);row['development']=score
                payload=dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),epoch=epoch,config=cfg)
                torch.save(payload,dest/f'p{p}-e{epoch:02d}.pt')
                if score>best:best=score;selected=copy.deepcopy(model);selected_epoch=epoch;torch.save(payload,dest/f'p{p}-model.pt')
                print(json.dumps({'branch':'v7-train','players':p,**row}),flush=True)
            with (dest/f'p{p}-training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        models[str(p)]=serial(selected)
        summaries.append(dict(players=p,root_games=cfg['root_games'],public_states=len(features),completion_rollouts=simulations,
            selected_epoch=selected_epoch,baseline_development=baseline,selected_development=best))
    dump(dest/'model.json',dict(format='ntw-small-player-v1',kind='specialist',models=models))
    dump(dest/'summary.json',dict(branch='v7-counterfactual',config=cfg,players=summaries,
        root_games=sum(s['root_games'] for s in summaries),completion_rollouts=sum(s['completion_rollouts'] for s in summaries),
        information='Teacher samples only own hand, public board, scores and seen cards. True opposing hands are discarded. Uniform unseen-card belief; no learned history-conditioned posterior.'))

def main():
    global START
    if Path(sys.prefix).name.lower()!='ntw-ai' or DEVICE.type!='cuda':raise RuntimeError('Use ntw-ai with CUDA')
    torch.set_num_threads(1);START=time.perf_counter()
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--branch',choices=['v5','v6','v7','all'],default='all');args=parser.parse_args()
    plan=OUT/'experiment-plan.json'
    if not plan.exists():dump(plan,dict(**PLAN,base=str(BASE),base_sha256=sha(BASE)))
    base=load(BASE);pool=opponents()
    for name,fn in [('v5',symmetry),('v6',evolution),('v7',distill)]:
        if args.branch not in ('all',name):continue
        dest=OUT/(('smoke-' if args.smoke else '')+name);dest.mkdir(parents=True,exist_ok=True)
        if (dest/'summary.json').exists():continue
        if any(dest.iterdir()):raise RuntimeError(f'Partial branch in {dest}; preserve it under a new run name before retrying')
        dump(dest/'protocol.json',dict(plan_sha256=sha(plan),sources={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'training/small_player_env.py',ROOT/'training/train_policy.py']},gpu=torch.cuda.get_device_name(0),torch=str(torch.__version__),smoke=args.smoke))
        fn(base,pool,dest,args.smoke)
        print(json.dumps({'completed':name,'seconds':time.perf_counter()-START}),flush=True)
    names=[('smoke-' if args.smoke else '')+v for v in ('v5','v6','v7')]
    if all((OUT/n/'summary.json').exists() for n in names):dump(OUT/('smoke-complete.json' if args.smoke else 'training-complete.json'),dict(status='complete',versions=names,seconds=time.perf_counter()-START))

if __name__=='__main__':main()
