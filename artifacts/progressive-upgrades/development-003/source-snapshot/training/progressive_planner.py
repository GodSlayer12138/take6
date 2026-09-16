"""Public-information rollout planners for the three-success upgrade campaign."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import sys
import numpy as np
import torch
import explore_small_strategies as e
from small_player_env import SmallGame
from train_policy import encode_candidate,CandidateValueNet

ROOT=e.ROOT
OUT=ROOT/'artifacts/progressive-upgrades'

def model_from_json(value,device):
    model=CandidateValueNet().to(device)
    with torch.no_grad():
        for layer,values in zip(model.layers,value['layers']):
            layer.weight.copy_(torch.tensor(values['weight'],device=device));layer.bias.copy_(torch.tensor(values['bias'],device=device))
    return model.eval()

def public_game(observation):
    p=observation['playerCount'];h=len(observation['hand'])
    if p not in (3,4) or observation['deckSize']!=104:raise ValueError('Campaign only supports 3/4 players and 104 cards')
    game=object.__new__(SmallGame);game.players=p;game.batch=1
    game.hands=np.zeros((1,p,h),dtype=np.int64);game.hands[0,0]=sorted(observation['hand'])
    game.rows=np.zeros((1,4,6),dtype=np.int64);game.lengths=np.array([[len(row) for row in observation['rows']]])
    for i,row in enumerate(observation['rows']):game.rows[0,i,:len(row)]=row
    game.seen=np.zeros((1,104),dtype=bool);game.seen[0,np.asarray(observation['seenCards'])-1]=True
    game.scores=np.array([observation.get('scores',[0]*p)],dtype=np.float32)
    return game

@torch.inference_mode()
def fast_utilities(model,features,device):
    """Factor the common 252 inputs once per hand, rather than once per candidate."""
    original=features.shape[:-1];h=features.shape[-2];flat=features.reshape(-1,h,270)
    outputs=[];first=model.layers[0]
    for group in np.array_split(flat,max(1,(len(flat)+2047)//2048)):
        f=torch.from_numpy(group).to(device)
        common=torch.nn.functional.linear(f[:,0,:252],first.weight[:,:252],first.bias)
        action=torch.nn.functional.linear(f[:,:,252:],first.weight[:,252:],None)
        values=torch.relu(common[:,None,:]+action)
        for layer in model.layers[1:-1]:values=torch.relu(layer(values))
        outputs.append((model.layers[-1](values).squeeze(-1)*10).cpu().numpy())
    return np.concatenate(outputs).reshape(original)

class Planner:
    def __init__(self,config,device='cpu'):
        self.config=config;self.device=torch.device(config.get('device',device));e.DEVICE=self.device
        torch.set_num_threads(1)
        self.base=e.load(e.BASE)
        self.pool=[self.base,e.load(ROOT/'artifacts/models/ntw-adaptive-rl-v3b.pt'),
            e.load(ROOT/'artifacts/models/ntw-champion-v2.pt'),e.load(ROOT/'artifacts/small-player-iterations/v4/model.pt')]
        self.continuation=self.base
        if config.get('continuation_checkpoint'):self.continuation=e.load(ROOT/config['continuation_checkpoint'])
        if config.get('opponents')=='actual-proxy':
            old=json.loads((ROOT/'artifacts/small-player-exploration/evaluation-manifest.json').read_text(encoding='utf-8'))
            mapping={entry['id']:entry for entry in old['entrants']}
            self.pool=[model_from_json(json.loads(Path(mapping[name]['path']).read_text(encoding='utf-8')),self.device)
                for name in ('ntw-adaptive-rl-v3b','ntw-adaptive-arena-v8-distill','ntw-champion')]
            proxy_root=ROOT/config.get('proxy_directory','artifacts/progressive-upgrades/opponent-proxies-v1')
            self.pool += [e.load(proxy_root/(name+'.pt')) for name in ('dirv-10000','alpha-2000','mcs','champion-search')]
        if config.get('engine')=='torch':
            path=Path(__file__).with_name('progressive_torch_env.py')
            name='torch_env_'+hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            if name not in sys.modules:
                spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
                sys.modules[name]=module;spec.loader.exec_module(module)
            self.torch_env=sys.modules[name]
        if config.get('belief_oversample'):
            if config.get('engine')!='torch':raise ValueError('History belief requires the Torch engine')
            path=Path(__file__).with_name('progressive_belief.py')
            name='belief_'+hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            if name not in sys.modules:
                spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
                sys.modules[name]=module;spec.loader.exec_module(module)
            self.belief=sys.modules[name]

    def utilities(self,model,features):return fast_utilities(model,features,self.device)

    def opponent_actions(self,features,owners,uniforms):
        b,p,h,_=features.shape;actions=np.zeros((b,p),dtype=np.int64)
        for i,model in enumerate(self.pool):
            take=owners==i
            if not take.any():continue
            utilities=self.utilities(model,features[take])
            if self.config.get('opponents')=='actual-proxy' and i>=3 and self.config.get('stochastic_proxy',True):
                logits=-utilities/float(self.config.get('proxy_temperature',1.))
                logits-=logits.max(axis=-1,keepdims=True);prob=np.exp(logits);prob/=prob.sum(axis=-1,keepdims=True)
                actions[take]=np.minimum((uniforms[take,None]>prob.cumsum(axis=-1)).sum(axis=-1),h-1)
            else:actions[take]=utilities.argmin(axis=-1)
        if self.config.get('opponents')=='actual-proxy':
            take=owners==7;actions[take]=np.minimum((uniforms[take]*h).astype(np.int64),h-1)
        else:
            heuristics=[features[...,264]*25+features[...,252]*.01,features[...,252],
                features[...,264]*25+features[...,267]*8-features[...,263]*.4+features[...,269]*.2]
            for i,score in enumerate(heuristics,len(self.pool)):
                take=owners==i;actions[take]=score[take].argmin(axis=-1)
            take=owners==len(self.pool)+3;actions[take]=np.minimum((uniforms[take]*h).astype(np.int64),h-1)
        return actions

    @torch.inference_mode()
    def choose(self,observation):
        if self.config.get('engine')=='torch':return self.choose_torch(observation)
        e.DEVICE=self.device
        started=time.perf_counter();hand=sorted(observation['hand']);h=len(hand)
        if h==1:return dict(card=hand[0],seconds=time.perf_counter()-started)
        cfg=self.config;root=public_game(observation)
        f=np.array([[encode_candidate(observation,c) for c in hand]],dtype=np.float32)
        permutations=np.concatenate([e.permute_features(f,p) for p in e.PERMS])
        prior=self.utilities(self.base,permutations).mean(axis=0)
        worlds=int(cfg.get('worlds',32));seed=int(observation.get('seed',1))+int(cfg.get('seed_offset',600013))
        if cfg.get('endgame_multiplier') and h<=3:worlds*=int(cfg['endgame_multiplier'])
        sampled=e.public_worlds(root,0,worlds,seed)
        game=e.take_games(sampled,np.tile(np.arange(worlds),h))
        rng=np.random.default_rng(seed+11)
        profile=cfg.get('opponents','mixed')
        if profile=='neural':owners0=rng.integers(0,len(self.pool),(worlds,root.players))
        elif profile=='base':owners0=np.zeros((worlds,root.players),dtype=np.int64)
        elif profile=='actual-proxy':
            owners0=np.zeros((worlds,root.players),dtype=np.int64)
            for i in range(worlds):owners0[i,1:]=rng.choice(8,root.players-1,replace=False)
        else:owners0=rng.integers(0,len(self.pool)+4,(worlds,root.players))
        owners0[:,0]=-1
        owners=np.tile(owners0,(h,1));initial_scores=game.scores.copy()
        immediate=None
        for turn in range(h):
            uniforms=np.random.default_rng(seed+200+turn).random((worlds,root.players))
            if turn==0:
                features=sampled.features();actions=self.opponent_actions(features,owners0,uniforms)
                first_cards=np.take_along_axis(sampled.hands,actions[...,None],axis=-1)[...,0]
                cards=np.tile(first_cards,(h,1));cards[:,0]=np.repeat(hand,worlds)
            else:
                features=game.features();actions=self.opponent_actions(features,owners,np.tile(uniforms,(h,1)))
                actions[:,0]=self.utilities(self.continuation,features[:,0]).argmin(axis=-1)
                cards=np.take_along_axis(game.hands,actions[...,None],axis=-1)[...,0]
            costs=game.step(cards)
            if turn==0:immediate=costs[:,0].copy()
        share,rank=game.outcome()
        reward=share[:,0].astype(np.float64)
        reward-=float(cfg.get('bullhead_weight',.002))*(game.scores[:,0]-initial_scores[:,0])
        reward-=float(cfg.get('rank_weight',0))*(rank[:,0]-1)/(root.players-1)
        values=reward.reshape(h,worlds).mean(axis=1)
        # A small prior influence may reduce sampling noise; selected only on development games.
        utility=-values+float(cfg.get('prior_weight',0))*(prior-prior.min())
        chosen=int(np.argmin(utility))
        return dict(card=hand[chosen],seconds=time.perf_counter()-started,
            candidates=hand,values=values.tolist(),utility=utility.tolist(),worlds=worlds,
            mean_immediate=immediate.reshape(h,worlds).mean(axis=1).tolist())

    @torch.inference_mode()
    def choose_torch(self,observation):
        started=time.perf_counter();hand=sorted(observation['hand']);h=len(hand)
        if h==1:return dict(card=hand[0],seconds=time.perf_counter()-started)
        cfg=self.config;root=public_game(observation);p=root.players
        f=np.array([[encode_candidate(observation,c) for c in hand]],dtype=np.float32)
        prior=self.utilities(self.base,np.concatenate([e.permute_features(f,perm) for perm in e.PERMS])).mean(axis=0)
        worlds=int(cfg.get('worlds',32));seed=int(observation.get('seed',1))+int(cfg.get('seed_offset',600013))
        if cfg.get('endgame_multiplier') and h<=3:worlds*=int(cfg['endgame_multiplier'])
        proposals=worlds*int(cfg.get('belief_oversample',1)) if observation.get('history') else worlds
        sampled=e.public_worlds(root,0,proposals,seed);rng=np.random.default_rng(seed+11);profile=cfg.get('opponents','mixed')
        if profile=='neural':owners=rng.integers(0,len(self.pool),(proposals,p))
        elif profile=='base':owners=np.zeros((proposals,p),dtype=np.int64)
        elif profile=='actual-proxy':
            owners=np.zeros((proposals,p),dtype=np.int64)
            for i in range(proposals):owners[i,1:]=rng.choice(8,p-1,replace=False)
        else:owners=rng.integers(0,len(self.pool)+4,(proposals,p))
        owners[:,0]=-1
        belief_diagnostics=None
        if cfg.get('belief_oversample'):
            sampled,owners,belief_diagnostics=self.belief.posterior(self,root,sampled,owners,observation,worlds,seed)
        owner_count=8 if profile=='actual-proxy' else len(self.pool)+4
        def groups(ids):return [torch.as_tensor(np.flatnonzero(ids.reshape(-1)==i),device=self.device) for i in range(owner_count)]
        root_groups=groups(owners);all_groups=groups(np.tile(owners,(h,1)))
        initial=self.torch_env.TorchGame(sampled,self.device);game=initial.repeat(h);initial_scores=game.scores.clone()
        def choose_actions(table,indices,uniforms,focal):
            common,action=table.features();b=table.batch;n=table.hands.shape[-1]
            c=common.reshape(b*p,252);a=action.reshape(b*p,n,18);u=uniforms.reshape(-1)
            choice=torch.zeros(b*p,dtype=torch.int64,device=self.device)
            for i,model in enumerate(self.pool):
                take=indices[i]
                if not len(take):continue
                utility=self.torch_env.split_utilities(model,c[take],a[take])
                if profile=='actual-proxy' and i>=3 and cfg.get('stochastic_proxy',True):
                    probability=torch.softmax(-utility/float(cfg.get('proxy_temperature',1.)),dim=-1)
                    choice[take]=(u[take,None]>probability.cumsum(dim=-1)).sum(dim=-1).clamp_max(n-1)
                else:choice[take]=utility.argmin(dim=-1)
            if profile=='actual-proxy':
                take=indices[7];choice[take]=(u[take]*n).long().clamp_max(n-1)
            else:
                scores=[a[...,12]*25+a[...,0]*.01,a[...,0],a[...,12]*25+a[...,15]*8-a[...,11]*.4+a[...,17]*.2]
                for i,score in enumerate(scores,len(self.pool)):
                    take=indices[i];choice[take]=score[take].argmin(dim=-1)
                take=indices[len(self.pool)+3];choice[take]=(u[take]*n).long().clamp_max(n-1)
            choice=choice.reshape(b,p)
            if focal:choice[:,0]=self.torch_env.split_utilities(self.continuation,common[:,0],action[:,0]).argmin(dim=-1)
            return choice
        immediate=None
        for turn in range(h):
            uniforms=torch.tensor(np.random.default_rng(seed+200+turn).random((worlds,p)),device=self.device)
            if turn==0:
                actions=choose_actions(initial,root_groups,uniforms,False)
                cards=initial.hands.gather(2,actions[:,:,None]).squeeze(-1).repeat(h,1)
                cards[:,0]=torch.tensor(hand,device=self.device).repeat_interleave(worlds)
            else:
                actions=choose_actions(game,all_groups,uniforms.repeat(h,1),True)
                cards=game.hands.gather(2,actions[:,:,None]).squeeze(-1)
            costs=game.step(cards)
            if turn==0:immediate=costs[:,0].clone()
        shares,rank=game.outcome();score_delta=(game.scores[:,0]-initial_scores[:,0]).cpu().numpy()
        reward=shares[:,0].cpu().numpy().astype(np.float64)
        reward-=float(cfg.get('bullhead_weight',.002))*score_delta
        reward-=float(cfg.get('rank_weight',0))*(rank[:,0].cpu().numpy()-1)/(p-1)
        values=reward.reshape(h,worlds).mean(axis=1);utility=-values+float(cfg.get('prior_weight',0))*(prior-prior.min())
        result=dict(card=hand[int(utility.argmin())],candidates=hand,values=values.tolist(),utility=utility.tolist(),worlds=worlds,
            mean_immediate=immediate.cpu().numpy().reshape(h,worlds).mean(axis=1).tolist())
        result['seconds']=time.perf_counter()-started
        if belief_diagnostics is not None:result['belief']=belief_diagnostics
        return result

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu');args=parser.parse_args()
    config=dict(worlds=32,opponents='mixed',prior_weight=.005)
    planner=Planner(config,args.device)
    observations=[]
    for p in (3,4):
        game=SmallGame(1,p,990103+p)
        for turn in range(10):
            if turn in (0,3,6,8):
                obs=game.record(0,0);obs['seed']=990100+turn;observations.append(obs)
            game.step(game.hands[:,:,0])
    for obs in observations:
        result=planner.choose(obs)
        print(json.dumps({'players':obs['playerCount'],'hand':len(obs['hand']),**result}),flush=True)
