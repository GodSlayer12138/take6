"""Public-information rollout planners for the three-success upgrade campaign."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import copy
import json
from pathlib import Path
import time
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

class Planner:
    def __init__(self,config,device='cpu'):
        self.config=config;self.device=torch.device(device);e.DEVICE=self.device
        torch.set_num_threads(1)
        self.base=e.load(e.BASE)
        self.pool=[self.base,e.load(ROOT/'artifacts/models/ntw-adaptive-rl-v3b.pt'),
            e.load(ROOT/'artifacts/models/ntw-champion-v2.pt'),e.load(ROOT/'artifacts/small-player-iterations/v4/model.pt')]
        self.continuation=self.base
        if config.get('continuation_checkpoint'):self.continuation=e.load(ROOT/config['continuation_checkpoint'])

    @torch.inference_mode()
    def choose(self,observation):
        started=time.perf_counter();hand=sorted(observation['hand']);h=len(hand)
        if h==1:return dict(card=hand[0],seconds=time.perf_counter()-started)
        cfg=self.config;root=public_game(observation)
        f=np.array([[encode_candidate(observation,c) for c in hand]],dtype=np.float32)
        prior=e.Policy(self.base,'symmetry',1.).score(f)[0]
        worlds=int(cfg.get('worlds',32));seed=int(observation.get('seed',1))+int(cfg.get('seed_offset',600013))
        if cfg.get('endgame_multiplier') and h<=3:worlds*=int(cfg['endgame_multiplier'])
        sampled=e.public_worlds(root,0,worlds,seed)
        game=e.take_games(sampled,np.tile(np.arange(worlds),h))
        rng=np.random.default_rng(seed+11)
        profile=cfg.get('opponents','mixed')
        if profile=='neural':owners0=rng.integers(0,len(self.pool),(worlds,root.players))
        elif profile=='base':owners0=np.zeros((worlds,root.players),dtype=np.int64)
        else:owners0=rng.integers(0,len(self.pool)+4,(worlds,root.players))
        owners=np.tile(owners0,(h,1));initial_scores=game.scores.copy()
        immediate=None
        for turn in range(h):
            features=game.features()
            actions=e.opponent_actions(features,owners,self.pool,np.random.default_rng(seed+100+turn))
            random_owner=owners==len(self.pool)+3
            random_scores=np.tile(np.random.default_rng(seed+200+turn).random((worlds,root.players,h-turn)),(h,1,1))
            actions[random_owner]=random_scores[random_owner].argmin(axis=-1)
            actions[:,0]=e.utilities(self.continuation,features[:,0]).argmin(axis=-1)
            cards=np.take_along_axis(game.hands,actions[...,None],axis=-1)[...,0]
            if turn==0:cards[:,0]=np.repeat(hand,worlds)
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
