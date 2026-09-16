"""Player-count-specialized win optimization against the learned opponent league.

This is a development simulator; its evaluation cannot certify an upgrade.
Actual frozen opponents are used separately by progressive_campaign.py.
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
from torch import nn
import explore_small_strategies as e
from progressive_planner import Planner
from small_player_env import SmallGame

CONFIG=dict(batch=512,updates=128,epochs=4,minibatch=512,lr=.00002,clip=.15,
    entropy=.005,anchor_kl=.01,gae_lambda=.95,critic_weight=.5,bullhead_weight=.002,
    evaluate_every=16,evaluation_games=2048,players=[3,4],deck_size=104,
    initial_policy='artifacts/progressive-upgrades/continuation-v1/model.pt',
    opponent_proxy='artifacts/progressive-upgrades/opponent-proxies-v2',
    selection='Largest mean deterministic evaluation win share across 3/4-player proxy development games',
    seed_namespace='progressive-ppo-proxy-v1',
    role='Training and proxy development only; neither proxy wins nor checkpoints count as accepted upgrades')


def seed(tag):
    return int.from_bytes(hashlib.sha256((CONFIG['seed_namespace']+':'+tag).encode()).digest()[:8],'little')|(1<<63)


class Critic(nn.Module):
    def __init__(self):
        super().__init__();self.layers=nn.Sequential(nn.Linear(252,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,1))
    def forward(self,x):return self.layers(x).squeeze(-1)


def environment(planner,p,b,tag):
    source=SmallGame(b,p,seed(tag))
    source.rows=source.rows[:,::-1].copy() # Match the arena's last-card-first row order.
    table=planner.torch_env.TorchGame(source,e.DEVICE)
    rng=np.random.default_rng(seed(tag+'-types'));owners=np.full((b,p),-1,dtype=np.int64)
    for i in range(b):owners[i,1:]=rng.choice(8,p-1,replace=False)
    groups=[torch.as_tensor(np.flatnonzero(owners.reshape(-1)==i),device=e.DEVICE) for i in range(8)]
    return table,groups


@torch.inference_mode()
def opponents(planner,table,common,action,groups):
    b,p,h=table.hands.shape;c=common.reshape(b*p,252);a=action.reshape(b*p,h,18)
    choices=torch.zeros(b*p,dtype=torch.int64,device=e.DEVICE)
    for i,model in enumerate(planner.pool):
        ix=groups[i]
        if not len(ix):continue
        utilities=planner.torch_env.split_utilities(model,c[ix],a[ix])
        choices[ix]=torch.distributions.Categorical(logits=-utilities).sample() if i>=3 else utilities.argmin(-1)
    choices[groups[7]]=torch.randint(h,(len(groups[7]),),device=e.DEVICE)
    return choices.reshape(b,p)


@torch.inference_mode()
def rollout(planner,policy,critic,p,b,tag,stochastic):
    table,groups=environment(planner,p,b,tag);trajectory=[]
    for turn in range(10):
        h=10-turn;common,action=table.features()
        features=torch.cat((common[:,0,None,:].expand(-1,h,-1),action[:,0]),dim=-1)
        logits=-policy(features)*10
        distribution=torch.distributions.Categorical(logits=logits)
        own=distribution.sample() if stochastic else logits.argmax(-1)
        indices=opponents(planner,table,common,action,groups);indices[:,0]=own
        cards=table.hands.gather(2,indices[:,:,None]).squeeze(-1)
        before=table.scores[:,0].clone();table.step(cards)
        reward=-CONFIG['bullhead_weight']*(table.scores[:,0]-before)
        if turn==9:reward+=table.outcome()[0][:,0]
        if stochastic:
            padded=torch.zeros((b,10,270),device=e.DEVICE);padded[:,:h]=features
            mask=torch.arange(10,device=e.DEVICE)[None].expand(b,-1)<h
            trajectory.append(dict(features=padded,mask=mask,action=own,old_logp=distribution.log_prob(own),
                value=critic(common[:,0]),reward=reward))
    shares=table.outcome()[0][:,0]
    return trajectory,float(shares.mean()),float(table.scores[:,0].mean())


@torch.inference_mode()
def evaluate(planner,policy,critic,tag):
    results={}
    # Fixed stochastic streams for fair development checkpoint comparisons.
    saved=torch.random.get_rng_state();saved_cuda=torch.cuda.get_rng_state()
    for p in CONFIG['players']:
        torch.manual_seed(seed('evaluation-actions-'+str(p)))
        wins=[];costs=[]
        for block in range(CONFIG['evaluation_games']//CONFIG['batch']):
            _,win,cost=rollout(planner,policy,critic,p,CONFIG['batch'],f'evaluation-{p}-{block}',False)
            wins.append(win);costs.append(cost)
        results[str(p)]=dict(win_rate=float(np.mean(wins)),bullheads=float(np.mean(costs)))
    torch.random.set_rng_state(saved);torch.cuda.set_rng_state(saved_cuda)
    return results


def main(args):
    out=e.ROOT/'artifacts/progressive-upgrades'/args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():print('Already complete');return
    if (out/'training.jsonl').exists():raise RuntimeError('Do not silently restart an incomplete PPO run in the same directory')
    torch.set_num_threads(1);e.DEVICE=torch.device('cuda');torch.manual_seed(seed('initial'))
    planner=Planner(dict(engine='torch',device='cuda',opponents='actual-proxy',proxy_directory=CONFIG['opponent_proxy']))
    policy=e.load(e.ROOT/CONFIG['initial_policy']);anchor=e.load(e.ROOT/CONFIG['initial_policy']);critic=Critic().to(e.DEVICE)
    optimizer=torch.optim.Adam(list(policy.parameters())+list(critic.parameters()),lr=CONFIG['lr'])
    e.dump(out/'protocol.json',dict(**CONFIG,source_sha256=e.sha(Path(__file__)),
        dependency_sha256={str(path):e.sha(path) for path in [e.ROOT/CONFIG['initial_policy'],
            *[e.ROOT/CONFIG['opponent_proxy']/(k+'.pt') for k in ('dirv-10000','alpha-2000','mcs','champion-search')]]}))
    initial=evaluate(planner,policy,critic,'initial');e.dump(out/'initial-development.json',initial)
    best=float(np.mean([v['win_rate'] for v in initial.values()]));start=time.perf_counter();evaluations=[]
    for update in range(1,CONFIG['updates']+1):
        p=CONFIG['players'][(update-1)%len(CONFIG['players'])]
        trajectory,win,cost=rollout(planner,policy,critic,p,CONFIG['batch'],f'train-{update}',True)
        advantage=torch.zeros(CONFIG['batch'],device=e.DEVICE);next_value=torch.zeros_like(advantage)
        for item in reversed(trajectory):
            residual=item['reward']+next_value-item['value']
            advantage=residual+CONFIG['gae_lambda']*advantage
            item['advantage']=advantage.clone();item['return']=advantage+item['value'];next_value=item['value']
        data={k:torch.cat([item[k] for item in trajectory]) for k in ('features','mask','action','old_logp','advantage','return')}
        data['advantage']=(data['advantage']-data['advantage'].mean())/data['advantage'].std().clamp_min(1e-6)
        losses=[];kls=[]
        for epoch in range(CONFIG['epochs']):
            for ix in torch.randperm(len(data['action']),device=e.DEVICE).split(CONFIG['minibatch']):
                features=data['features'][ix];mask=data['mask'][ix];actions=data['action'][ix]
                logits=(-policy(features)*10).masked_fill(~mask,-1e9);distribution=torch.distributions.Categorical(logits=logits)
                logp=distribution.log_prob(actions);ratio=(logp-data['old_logp'][ix]).exp();adv=data['advantage'][ix]
                policy_loss=-torch.minimum(ratio*adv,ratio.clamp(1-CONFIG['clip'],1+CONFIG['clip'])*adv).mean()
                value=critic(features[:,0,:252]);critic_loss=(value-data['return'][ix]).square().mean()
                with torch.no_grad():anchor_prob=torch.softmax((-anchor(features)*10).masked_fill(~mask,-1e9),dim=-1)
                anchor_kl=torch.nn.functional.kl_div(torch.log_softmax(logits,-1),anchor_prob,reduction='batchmean')
                loss=policy_loss+CONFIG['critic_weight']*critic_loss-CONFIG['entropy']*distribution.entropy().mean()+CONFIG['anchor_kl']*anchor_kl
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite PPO loss')
                optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(policy.parameters())+list(critic.parameters()),.5);optimizer.step()
                losses.append(float(loss.detach()));kls.append(float((data['old_logp'][ix]-logp).mean().detach()))
            if np.mean(kls[-10:])>.025:break
        row=dict(update=update,players=p,training_games=update*CONFIG['batch'],sampled_win_rate=win,
            sampled_bullheads=cost,loss=float(np.mean(losses)),approx_kl=float(np.mean(kls)),seconds=time.perf_counter()-start)
        if update%CONFIG['evaluate_every']==0:
            metrics=evaluate(planner,policy,critic,str(update));row['development']=metrics
            average=float(np.mean([v['win_rate'] for v in metrics.values()]));evaluations.append(dict(update=update,results=metrics))
            snapshot=dict(state_dict={k:v.cpu().clone() for k,v in policy.state_dict().items()},update=update,development=metrics)
            torch.save(snapshot,out/f'checkpoint-{update:03d}.pt')
            if average>best:
                best=average;torch.save(snapshot,out/'model.pt');e.dump(out/'model.json',e.serial(policy))
        with (out/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        if update%8==0:print(json.dumps(row),flush=True)
    e.dump(out/'complete.json',dict(training_games=CONFIG['updates']*CONFIG['batch'],initial=initial,
        best_development_mean=best,evaluations=evaluations,selected_checkpoint_exists=(out/'model.pt').exists()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--players',type=int,choices=[3,4],required=True);parser.add_argument('--updates',type=int,default=256)
    args=parser.parse_args()
    CONFIG.update(players=[args.players],updates=args.updates,evaluate_every=32,
        initial_policy='artifacts/progressive-upgrades/ppo-proxy-v1/model.pt',
        seed_namespace=f'progressive-ppo-specialist-{args.players}-v1',
        selection='Largest deterministic proxy development win share for this player count; separate 3/4-player policies')
    main(args)
