"""Infer hypothetical remaining hands and policy types from public past actions.

Importance sampling, with tempered action likelihoods and an effective-sample-size
floor. No actual opponent hand or policy identity is accepted by this module.
"""
import numpy as np
import torch
from small_player_env import SmallGame
from explore_small_strategies import take_games


def history_states(root, history):
    """Recover pre-turn scores and public cards in the focal player's seat order."""
    p=root.players;costs=[];actions=[]
    for item in history:
        played=np.array([item['ownCard']]+item['opponentCards'],dtype=np.int64)
        assert len(played)==p and len(set(played))==p
        table=object.__new__(SmallGame);table.batch=1;table.players=p
        table.rows=np.zeros((1,4,6),dtype=np.int64)
        table.lengths=np.array([[len(r) for r in item['rows']]],dtype=np.int64)
        for i,row in enumerate(item['rows']):table.rows[0,i,:len(row)]=row
        table.hands=played[None,:,None].copy();table.scores=np.zeros((1,p),dtype=np.float32)
        table.seen=np.zeros((1,104),dtype=bool)
        table.seen[0,np.array(item['seenCards'])-1]=True
        costs.append(table.step(played[None])[0]);actions.append(played)
    if not history:return [],np.empty((0,p),dtype=np.int64)
    actions=np.stack(actions);costs=np.stack(costs)
    initial=root.scores[0]-costs.sum(axis=0)
    scores=initial[None]+np.concatenate((np.zeros((1,p),dtype=np.float32),costs.cumsum(axis=0)[:-1]))
    return scores,actions


def historical_worlds(sampled, item, scores, later_actions):
    """At an earlier turn, append cards that are now public but were then held."""
    b,p,h=sampled.hands.shape
    table=take_games(sampled,np.arange(b))
    known=np.broadcast_to(later_actions.T[None],(b,p,len(later_actions)))
    table.hands=np.sort(np.concatenate((sampled.hands,known),axis=-1),axis=-1)
    table.rows.fill(0)
    for i,row in enumerate(item['rows']):table.rows[:,i,:len(row)]=row
    table.lengths[:]=np.array([len(row) for row in item['rows']])
    table.scores[:]=scores;table.seen.fill(False)
    table.seen[:,np.array(item['seenCards'])-1]=True
    assert (np.diff(table.hands,axis=-1)>0).all(), 'A historical hand contains duplicate cards'
    return table


@torch.inference_mode()
def posterior(planner, root, sampled, owners, observation, target_worlds, seed):
    cfg=planner.config;history=observation.get('history',[]);m=sampled.batch;p=root.players
    if not history:return take_games(sampled,np.arange(target_worlds)),owners[:target_worlds],dict(history_turns=0,ess=m)
    scores,actions=history_states(root,history)
    window=min(len(history),int(cfg.get('belief_history',3)))
    log_weight=np.zeros(m,dtype=np.float64)
    eps=float(cfg.get('belief_epsilon',.15));temperature=float(cfg.get('belief_temperature',.7))
    if not (0<eps<1 and temperature>0):raise ValueError('Invalid belief smoothing')
    for t in range(len(history)-window,len(history)):
        past=historical_worlds(sampled,history[t],scores[t],actions[t:])
        h=past.hands.shape[-1];choice=(past.hands==actions[t][None,:,None]).argmax(axis=-1)
        assert (past.hands==actions[t][None,:,None]).sum(axis=-1).min()==1
        if cfg.get('engine')=='torch':
            table=planner.torch_env.TorchGame(past,planner.device);common,features=table.features()
        else:features=past.features()
        for i,model in enumerate(planner.pool):
            take=owners==i
            if not take.any():continue
            if cfg.get('engine')=='torch':
                ix=torch.as_tensor(take,device=planner.device)
                utilities=planner.torch_env.split_utilities(model,common[ix],features[ix]).cpu().numpy()
            else:utilities=planner.utilities(model,features[take])
            temp=float(cfg.get('proxy_temperature',1.)) if cfg.get('opponents')=='actual-proxy' and i>=3 else temperature
            logits=-utilities/temp;logits-=logits.max(axis=-1,keepdims=True)
            probability=np.exp(logits);probability/=probability.sum(axis=-1,keepdims=True)
            observed=probability[np.arange(len(probability)),choice[take]]
            # An epsilon-uniform component tolerates model error and avoids zero weights.
            likelihood=(1-eps)*observed+eps/h
            rows=np.where(take)[0];np.add.at(log_weight,rows,np.log(likelihood))
        # The remaining supported type is uniform random, with constant likelihood.
        if cfg.get('opponents')=='actual-proxy':log_weight+=(owners==7).sum(axis=1)*np.log(1/h)
        elif cfg.get('opponents') not in ('neural','base'):raise ValueError('Belief requires neural, base, or actual-proxy opponents')
    power=float(cfg.get('belief_strength',.5));minimum=float(cfg.get('belief_min_ess',.2))*m
    for _ in range(24):
        weights=np.exp((log_weight-log_weight.max())*power);weights/=weights.sum()
        ess=float(1/np.square(weights).sum())
        if ess>=minimum:break
        power*=.8
    mixture=float(cfg.get('belief_prior_mix',.05));weights=(1-mixture)*weights+mixture/m
    rng=np.random.default_rng(seed+820003)
    points=(np.arange(target_worlds)+rng.random(target_worlds))/target_worlds
    indices=np.searchsorted(weights.cumsum(),points).clip(max=m-1)
    return take_games(sampled,indices),owners[indices],dict(history_turns=window,ess=ess,
        proposals=m,worlds=target_worlds,likelihood_power=power,unique_worlds=len(np.unique(indices)))
