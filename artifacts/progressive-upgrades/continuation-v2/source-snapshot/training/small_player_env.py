"""Batched 104-card environment and exact 270-feature public observations, 2-4 players."""
import numpy as np

POINTS=np.array([0]+[7 if c==55 else 5 if c%11==0 else 3 if c%10==0 else 2 if c%5==0 else 1 for c in range(1,105)],dtype=np.float32)

class SmallGame:
    def __init__(self,batch,players,seed):
        if players not in (2,3,4):raise ValueError('Only 2-4 players are in scope')
        self.batch=batch;self.players=players
        rng=np.random.default_rng(seed)
        cards=np.stack([rng.permutation(104)+1 for _ in range(batch)])
        self.hands=np.sort(cards[:,:players*10].reshape(batch,players,10),axis=-1)
        self.rows=np.zeros((batch,4,6),dtype=np.int64);self.rows[:,:,0]=cards[:,-4:]
        self.lengths=np.ones((batch,4),dtype=np.int64);self.scores=np.zeros((batch,players),dtype=np.float32)
        self.seen=np.zeros((batch,104),dtype=bool)
        self.seen[np.arange(batch)[:,None],self.rows[:,:,0]-1]=True

    def features(self):
        b,p,h=self.hands.shape;batch=np.arange(b)[:,None]
        handmask=np.zeros((b,p,104),dtype=bool)
        handmask[np.arange(b)[:,None,None],np.arange(p)[None,:,None],self.hands-1]=True
        tails=np.take_along_axis(self.rows,(self.lengths-1)[:,:,None],axis=2)[:,:,0]
        costs=POINTS[self.rows].sum(axis=2)
        summary=np.stack((self.lengths/5,np.clip(costs/25,0,1),tails/104),axis=-1).reshape(b,12)
        others=np.broadcast_to(self.scores[:,None,:],(b,p,p)).copy()
        diag=np.eye(p,dtype=bool)[None]
        score_min=np.where(diag,np.inf,others).min(axis=2)
        score_max=np.where(diag,-np.inf,others).max(axis=2)
        score_mean=(self.scores.sum(axis=1)[:,None]-self.scores)/(p-1)
        score_summary=np.clip(np.stack((self.scores,score_min,score_mean,score_max),axis=-1)/50,0,1)
        common=np.concatenate((handmask,np.broadcast_to(self.seen[:,None,:],(b,p,104)),
            np.broadcast_to((self.rows/104).reshape(b,1,24),(b,p,24)),np.broadcast_to(summary[:,None,:],(b,p,12)),
            score_summary,np.broadcast_to(np.array([1,p/10,h/10,(10-h)/10]),(b,p,4))),axis=-1)
        cards=self.hands
        eligible=tails[:,None,None,:]<cards[:,:,:,None]
        target=np.where(eligible,tails[:,None,None,:],-1).argmax(axis=-1)
        low=~eligible.any(axis=-1)
        cheapest=(costs*100+self.lengths*10+np.arange(4)).argmin(axis=1)
        target=np.where(low,cheapest[:,None,None],target)
        row_tail=tails[np.arange(b)[:,None,None],target]
        row_length=self.lengths[np.arange(b)[:,None,None],target]
        row_cost=costs[np.arange(b)[:,None,None],target]
        captures=low|(row_length>=5)
        unknown=~(handmask|self.seen[:,None,:])
        prefix=np.concatenate((np.zeros((b,p,1),dtype=np.int64),np.cumsum(unknown,axis=-1)),axis=-1)
        below=np.take_along_axis(prefix,cards-1,axis=-1)
        at_tail=np.take_along_axis(prefix,row_tail,axis=-1)
        interval=np.where(low,0,below-at_tail)/np.maximum(1,unknown.sum(axis=-1))[:,:,None]
        trapped=(cards<tails.min(axis=1)[:,None,None]).sum(axis=-1)[:,:,None]-(cards<tails.min(axis=1)[:,None,None])
        left=cards-np.concatenate((np.zeros((b,p,1),dtype=np.int64),cards[:,:,:-1]),axis=-1)
        right=np.concatenate((cards[:,:,1:],np.full((b,p,1),105)),axis=-1)-cards
        rank=np.broadcast_to(np.arange(h)/max(1,h-1),(b,p,h))
        single=lambda x:x[:,:,:,None]
        action=np.concatenate((single(cards/104),single(cards/104),single(POINTS[cards]/7),single(rank),
            single(left/104),single(right/104),np.eye(4)[target],single(np.where(low,0,np.maximum(0,cards-row_tail-1))/104),
            single(np.where(low,0,np.maximum(0,5-row_length))/5),single(np.where(captures,row_cost,0)/25),
            single(low),single(captures),single(interval),single(interval*(p-1)/9),single(trapped/9)),axis=-1)
        features=np.concatenate((np.broadcast_to(common[:,:,None,:],(b,p,h,252)),np.clip(action,0,1)),axis=-1).astype(np.float32)
        assert features.shape==(b,p,h,270) and np.isfinite(features).all()
        return features

    def step(self,actions):
        b,p,h=self.hands.shape
        actions=np.asarray(actions,dtype=np.int64)
        if actions.shape!=(b,p) or not np.all((self.hands==actions[:,:,None]).sum(axis=-1)==1):raise ValueError('Illegal actions')
        before=self.scores.copy();order=actions.argsort(axis=1);batch=np.arange(b)
        for position in range(p):
            seat=order[:,position];card=actions[batch,seat]
            tails=np.take_along_axis(self.rows,(self.lengths-1)[:,:,None],axis=-1)[:,:,0]
            eligible=tails<card[:,None];low=~eligible.any(axis=1)
            costs=POINTS[self.rows].sum(axis=-1)
            row=np.where(eligible,tails,-1).argmax(axis=1)
            cheapest=(costs*100+self.lengths*10+np.arange(4)).argmin(axis=1)
            row=np.where(low,cheapest,row)
            capture=low|(self.lengths[batch,row]==5)
            self.scores[batch,seat]+=np.where(capture,costs[batch,row],0)
            captured=batch[capture];cr=row[capture]
            self.rows[captured,cr,:]=0;self.lengths[captured,cr]=0
            self.rows[batch,row,self.lengths[batch,row]]=card
            self.lengths[batch,row]+=1
        self.hands=self.hands[self.hands!=actions[:,:,None]].reshape(b,p,h-1)
        self.seen[np.arange(b)[:,None],actions-1]=True
        return self.scores-before

    def outcome(self):
        minimum=self.scores.min(axis=1,keepdims=True)
        wins=self.scores==minimum;share=wins/wins.sum(axis=1,keepdims=True)
        lower=(self.scores[:,None,:]<self.scores[:,:,None]).sum(axis=-1)
        tied=(self.scores[:,None,:]==self.scores[:,:,None]).sum(axis=-1)
        rank=1+lower+(tied-1)/2
        return share.astype(np.float32),rank.astype(np.float32)

    def record(self,game,seat):
        order=[seat]+[i for i in range(self.players) if i!=seat]
        return dict(rows=[self.rows[game,i,:n].tolist() for i,n in enumerate(self.lengths[game])],
            hand=self.hands[game,seat].tolist(),seenCards=(np.flatnonzero(self.seen[game])+1).tolist(),
            deckSize=104,deckMode='classic',playerCount=self.players,scores=self.scores[game,order].tolist())
