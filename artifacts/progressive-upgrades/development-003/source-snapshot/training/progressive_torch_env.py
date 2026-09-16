"""Torch simulation with split public/candidate features; equivalent to SmallGame."""
import torch
import numpy as np
from small_player_env import POINTS

class TorchGame:
    def __init__(self,source,device):
        self.device=torch.device(device);self.batch=source.batch;self.players=source.players
        for name in ('hands','rows','lengths','scores','seen'):
            setattr(self,name,torch.as_tensor(getattr(source,name),device=self.device).clone())
        self.points=torch.tensor(POINTS,device=self.device)
        self.batch_ids=torch.arange(self.batch,device=self.device)
        self.row_ids=torch.arange(4,device=self.device)

    def repeat(self,count):
        result=object.__new__(TorchGame);result.device=self.device;result.batch=self.batch*count;result.players=self.players
        for name in ('hands','rows','lengths','scores','seen'):
            value=getattr(self,name);setattr(result,name,value.repeat((count,)+(1,)*(value.ndim-1)))
        result.points=self.points;result.row_ids=self.row_ids;result.batch_ids=torch.arange(result.batch,device=self.device)
        return result

    def features(self):
        b,p,h=self.hands.shape;cards=self.hands
        handmask=torch.zeros((b,p,104),dtype=torch.bool,device=self.device).scatter_(2,cards-1,True)
        tails=self.rows.gather(2,(self.lengths-1).unsqueeze(-1)).squeeze(-1)
        costs=self.points[self.rows].sum(dim=2)
        summary=torch.stack((self.lengths/5,(costs/25).clamp(0,1),tails/104),dim=-1).reshape(b,12)
        diag=torch.eye(p,dtype=torch.bool,device=self.device).unsqueeze(0)
        others=self.scores[:,None,:].expand(b,p,p)
        minimum=others.masked_fill(diag,float('inf')).min(dim=2).values
        maximum=others.masked_fill(diag,-float('inf')).max(dim=2).values
        average=(self.scores.sum(dim=1,keepdim=True)-self.scores)/(p-1)
        scores=(torch.stack((self.scores,minimum,average,maximum),dim=-1)/50).clamp(0,1)
        globals_=torch.tensor([1,p/10,h/10,(10-h)/10],dtype=torch.float32,device=self.device)
        common=torch.cat((handmask.float(),self.seen[:,None,:].expand(b,p,104).float(),
            (self.rows.float()/104).reshape(b,1,24).expand(b,p,24),summary[:,None,:].expand(b,p,12),
            scores,globals_[None,None,:].expand(b,p,4)),dim=-1)
        eligible=tails[:,None,None,:]<cards[:,:,:,None]
        target=torch.where(eligible,tails[:,None,None,:],-1).argmax(dim=-1)
        low=~eligible.any(dim=-1)
        cheapest=(costs*100+self.lengths*10+self.row_ids).argmin(dim=1)
        target=torch.where(low,cheapest[:,None,None],target)
        batch=self.batch_ids[:,None,None]
        row_tail=tails[batch,target];row_length=self.lengths[batch,target];row_cost=costs[batch,target]
        captures=low|(row_length>=5)
        unknown=~(handmask|self.seen[:,None,:])
        prefix=torch.cat((torch.zeros((b,p,1),dtype=torch.int64,device=self.device),unknown.cumsum(dim=-1)),dim=-1)
        below=prefix.gather(2,cards-1);at_tail=prefix.gather(2,row_tail)
        interval=torch.where(low,0,below-at_tail)/unknown.sum(dim=-1).clamp_min(1)[:,:,None]
        small=cards<tails.min(dim=1).values[:,None,None]
        trapped=small.sum(dim=-1,keepdim=True)-small.long()
        left=cards-torch.cat((torch.zeros((b,p,1),dtype=torch.int64,device=self.device),cards[:,:,:-1]),dim=-1)
        right=torch.cat((cards[:,:,1:],torch.full((b,p,1),105,dtype=torch.int64,device=self.device)),dim=-1)-cards
        rank=(torch.arange(h,device=self.device)/max(1,h-1))[None,None,:].expand(b,p,h)
        onehot=torch.nn.functional.one_hot(target,4).float()
        action=torch.cat((torch.stack((cards/104,cards/104,self.points[cards]/7,rank,left/104,right/104),dim=-1),
            onehot,torch.stack((torch.where(low,0,(cards-row_tail-1).clamp_min(0))/104,
            torch.where(low,0,(5-row_length).clamp_min(0))/5,torch.where(captures,row_cost,0)/25,
            low.float(),captures.float(),interval,interval*(p-1)/9,trapped/9),dim=-1)),dim=-1).clamp(0,1)
        return common,action

    def step(self,cards):
        b,p,h=self.hands.shape;before=self.scores.clone();order=cards.argsort(dim=1);batch=self.batch_ids
        for position in range(p):
            seat=order[:,position];card=cards[batch,seat]
            tails=self.rows.gather(2,(self.lengths-1).unsqueeze(-1)).squeeze(-1)
            eligible=tails<card[:,None];low=~eligible.any(dim=1);costs=self.points[self.rows].sum(dim=-1)
            row=torch.where(eligible,tails,-1).argmax(dim=1)
            cheapest=(costs*100+self.lengths*10+self.row_ids).argmin(dim=1)
            row=torch.where(low,cheapest,row);capture=low|(self.lengths[batch,row]==5)
            self.scores[batch,seat]+=torch.where(capture,costs[batch,row],0)
            selected_rows=self.rows[batch,row];selected_rows=torch.where(capture[:,None],0,selected_rows)
            length=torch.where(capture,0,self.lengths[batch,row])
            selected_rows.scatter_(1,length[:,None],card[:,None]);self.rows[batch,row]=selected_rows
            self.lengths[batch,row]=length+1
        self.hands=torch.where(self.hands==cards[:,:,None],105,self.hands).sort(dim=-1).values[:,:,:h-1]
        self.seen.scatter_(1,cards-1,True)
        return self.scores-before

    def outcome(self):
        winner=self.scores==self.scores.min(dim=1,keepdim=True).values
        shares=winner.float()/winner.sum(dim=1,keepdim=True)
        lower=(self.scores[:,None,:]<self.scores[:,:,None]).sum(dim=-1)
        tied=(self.scores[:,None,:]==self.scores[:,:,None]).sum(dim=-1)
        return shares,1+lower+(tied-1)/2

def split_utilities(model,common,action):
    first=model.layers[0]
    x=torch.nn.functional.linear(common,first.weight[:,:252],first.bias).unsqueeze(-2)
    x=torch.relu(x+torch.nn.functional.linear(action,first.weight[:,252:]))
    for layer in model.layers[1:-1]:x=torch.relu(layer(x))
    return model.layers[-1](x).squeeze(-1)*10

@torch.inference_mode()
def verify():
    from small_player_env import SmallGame
    import explore_small_strategies as e
    torch.set_num_threads(1);e.DEVICE=torch.device('cuda');model=e.load(e.BASE)
    turns=0;vectors=0
    for p in (3,4):
        source=SmallGame(32,p,90918001+p);target=TorchGame(source,'cuda');rng=np.random.default_rng(1881+p)
        for turn in range(10):
            f=source.features();c,a=target.features()
            encoded=torch.cat((c[:,:,None,:].expand(-1,-1,10-turn,-1),a),dim=-1).cpu().numpy()
            np.testing.assert_allclose(f,encoded,atol=1e-7,rtol=1e-6)
            expected=e.utilities(model,f);actual=split_utilities(model,c,a).cpu().numpy()
            np.testing.assert_allclose(expected,actual,atol=1e-5,rtol=1e-5)
            idx=rng.integers(0,10-turn,(32,p));played=np.take_along_axis(source.hands,idx[:,:,None],axis=-1)[:,:,0]
            costs=source.step(played);other=target.step(torch.tensor(played,device='cuda')).cpu().numpy()
            np.testing.assert_array_equal(costs,other)
            for name in ('hands','rows','lengths','scores','seen'):np.testing.assert_array_equal(getattr(source,name),getattr(target,name).cpu().numpy())
            turns+=32;vectors+=32*p*(10-turn)
        for left,right in zip(source.outcome(),target.outcome()):np.testing.assert_array_equal(left,right.cpu().numpy())
    print({'status':'passed','turns':turns,'candidate_vectors':vectors})

if __name__=='__main__':verify()
