"""Compare both frozen proxy revisions on their shared development validation set."""
import json
import torch
import explore_small_strategies as e
from train_progressive_opponents import KINDS


@torch.inference_mode()
def main():
    torch.set_num_threads(1);e.DEVICE=torch.device('cuda');root=e.ROOT/'artifacts/progressive-upgrades'
    data=torch.load(root/'opponent-proxies-v2/dataset.pt',map_location='cpu',weights_only=True)
    train=set(data['deal_seed'][~data['validation']].tolist());val=set(data['deal_seed'][data['validation']].tolist())
    assert train.isdisjoint(val)
    results={}
    for revision in ('v1','v2'):
        results[revision]={}
        for kind,name in enumerate(KINDS):
            model=e.load(root/f'opponent-proxies-{revision}'/(name+'.pt'))
            ids=torch.nonzero(data['validation']&(data['kind']==kind),as_tuple=True)[0];loss=correct=0
            for ix in ids.split(512):
                f=data['features'][ix].to(e.DEVICE);mask=data['mask'][ix].to(e.DEVICE);target=data['action'][ix].to(e.DEVICE)
                logits=(-model(f)*10).masked_fill(~mask,-1e9)
                loss+=float(torch.nn.functional.cross_entropy(logits,target,reduction='sum'))
                correct+=int((logits.argmax(-1)==target).sum())
            results[revision][name]=dict(states=len(ids),nll=loss/len(ids),accuracy=correct/len(ids))
    report=dict(results=results,split_audit='passed',training_deals=len(train),validation_deals=len(val),
        interpretation='Development validation, including checkpoint selection; not an independent game-strength test')
    e.dump(root/'opponent-proxy-comparison.json',report);print(json.dumps(report))


if __name__=='__main__':main()
