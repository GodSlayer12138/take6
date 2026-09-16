"""Paired development comparisons of continuation networks against learned proxies."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import train_progressive_ppo as t


@torch.inference_mode()
def main(args):
    root=t.e.ROOT/'artifacts/progressive-upgrades';out=root/args.output
    if (out/'results.json').exists():print('Already complete');return
    assert args.games>0 and args.games%512==0
    specifications=json.loads(Path(args.models).read_text(encoding='utf-8'))
    assert 'base' in specifications
    torch.set_num_threads(1);t.e.DEVICE=torch.device('cuda');t.CONFIG['seed_namespace']=args.output
    planner=t.Planner(dict(engine='torch',device='cuda',opponents='actual-proxy',proxy_directory=t.CONFIG['opponent_proxy']))
    paths={path for spec in specifications.values() for path in (spec.values() if isinstance(spec,dict) else [spec])}
    models={path:t.e.load(t.e.ROOT/path) for path in paths}
    protocol=dict(games_per_model_per_player_count=args.games,players=[3,4],deck_size=104,
        seed_namespace=args.output,source_sha256=t.e.sha(Path(__file__)),models=specifications,
        weights_sha256={path:t.e.sha(t.e.ROOT/path) for path in paths},
        role='Development on learned opponent proxies; does not count as formal game-strength certification')
    if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==protocol
    t.e.dump(out/'protocol.json',protocol);scores={};started=time.perf_counter()
    for p in (3,4):
        scores[str(p)]={}
        for name,spec in specifications.items():
            policy=models[spec[str(p)] if isinstance(spec,dict) else spec]
            torch.manual_seed(t.seed('actions-'+str(p)));parts=[]
            for block in range(args.games//512):
                table,groups=t.environment(planner,p,512,f'{p}-{block}')
                for turn in range(10):
                    common,action=table.features();h=10-turn
                    features=torch.cat((common[:,0,None,:].expand(-1,h,-1),action[:,0]),dim=-1)
                    indices=t.opponents(planner,table,common,action,groups);indices[:,0]=policy(features).argmin(-1)
                    table.step(table.hands.gather(2,indices[:,:,None]).squeeze(-1))
                parts.append(table.scores.cpu().numpy())
            scores[str(p)][name]=np.concatenate(parts).tolist()
            print(json.dumps(dict(players=p,model=name,seconds=time.perf_counter()-started)),flush=True)
    t.e.dump(out/'scores.json',scores);results={}
    for p,items in scores.items():
        shares={}
        for name,values in items.items():
            x=np.asarray(values);win=x==x.min(axis=1,keepdims=True);shares[name]=(win/win.sum(axis=1,keepdims=True))[:,0]
        results[p]=[];rng=np.random.default_rng(t.seed('bootstrap-'+p))
        differences=np.stack([values-shares['base'] for values in shares.values()],axis=-1)
        boot=np.empty((10000,len(shares)))
        for start in range(0,10000,100):
            boot[start:start+100]=differences[rng.integers(0,args.games,(100,args.games))].mean(axis=1)*100
        for index,(name,values) in enumerate(shares.items()):
            results[p].append(dict(model=name,win_rate=float(values.mean()),delta_pp=float(differences[:,index].mean()*100),
                development_delta_ci95_pp=np.quantile(boot[:,index],[.025,.975]).tolist()))
    report=dict(results=results,games=2*args.games*len(specifications),role=protocol['role'])
    t.e.dump(out/'results.json',report);print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--models',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--games',type=int,default=4096);main(parser.parse_args())
