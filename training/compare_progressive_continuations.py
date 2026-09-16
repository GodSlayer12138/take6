"""Fresh paired proxy development games for the inexpensive continuation networks."""
import json
from pathlib import Path
import time
import numpy as np
import torch
import train_progressive_ppo as t

OUT=t.e.ROOT/'artifacts/progressive-upgrades/continuation-comparison-001'
MODELS=dict(base='artifacts/models/ntw-adaptive5-v1.pt',
    imitation='artifacts/progressive-upgrades/continuation-v1/model.pt',
    ppo='artifacts/progressive-upgrades/ppo-proxy-v1/model.pt',
    qstudent='artifacts/progressive-upgrades/qstudent-v1/model.pt')


@torch.inference_mode()
def main():
    if (OUT/'results.json').exists():print('Already complete');return
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);t.e.DEVICE=torch.device('cuda')
    t.CONFIG['seed_namespace']='continuation-comparison-001'
    planner=t.Planner(dict(engine='torch',device='cuda',opponents='actual-proxy',proxy_directory=t.CONFIG['opponent_proxy']))
    paths={k:t.e.ROOT/v for k,v in MODELS.items()};models={k:t.e.load(path) for k,path in paths.items()}
    t.e.dump(OUT/'protocol.json',dict(games_per_model_per_player_count=2048,players=[3,4],deck_size=104,
        seed_namespace=t.CONFIG['seed_namespace'],source_sha256=t.e.sha(Path(__file__)),
        weights_sha256={k:t.e.sha(v) for k,v in paths.items()},role='Proxy development; no real-opponent certification'))
    scores={};start=time.perf_counter()
    for p in (3,4):
        scores[str(p)]={}
        for name,policy in models.items():
            torch.manual_seed(t.seed('actions-'+str(p)));parts=[]
            for block in range(4):
                table,groups=t.environment(planner,p,512,f'{p}-{block}')
                for turn in range(10):
                    common,action=table.features();h=10-turn
                    features=torch.cat((common[:,0,None,:].expand(-1,h,-1),action[:,0]),dim=-1)
                    indices=t.opponents(planner,table,common,action,groups);indices[:,0]=policy(features).argmin(-1)
                    table.step(table.hands.gather(2,indices[:,:,None]).squeeze(-1))
                parts.append(table.scores.cpu().numpy())
            scores[str(p)][name]=np.concatenate(parts).tolist()
            print(json.dumps(dict(players=p,model=name,seconds=time.perf_counter()-start)),flush=True)
    t.e.dump(OUT/'scores.json',scores);results={}
    for p,items in scores.items():
        shares={}
        for name,values in items.items():
            x=np.asarray(values);win=x==x.min(axis=1,keepdims=True);shares[name]=(win/win.sum(axis=1,keepdims=True))[:,0]
        results[p]=[];rng=np.random.default_rng(90932000+int(p));ix=rng.integers(0,2048,(10000,2048))
        for name,values in shares.items():
            differences=values-shares['base'];boot=differences[ix].mean(axis=1)*100
            results[p].append(dict(model=name,win_rate=float(values.mean()),delta_pp=float(differences.mean()*100),
                development_delta_ci95_pp=np.quantile(boot,[.025,.975]).tolist()))
    report=dict(results=results,games=16384,role='Development evidence on learned opponent proxies; not real frozen opponents')
    t.e.dump(OUT/'results.json',report);print(json.dumps(report))


if __name__=='__main__':main()
