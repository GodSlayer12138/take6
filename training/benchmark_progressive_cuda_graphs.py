"""Full archived decision checks and cold/warm timings for graph execution."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import progressive_campaign as campaign
import progressive_cuda_graphs as graphs
from small_player_env import SmallGame


def main(args):
    folder=campaign.OUT/args.archive;manifest=campaign.load(folder/'manifest.json')
    campaign.verify(manifest,folder)
    entry=next(e for e in manifest['entrants'] if e['id']==args.policy)
    name='graph_benchmark_archived_planner'
    spec=importlib.util.spec_from_file_location(name,entry['runtime']);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module)
    planner=module.Planner(campaign.load(entry['path'])['params'],'cpu')
    stats=(lambda:[]) if args.direct else graphs.install(planner.torch_env,max_batch=args.max_batch)
    records=[];counts={3:0,4:0}
    for line in (folder/'blocks.jsonl').read_text(encoding='utf-8').splitlines():
        block=json.loads(line);p=block['players']
        if counts[p]>=2:continue
        counts[p]+=1;records.extend(r for r in block['records'] if r['candidate']==args.policy)
    assert len(records)==14
    result=dict(policy=args.policy,archive=args.archive,direct=args.direct,source_sha256=campaign.sha(Path(graphs.__file__)),runs=[])
    for repeat in range(2):
        started=time.perf_counter();mismatches=[];decisions=0
        for r in records:
            p=len(r['seats']);seat=r['seats'].index(r['candidate']);order=[(seat+i)%p for i in range(p)]
            game=SmallGame(1,p,r['deal_seed']);game.rows=game.rows[:,::-1].copy();history=[]
            for turn in range(10):
                obs=game.record(0,seat);obs.update(scores=game.scores[0,order].tolist(),history=list(history),
                    seed=int(r['deal_seed']+r['rotation']*1000003+turn*104729+seat*8191))
                chosen=planner.choose(obs);played=np.asarray(r['actions'][turn])+1;decisions+=1
                if chosen['card']!=played[seat]:
                    mismatches.append(dict(players=p,seed=r['deal_seed'],rotation=r['rotation'],turn=turn,
                        expected=int(played[seat]),actual=chosen['card']))
                history.append(dict(rows=obs['rows'],seenCards=obs['seenCards'],ownCard=int(played[seat]),opponentCards=played[order[1:]].tolist()))
                game.step(played[None])
            np.testing.assert_array_equal(game.scores[0],r['bullheads'])
        row=dict(repeat=repeat,decisions=decisions,mismatches=mismatches,seconds=time.perf_counter()-started,
            cache=stats(),peak_cuda_bytes=torch.cuda.max_memory_allocated())
        result['runs'].append(row);print(json.dumps(row),flush=True)
    result['status']='passed' if all(not r['mismatches'] for r in result['runs']) else 'failed'
    suffix='-direct' if args.direct else ''
    campaign.dump(campaign.OUT/'cuda-graph-validation'/f'{args.policy}{suffix}.json',result)
    campaign.dump(campaign.OUT/'cuda-graph-validation'/f'{args.policy}{suffix}-{result["source_sha256"][:12]}.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--archive',default='development-003')
    parser.add_argument('--policy',default='proxy-v2-512-lowprior');parser.add_argument('--max-batch',type=int,default=16384)
    parser.add_argument('--direct',action='store_true')
    main(parser.parse_args())
