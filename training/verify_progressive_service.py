"""Check a transport service against archived deterministic planner actions."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from multiprocessing.connection import Client
from pathlib import Path
import time
import numpy as np
from small_player_env import SmallGame

ROOT=Path(__file__).resolve().parents[1]


def main(args):
    state=json.loads(Path(args.state).read_text(encoding='utf-8'));manifest=Path(args.manifest)
    manifest_sha=hashlib.sha256(manifest.read_bytes()).hexdigest();assert manifest_sha in state['manifests']
    source=ROOT/'artifacts/progressive-upgrades'/args.archive/'blocks.jsonl';records=[];counts={3:0,4:0}
    trace_sha=hashlib.sha256(source.read_bytes()).hexdigest();record_policy=args.archive_policy or args.policy
    record_policies={3:args.archive_policy_3 or record_policy,4:args.archive_policy_4 or record_policy}
    for line in source.read_text(encoding='utf-8').splitlines():
        block=json.loads(line);p=block['players']
        if counts[p]>=2:continue
        counts[p]+=1;records.extend(r for r in block['records'] if r['candidate']==record_policies[p])
    assert len(records)==14
    inputs=Path(args.state).parent/f'verification-inputs-{args.policy}.json'
    inputs.write_text(json.dumps(records),encoding='utf-8')
    decisions=0;compute=[];start=time.perf_counter()
    def check(r):
        times=[]
        with Client(state['address'],family='AF_PIPE',authkey=bytes.fromhex(state['authkey'])) as connection:
            p=len(r['seats']);seat=r['seats'].index(r['candidate']);order=[(seat+i)%p for i in range(p)]
            game=SmallGame(1,p,r['deal_seed']);game.rows=game.rows[:,::-1].copy();history=[]
            for turn in range(10):
                obs=game.record(0,seat);obs.update(scores=game.scores[0,order].tolist(),history=list(history),
                    seed=int(r['deal_seed']+r['rotation']*1000003+turn*104729+seat*8191))
                connection.send(dict(type='choose',manifest=manifest_sha,policy=args.policy,observation=obs))
                response=connection.recv();assert 'error' not in response,response.get('error')
                result=response['result'];played=np.asarray(r['actions'][turn])+1
                assert result['card']==played[seat],(p,r['deal_seed'],r['rotation'],turn,result['card'],int(played[seat]))
                times.append(result['seconds'])
                history.append(dict(rows=obs['rows'],seenCards=obs['seenCards'],ownCard=int(played[seat]),opponentCards=played[order[1:]].tolist()))
                game.step(played[None])
            np.testing.assert_array_equal(game.scores[0],r['bullheads'])
        return times
    with ThreadPoolExecutor(max_workers=args.clients) as executor:
        for times in executor.map(check,records):
            compute.extend(times);decisions+=len(times)
            print(json.dumps(dict(games_verified=decisions//10,decisions=decisions)),flush=True)
    report=dict(status='passed',games=len(records),decisions=decisions,players=[3,4],policy=args.policy,
        clients=args.clients,streams=state.get('streams',1),
        check='Every selected card equals the archived original inference; reconstructed final scores also equal the archive',
        mean_compute_ms=float(np.mean(compute)*1000),wall_seconds=time.perf_counter()-start,
        source_trace_sha256=trace_sha,archive_policy_by_players=record_policies,
        selected_inputs_sha256=hashlib.sha256(inputs.read_bytes()).hexdigest(),service_source_sha256=state['source_sha256'],
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out=Path(args.state).parent/f'verification-{args.policy}.json';out.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--state',required=True);parser.add_argument('--manifest',required=True)
    parser.add_argument('--archive',default='development-002');parser.add_argument('--policy',default='gpu128-proxy')
    parser.add_argument('--archive-policy');parser.add_argument('--archive-policy-3');parser.add_argument('--archive-policy-4')
    parser.add_argument('--clients',type=int,default=1);main(parser.parse_args())
