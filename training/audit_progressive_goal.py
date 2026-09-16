"""Read-only completion audit independent of the campaign's publication routine.

Only completed attempts listed in state are scored. Pending traces are not read.
The only write is a separate audit report, never a policy, result or goal state.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
from report_local_tournament import replay

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/progressive-upgrades'


def load(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def seed(namespace,p,block):
    return int.from_bytes(hashlib.sha256(f'ntw-progressive-20260909:{namespace}:{p}:{block}'.encode()).digest()[:4],'little')


def identity(entry):
    return (entry['sha256'],entry.get('runtime_sha256'),entry.get('dependency_sha256',{}))


def verify_entry(entry):
    if 'sha256' in entry:assert sha(entry['path'])==entry['sha256']
    if 'runtime_sha256' in entry:assert sha(entry['runtime'])==entry['runtime_sha256']
    for path,digest in entry.get('dependency_sha256',{}).items():assert sha(path)==digest,path


def main(args):
    state=load(OUT/'state.json');protocol=load(OUT/'protocol.json');attempts=state['certificate_attempts']
    assert protocol['certificate_deals_per_player_count']==2048 and protocol['bootstrap_replicates']==200000
    assert protocol['alpha_total']==.05 and protocol['minimum_observed_delta_pp']==.5
    # Check the historical development sources used by this campaign as well.
    historical=set()
    for name in ('small-player-iterations','small-player-exploration'):
        for line in (ROOT/'artifacts'/name/'heldout-blocks.jsonl').read_text(encoding='utf-8').splitlines():
            block=json.loads(line)
            if block['mode'] in ('classic3','classic4'):historical.add(block['deal_seed'])
    for line in (ROOT/'artifacts/local-tournament/classic4/blocks.jsonl').read_text(encoding='utf-8').splitlines():
        historical.update(r['deal_seed'] for r in json.loads(line))
    # Reserve all campaign deal seeds, including pending runs, without reading outcomes.
    reserved={}
    for path in OUT.glob('*/manifest.json'):
        manifest=load(path)
        for p in (3,4):
            for block in range(manifest['deals']):
                value=seed(manifest['namespace'],p,block)
                assert value not in historical,(path.parent.name,'historical development seed')
                assert value not in reserved,(path.parent.name,reserved.get(value))
                reserved[value]=path.parent.name
    predecessor='v5';previous_policy=None;accepted=[];verified_games=0;reports=[]
    for number,attempt in enumerate(attempts,1):
        assert attempt['attempt']==number
        folder=OUT/attempt['run'];manifest=load(folder/'manifest.json');result=load(folder/'results.json')
        assert manifest['purpose']=='certificate' and manifest['deals']==2048
        assert manifest['certificate_attempt']==number and manifest['predecessor']==predecessor
        assert manifest['protocol_sha256']==sha(OUT/'protocol.json')
        alpha=.05/(number*(number+1));tail=alpha/2
        assert manifest['alpha_attempt']==alpha and manifest['alpha_per_mode']==tail and attempt['alpha']==alpha
        candidate=attempt['candidate'];assert manifest['versions']==['reference',candidate]
        entries={e['id']:e for e in manifest['entrants']}
        for entry in entries.values():verify_entry(entry)
        for name,digest in manifest['source_sha256'].items():assert sha(folder/'source-snapshot'/name)==digest
        for name,digest in manifest.get('common_dependency_sha256',{}).items():assert sha(name)==digest
        if previous_policy is None:
            assert entries['reference']['sha256']==sha(ROOT/'artifacts/small-player-exploration/v5/model.json')
        else:assert identity(entries['reference'])==identity(previous_policy)
        development=OUT/manifest['selection_run'];dev=load(development/'manifest.json')
        assert dev['purpose']=='development' and (development/'results.json').exists()
        assert identity(entries[candidate])==identity(next(e for e in dev['entrants'] if e['id']==candidate))
        assert identity(entries['reference'])==identity(next(e for e in dev['entrants'] if e['id']=='reference'))
        assert result['audit']=='passed' and result['trace_sha256']==sha(folder/'blocks.jsonl')
        data={3:{},4:{}}
        expected={}
        for p in (3,4):
            rng=np.random.default_rng(seed(manifest['namespace'],p,-1))
            for block in range(2048):expected[(p,block)]=rng.choice(protocol['opponents'],p-1,replace=False).tolist()
        count=0
        for line in (folder/'blocks.jsonl').read_text(encoding='utf-8').splitlines():
            block=json.loads(line);p=block['players'];bid=block['block'];opponents=expected[(p,bid)]
            assert bid not in data[p] and block['opponents']==opponents
            assert block['deal_seed']==seed(manifest['namespace'],p,bid) and len(block['records'])==2*p
            values=[]
            for version in ('reference',candidate):
                records=[r for r in block['records'] if r['candidate']==version];assert len(records)==p
                wins=[]
                for rotation,r in enumerate(records):
                    group=[version]+opponents;assert r['rotation']==rotation and r['seats']==group[rotation:]+group[:rotation]
                    assert r['deal_seed']==block['deal_seed'];replay(r,p,104);count+=1
                    focal=r['seats'].index(version);cost=np.asarray(r['bullheads']);winners=cost==cost.min()
                    wins.append(float(winners[focal]/winners.sum()))
                values.append(float(np.mean(wins)))
            data[p][bid]=values
        assert count==28672==result['games'];verified_games+=count
        modes={};passes=[]
        for p in (3,4):
            assert len(data[p])==2048
            x=np.asarray([data[p][i] for i in range(2048)]);delta=x[:,1]-x[:,0]
            rng=np.random.default_rng(seed(manifest['namespace']+'-bootstrap',p,-1));boot=np.empty(200000)
            for start in range(0,len(boot),100):
                boot[start:start+100]=delta[rng.integers(0,2048,(100,2048))].mean(axis=1)*100
            gain=float(delta.mean()*100);lower=float(np.quantile(boot,tail));interval=np.quantile(boot,[.025,.975])
            published={r['version']:r for r in result['results'][str(p)]}
            for index,version in enumerate(('reference',candidate)):
                assert published[version]['games']==2048*p
                np.testing.assert_allclose(published[version]['win_rate'],x[:,index].mean(),atol=1e-12,rtol=0)
            np.testing.assert_allclose(published[candidate]['delta_pp'],gain,atol=1e-10,rtol=0)
            np.testing.assert_allclose(published[candidate]['adjusted_one_sided_lower_pp'],lower,atol=1e-9,rtol=0)
            np.testing.assert_allclose(published[candidate]['delta_ci95_pp'],interval,atol=1e-9,rtol=0)
            passes.append(lower>0 and gain>=.5)
            modes[str(p)]=dict(delta_pp=gain,adjusted_lower_pp=lower,independent_deals=2048,games_per_version=2048*p)
        passed=all(passes);assert attempt['passed']==passed and attempt['results']==result['results']
        if passed:
            accepted.append(attempt)
            recorded=state['accepted_upgrades'][len(accepted)-1]
            assert recorded['upgrade']==len(accepted) and recorded['predecessor']==predecessor
            for key,value in attempt.items():assert recorded[key]==value
            assert identity(recorded['policy'])==identity(entries[candidate])
            predecessor=candidate;previous_policy=entries[candidate]
        reports.append(dict(attempt=number,candidate=candidate,passed=passed,modes=modes,games=count))
    assert len(accepted)==len(state['accepted_upgrades']) and state['active_baseline']==predecessor
    complete=len(accepted)>=3
    assert state['goal_complete']==complete
    pending=state.get('pending_certificate')
    if pending:assert pending['attempt']==len(attempts)+1
    report=dict(status='passed',observed_at=datetime.now().astimezone().isoformat(),
        goal_complete=complete,accepted_upgrades=len(accepted),completed_attempts=len(attempts),
        verified_games=verified_games,attempts=reports,all_campaign_seed_reservations_unique=True,
        campaign_seeds_disjoint_from_historical_development=True,
        pending_outcomes_read=False,protocol_sha256=sha(OUT/'protocol.json'),auditor_sha256=sha(Path(__file__)))
    (OUT/'independent-completion-audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)
    if args.require_complete:assert complete and not pending,'Three consecutive accepted upgrades are not yet established'


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--require-complete',action='store_true');main(parser.parse_args())
