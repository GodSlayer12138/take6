"""Audit saved parameters, development selection and budgets without new games."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
from explore_small_strategies import ROOT,OUT,BASE,dump,sha

def read(path):return json.loads(path.read_text(encoding='utf-8'))

def validate_layers(export,state):
    for i,layer in enumerate(export['layers']):
        for name in ('weight','bias'):
            np.testing.assert_array_equal(np.asarray(layer[name],dtype=np.float32),state[f'layers.{i}.{name}'].numpy())

def main():
    if Path(sys.prefix).name.lower()!='ntw-ai':raise RuntimeError('Use ntw-ai')
    torch.set_num_threads(1)
    plan=read(OUT/'experiment-plan.json');base=torch.load(BASE,map_location='cpu',weights_only=True)['state_dict']
    assert sha(BASE)==plan['base_sha256']
    records={}
    for version in ('v5','v6','v7'):
        folder=OUT/version;protocol=read(folder/'protocol.json');summary=read(folder/'summary.json');model=read(folder/'model.json')
        assert not protocol['smoke'] and protocol['plan_sha256']==sha(OUT/'experiment-plan.json')
        for path,value in protocol['sources'].items():assert sha(ROOT/path)==value,path
        if version in ('v5','v6'):validate_layers(model['base'],base)
        if version=='v5':
            for p in (2,3,4):
                rows=[r for r in summary['development'] if r['players']==p]
                assert [r['blend'] for r in rows]==plan['symmetry_blends']
                assert model['settings'][str(p)]==max(rows,key=lambda r:r['win_rate'])['blend']
            records[version]=dict(original_weights_unchanged=True,selection='development only',new_training_games=0)
        if version=='v6':
            games=0
            for p in (2,3,4):
                rows=[json.loads(line) for line in (folder/f'p{p}-training.jsonl').read_text().splitlines()]
                assert [r['generation'] for r in rows]==list(range(1,17))
                assert all(r['games']==24*256 and len(r['fitness'])==24 for r in rows)
                assert all(np.isfinite(r['mean']).all() and np.isfinite(r['fitness']).all() for r in rows)
                selected=max([r for r in rows if 'development' in r],key=lambda r:r['development'])
                np.testing.assert_array_equal(model['settings'][str(p)],selected['mean'])
                assert np.any(np.asarray(model['settings'][str(p)])!=0)
                games+=sum(r['games'] for r in rows)
            assert games==294912==summary['training_games']
            records[version]=dict(strategy_evaluation_games=games,independent_training_deals=3*16*256,coefficients_trained=81,selection='development only')
        if version=='v7':
            states=0;changed={}
            for p in (2,3,4):
                rows=[json.loads(line) for line in (folder/f'p{p}-training.jsonl').read_text().splitlines()]
                assert [r['epoch'] for r in rows]==list(range(1,25))
                assert all(np.isfinite(r['loss']) for r in rows)
                best=max([r for r in rows if 'development' in r],key=lambda r:r['development'])
                checkpoint=torch.load(folder/f'p{p}-model.pt',map_location='cpu',weights_only=True)
                assert checkpoint['epoch']==best['epoch']
                validate_layers(model['models'][str(p)],checkpoint['state_dict'])
                changed[p]=any(not torch.equal(base[k],v) for k,v in checkpoint['state_dict'].items());assert changed[p]
                dataset=torch.load(folder/f'p{p}-counterfactual-data.pt',map_location='cpu',weights_only=True)
                f=dataset['features'];q=dataset['win_values'];mask=dataset['mask']
                assert f.shape==(512,10,270) and q.shape==mask.shape==(512,10)
                assert torch.isfinite(f).all() and torch.isfinite(q).all() and ((q>=0)&(q<=1)).all()
                assert set(mask.sum(dim=1).tolist())=={2,4,7,10}
                states+=len(f)
            assert summary['root_games']==384 and summary['completion_rollouts']==282624
            records[version]=dict(root_games=384,public_states=states,hypothetical_completion_rollouts=282624,parameters_changed=changed,selection='development only')
    prior=read(ROOT/'artifacts/small-player-iterations/experiment-plan.json')
    prior_seeds={prior['heldout_seeds'][m]+100000000+i*104729 for m,n in prior['heldout_deals'].items() for i in range(n)}
    new_seeds={plan['heldout_seeds'][m]+100000000+i*104729 for m,n in plan['heldout_deals'].items() for i in range(n)}
    assert len(new_seeds)==1024 and not (prior_seeds&new_seeds)
    for path,value in read(ROOT/'artifacts/small-player-iterations/evaluation-manifest.json')['source_sha256'].items():assert sha(ROOT/path)==value,path
    result=dict(status='passed',branches=records,new_heldout_seeds_disjoint_from_previous=True,previous_experiment_frozen_sources_unchanged=True)
    dump(OUT/'training-audit.json',result);print(json.dumps(result))

if __name__=='__main__':main()
