"""Explain behavioral changes on fresh diagnostic states, never select on heldout games."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
import explore_small_strategies as e

def main():
    if Path(sys.prefix).name.lower()!='ntw-ai':raise RuntimeError('Use ntw-ai')
    torch.set_num_threads(1)
    base=e.load(e.BASE);pool=e.opponents()
    exported={v:json.loads((e.OUT/v/'model.json').read_text(encoding='utf-8')) for v in ('v5','v6','v7')}
    rows=[]
    for p in (2,3,4):
        policies={v:e.Policy(base,m['kind'],m['settings'][str(p)]) for v,m in exported.items() if v!='v7'}
        policies['v7']=e.Policy(e.load(e.OUT/'v7'/f'p{p}-model.pt'))
        counts={v:0 for v in policies};permutation_changes=0;total=0
        for turn,root in e.roots(pool,base,p,256,60878001+p,[0,3,6,8]):
            features=root.features()[:,0];original=e.utilities(base,features).argmin(axis=-1)
            changed=e.utilities(base,e.permute_features(features,[3,2,1,0])).argmin(axis=-1)
            permutation_changes+=int(np.sum(original!=changed));total+=len(original)
            for v,policy in policies.items():counts[v]+=int(np.sum(policy.score(features).argmin(axis=-1)!=original))
        rows.append(dict(players=p,public_states=total,baseline_action_changed_by_row_reversal=permutation_changes/total,
            branch_action_disagreement_with_baseline={v:n/total for v,n in counts.items()}))
    result=dict(purpose='Behavioral diagnosis only; fresh seed, no test-set access and no model selection.',results=rows,
        caveat='Action differences and row sensitivity do not by themselves establish higher win rate.')
    e.dump(e.OUT/'behavior-diagnostics.json',result);print(json.dumps(result))

if __name__=='__main__':main()
