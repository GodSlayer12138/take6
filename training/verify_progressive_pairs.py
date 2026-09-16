"""Check legal two-card plans, final-two-card equivalence and graph consistency."""
import copy
import json
from pathlib import Path
import numpy as np
import progressive_campaign as campaign
import progressive_planner as runtime
from small_player_env import SmallGame


def main():
    common=dict(engine='torch',device='cuda',opponents='actual-proxy',
        proxy_directory='artifacts/progressive-upgrades/opponent-proxies-v2',worlds=32,prior_weight=.003,bullhead_weight=.002)
    ordinary=runtime.Planner(common)
    planned=runtime.Planner(dict(common,pair_worlds=64,pair_topk=3))
    captured=runtime.Planner(dict(common,pair_worlds=64,pair_topk=3,cuda_graphs=True,graph_max_batch=2048))
    checks=0
    for p in (3,4):
        for repetition in range(3):
            game=SmallGame(1,p,90946000+p*100+repetition)
            for turn in range(10):
                if turn in (6,8):
                    obs=game.record(0,0);obs['seed']=90946500+p*100+repetition*10+turn
                    if turn==8:
                        a=ordinary.choose(obs)
                        pair=runtime.Planner(dict(common,pair_card_subset=obs['hand']))
                        b=pair.choose(obs)
                        np.testing.assert_array_equal(a['values'],b['values'])
                        assert a['card']==b['card'];checks+=2
                    else:
                        a=planned.choose(obs);b=captured.choose(obs)
                        assert a['card']==b['card'];np.testing.assert_allclose(a['utility'],b['utility'],atol=1e-7)
                        sequence=a['two_card_plan']['selected_sequence']
                        assert sequence[0]==a['card'] and len(set(sequence))==2 and set(sequence).issubset(obs['hand'])
                        private=copy.deepcopy(obs);private['opponentHands']=[[104]*4]*(p-1);private['opponentIds']=['random']*(p-1)
                        other=planned.choose(private);assert other['card']==a['card']
                        np.testing.assert_array_equal(other['utility'],a['utility']);checks+=5
                game.step(game.hands[:,:,0])
    result=dict(status='passed',checks=checks,scope='Final two cards agree with ordinary rollout; plans are legal and use public information; graph and original execution agree',
        source_sha256=campaign.sha(Path(runtime.__file__)))
    campaign.dump(campaign.OUT/'two-card-plan-verification.json',result);print(json.dumps(result))


if __name__=='__main__':main()
