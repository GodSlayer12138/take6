"""Non-statistical correctness checks for the progressive strategy campaign."""
import json
import numpy as np
from small_player_env import SmallGame
from progressive_planner import public_game,Planner
from progressive_belief import history_states,historical_worlds,posterior
import explore_small_strategies as e


def verify():
    checks=0
    for p in (3,4):
        game=SmallGame(1,p,90922000+p);history=[];past=[]
        for turn in range(8):
            past.append(e.take_games(game,np.array([0])))
            played=game.hands[:,:,turn%(10-turn)]
            obs=game.record(0,0)
            history.append(dict(rows=obs['rows'],seenCards=obs['seenCards'],ownCard=int(played[0,0]),
                opponentCards=played[0,1:].tolist()))
            game.step(played)
        obs=game.record(0,0);obs.update(history=history,seed=90922400+p)
        root=public_game(obs);scores,actions=history_states(root,history)
        for turn in range(8):
            reconstructed=historical_worlds(game,history[turn],scores[turn],actions[turn:])
            for name in ('hands','rows','lengths','scores','seen'):
                np.testing.assert_array_equal(getattr(reconstructed,name),getattr(past[turn],name))
                checks+=1
        # public_worlds must be independent of actual opponents' remaining hands.
        worlds=e.public_worlds(root,0,64,1703)
        altered=e.take_games(root,np.array([0]));altered.hands[:,1:]=103
        repeated=e.public_worlds(altered,0,64,1703)
        np.testing.assert_array_equal(worlds.hands,repeated.hands);checks+=1
        for profile in ('neural','actual-proxy'):
            config=dict(engine='torch',device='cpu',opponents=profile,worlds=16,belief_history=3,
                belief_strength=.5,belief_min_ess=.2)
            planner=Planner(config)
            owners=np.random.default_rng(703).integers(0,4 if profile=='neural' else 8,(64,p));owners[:,0]=-1
            samples,types,diagnostics=posterior(planner,root,worlds,owners,obs,16,600)
            assert samples.batch==16 and types.shape==(16,p)
            assert diagnostics['ess']>=64*.2-1e-6
            assert np.isfinite(samples.features()).all()
            # Extra hidden state and opponent identities are deliberately ignored.
            tainted=dict(obs,opponentHands=[[104,103]]*(p-1),opponentIds=['oracle']*(p-1))
            again,again_types,again_diag=posterior(planner,root,worlds,owners,tainted,16,600)
            np.testing.assert_array_equal(again.hands,samples.hands)
            np.testing.assert_array_equal(again_types,types);assert again_diag==diagnostics
            checks+=6
            config['belief_oversample']=4
            integrated=Planner(config)
            result=integrated.choose(obs);repeat=integrated.choose(tainted)
            assert result['card'] in obs['hand'] and result['worlds']==16
            assert result['card']==repeat['card'] and result['values']==repeat['values']
            assert result['belief']['history_turns']==3
            checks+=3
        common=dict(engine='torch',device='cpu',opponents='actual-proxy',worlds=16,belief_oversample=0,
            prior_from_continuation=True,prior_weight=.01)
        overrides={'3':dict(worlds=24,belief_oversample=2),'4':dict(worlds=16,belief_oversample=0)}
        routed=Planner(dict(common,player_overrides=overrides,
            continuation_by_players={str(p):'artifacts/progressive-upgrades/continuation-v1/model.pt'}))
        direct=Planner(dict(common,**overrides[str(p)],continuation_checkpoint='artifacts/progressive-upgrades/continuation-v1/model.pt'))
        left=routed.choose(obs);right=direct.choose(obs)
        assert left['card']==right['card'] and left['values']==right['values'] and left['utility']==right['utility']
        assert routed.config['worlds']==16 and routed.config['belief_oversample']==0
        checks+=2
        adaptive=Planner(dict(common,refine_worlds=32,refine_margin=10))
        full=Planner(dict(common,worlds=32))
        left=adaptive.choose(obs);right=full.choose(obs)
        assert left['refinement']['initial_worlds']==16 and left['worlds']==32
        assert left['card']==right['card'] and left['values']==right['values'] and left['utility']==right['utility']
        assert adaptive.config['worlds']==16
        checks+=3
    result=dict(status='passed',checks=checks,scope='3/4 players, public history reconstruction and hidden-information invariance')
    e.dump(e.ROOT/'artifacts/progressive-upgrades/correctness-checks.json',result)
    print(json.dumps(result))


if __name__=='__main__':verify()
