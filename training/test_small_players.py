import copy
import unittest
import numpy as np
import torch
from small_player_env import SmallGame
from train_policy import encode_candidate
from run_local_tournament import ArenaTable
from iterate_small_players import load,BASE,rollout,STAGES,update_ppo,Critic,DEVICE

class SmallPlayerTests(unittest.TestCase):
    def test_features_and_rules(self):
        for players in (2,3,4):
            game=SmallGame(12,players,60834001+players)
            reference=[ArenaTable([[c-1 for c in row] for row in game.record(i,0)['rows']],
                (game.hands[i]-1).tolist()) for i in range(12)]
            rng=np.random.default_rng(12)
            for turn in range(10):
                features=game.features()
                for i in range(12):
                    for seat in range(players):
                        obs=game.record(i,seat)
                        expected=np.array([encode_candidate(obs,c) for c in obs['hand']],dtype=np.float32)
                        np.testing.assert_allclose(features[i,seat],expected,atol=1e-7,rtol=1e-6)
                h=game.hands.shape[-1];chosen=rng.integers(0,h,(12,players))
                actions=np.take_along_axis(game.hands,chosen[:,:,None],axis=-1)[:,:,0]
                penalties=game.step(actions)
                for i,ref in enumerate(reference):
                    expected=-ref.step((actions[i]-1).tolist())
                    np.testing.assert_array_equal(penalties[i],expected)
                    self.assertEqual([[c+1 for c in r] for r in ref.rows],game.record(i,0)['rows'])
            share,rank=game.outcome()
            np.testing.assert_allclose(share.sum(axis=1),1)
            self.assertTrue(np.all((rank>=1)&(rank<=players)))

    def test_hidden_hands_do_not_enter_own_features(self):
        a=SmallGame(8,4,19);b=copy.deepcopy(a)
        b.hands[:,1:]=b.hands[:,[3,1,2]]
        np.testing.assert_array_equal(a.features()[:,0],b.features()[:,0])

    def test_rollout_reproducible_and_optimizer_changes_policy(self):
        torch.set_num_threads(1);model=load(BASE);anchor=copy.deepcopy(model).eval()
        before={k:v.clone() for k,v in model.state_dict().items()}
        first,_=rollout(model,[anchor],4,8,101)
        second,_=rollout(model,[anchor],4,8,101)
        for key in first:torch.testing.assert_close(first[key],second[key],rtol=0,atol=0)
        critic=Critic().to(DEVICE)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5)
        critic_optimizer=torch.optim.AdamW(critic.parameters(),lr=3e-4)
        stats=update_ppo(model,anchor,critic,first,optimizer,critic_optimizer,STAGES['v3'])
        self.assertTrue(all(np.isfinite(v) for v in stats.values()))
        self.assertTrue(any(not torch.equal(v,before[k]) for k,v in model.state_dict().items()))
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in anchor.state_dict().items()))

if __name__=='__main__':unittest.main()
