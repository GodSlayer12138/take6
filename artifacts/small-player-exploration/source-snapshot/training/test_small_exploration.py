import copy
import json
import subprocess
import unittest
import numpy as np
import torch
import explore_small_strategies as experiment
from small_player_env import SmallGame

class ExplorationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.base=experiment.load(experiment.BASE)
        cls.pool=[cls.base]

    def test_counterfactual_does_not_use_true_hidden_hands(self):
        root=SmallGame(2,4,681)
        for _ in range(6):root.step(root.hands[:,:,0])
        changed=copy.deepcopy(root);changed.hands[:,1:]=changed.hands[:,[3,1,2]]
        a=experiment.public_worlds(root,0,8,191);b=experiment.public_worlds(changed,0,8,191)
        np.testing.assert_array_equal(a.hands,b.hands)
        for i in range(a.batch):
            cards=a.hands[i].reshape(-1)
            self.assertEqual(len(np.unique(cards)),len(cards))
            self.assertFalse(a.seen[i,cards-1].any())
        q1,_=experiment.counterfactual_values(root,self.pool,self.base,8,123)
        q2,_=experiment.counterfactual_values(changed,self.pool,self.base,8,123)
        np.testing.assert_array_equal(q1,q2)

    def test_known_last_turn_outcome(self):
        root=SmallGame(1,2,119)
        root.hands=np.array([[[50],[60]]]);root.rows[:]=0
        root.rows[0,0,0]=1;root.rows[0,1,0]=5;root.rows[0,2,0]=20
        root.rows[0,3,:5]=[36,37,38,39,40];root.lengths=np.array([[1,1,1,5]])
        root.seen[:]=True;root.seen[:,[49,59]]=False;root.scores[:]=0
        q,n=experiment.counterfactual_values(root,self.pool,self.base,4,777)
        np.testing.assert_array_equal(q,[[0.]])
        self.assertEqual(n,4)

    def test_population_members_receive_identical_games(self):
        for p in (2,3,4):
            policy=experiment.Policy(self.base,'residual',np.zeros((3*12,27),dtype=np.float32))
            wins,scores=experiment.games(policy,self.pool,p,12,615,copies=3)
            np.testing.assert_array_equal(wins,np.tile(wins[:1],(3,1)))
            np.testing.assert_array_equal(scores,np.tile(scores[:1],(3,1)))

    def test_symmetry_average_is_invariant(self):
        f=SmallGame(2,4,909).features()[:,0]
        p=[2,0,3,1];changed=experiment.permute_features(f,p)
        inverse=np.argsort(p);restored=experiment.permute_features(changed,inverse)
        np.testing.assert_array_equal(f,restored)
        policy=experiment.Policy(self.base,'symmetry',1.)
        np.testing.assert_allclose(policy.score(f),policy.score(changed),atol=3e-6,rtol=1e-6)

    def test_exported_javascript_matches_python(self):
        cases=[]
        for version in ('v5','v6','v7'):
            folder=experiment.OUT/version
            if not (folder/'model.json').exists():folder=experiment.OUT/('smoke-'+version)
            path=folder/'model.json';model=json.loads(path.read_text(encoding='utf-8'))
            for p in (2,3,4):
                if model['kind']=='specialist':policy=experiment.Policy(experiment.load(folder/f'p{p}-model.pt'))
                else:policy=experiment.Policy(self.base,model['kind'],model['settings'][str(p)])
                game=SmallGame(4,p,998+p)
                for turn in range(10):
                    if turn in (0,4,8):
                        scores=policy.score(game.features()[:,0])
                        for i in range(game.batch):
                            cases.append(dict(model=str(path),observation=game.record(i,0),expected_card=int(game.hands[i,0,scores[i].argmin()]),expected_scores=scores[i].tolist()))
                    game.step(game.hands[:,:,0])
        script="""import { readFileSync } from 'node:fs';
import { evaluateSmallCards } from './scripts/small-strategy-runtime.mjs';
const cases=JSON.parse(readFileSync(0,'utf8'));const cache=new Map();
const results=cases.map(c=>{if(!cache.has(c.model))cache.set(c.model,JSON.parse(readFileSync(c.model,'utf8')));
 const rows=evaluateSmallCards(c.observation,cache.get(c.model));
 return {card:rows[0].card,scores:c.observation.hand.map(card=>rows.find(r=>r.card===card).utility)};});
process.stdout.write(JSON.stringify(results));"""
        proc=subprocess.run([r'D:\Programs\nodejs\node.exe','--input-type=module','-e',script],cwd=experiment.ROOT,input=json.dumps(cases),capture_output=True,text=True,check=True)
        results=json.loads(proc.stdout)
        self.assertEqual(len(cases),108)
        for case,result in zip(cases,results):
            self.assertEqual(case['expected_card'],result['card'])
            np.testing.assert_allclose(case['expected_scores'],result['scores'],atol=1e-5,rtol=1e-5)

if __name__=='__main__':unittest.main()
