"""Check graph execution, repeated replay and shared-pool reuse against NumPy."""
import json
from pathlib import Path
import numpy as np
import torch
import progressive_torch_env as env
import progressive_cuda_graphs as graphs
import explore_small_strategies as e
from small_player_env import SmallGame


def main():
    torch.set_num_threads(1);e.DEVICE=torch.device('cuda');model=e.load(e.BASE)
    stats=graphs.install(env,max_batch=512);turns=0
    for repetition in range(2):
        for p in (3,4):
            source=SmallGame(32,p,90943000+p+repetition*100);target=env.TorchGame(source,'cuda')
            rng=np.random.default_rng(430+p+repetition)
            for turn in range(10):
                f=source.features();c,a=target.features()
                encoded=torch.cat((c[:,:,None,:].expand(-1,-1,10-turn,-1),a),-1).cpu().numpy()
                np.testing.assert_allclose(f,encoded,atol=1e-7,rtol=1e-6)
                expected=e.utilities(model,f)
                actual=env.split_utilities(model,c.reshape(-1,252),a.reshape(-1,10-turn,18)).reshape(32,p,10-turn).cpu().numpy()
                np.testing.assert_allclose(expected,actual,atol=1e-5,rtol=1e-5)
                ix=rng.integers(0,10-turn,(32,p));played=np.take_along_axis(source.hands,ix[:,:,None],axis=-1)[:,:,0]
                expected_cost=source.step(played);actual_cost=target.step(torch.tensor(played,device='cuda')).cpu().numpy()
                np.testing.assert_array_equal(expected_cost,actual_cost)
                for name in ('hands','rows','lengths','scores','seen'):
                    np.testing.assert_array_equal(getattr(source,name),getattr(target,name).cpu().numpy())
                turns+=32
    result=dict(status='passed',turns=turns,cache=stats(),source_sha256=e.sha(Path(graphs.__file__)))
    e.dump(e.ROOT/'artifacts/progressive-upgrades/cuda-graph-validation/environment.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
