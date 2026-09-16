"""Train 3-player and 4-player PPO specialists sequentially in one GPU process."""
from argparse import Namespace
from pathlib import Path
import train_progressive_ppo_specialist as trainer


if __name__=='__main__':
    for players in (3,4):
        trainer.CONFIG.update(players=[players],updates=256,evaluate_every=32,
            initial_policy='artifacts/progressive-upgrades/ppo-proxy-v1/model.pt',
            seed_namespace=f'progressive-ppo-specialist-{players}-v1',
            selection='Largest deterministic proxy development win share for this player count; separate 3/4-player policies',
            driver_source_sha256=trainer.e.sha(Path(__file__)))
        trainer.main(Namespace(output=f'ppo-specialist-{players}-v1'))
