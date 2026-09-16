"""Frozen CPU development games for three continuations and two proxy pools."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
import train_progressive_ppo as t
from small_player_env import SmallGame
from train_progressive_opponents_v3 import sha, dump

ROOT = t.e.ROOT
OUT = ROOT / 'artifacts/progressive-upgrades/four-player-proxy-diagnostic-001'
MODELS = {'original-ppo4': 'artifacts/progressive-upgrades/ppo-specialist-4-v1/model.pt',
          'shared-distill-v2': 'artifacts/progressive-upgrades/continuation-v2/model.pt',
          'fourp-distill-v3': 'artifacts/progressive-upgrades/continuation-v3-fourp/model.pt'}
POOLS = {'proxy-v2': 'artifacts/progressive-upgrades/opponent-proxies-v2',
         'proxy-v3': 'artifacts/progressive-upgrades/opponent-proxies-v3-devshift'}
KINDS = ('dirv-10000', 'alpha-2000', 'mcs', 'champion-search')
GAMES, BATCH, BOOTSTRAPS = 4096, 128, 20000


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def specification():
    paths = [Path(__file__), ROOT / 'training/train_progressive_ppo.py', ROOT / 'training/progressive_planner.py',
             ROOT / 'training/progressive_torch_env.py', ROOT / 'training/small_player_env.py',
             ROOT / 'training/explore_small_strategies.py', ROOT / 'training/train_policy.py',
             ROOT / 'training/train_progressive_opponents_v3.py']
    evaluation = ROOT / 'artifacts/small-player-exploration/evaluation-manifest.json'
    entrants = {e['id']: e for e in load(evaluation)['entrants']}
    weights = {str(ROOT / path): sha(ROOT / path) for path in MODELS.values()}
    for directory in POOLS.values():
        for kind in KINDS:
            path = ROOT / directory / (kind + '.pt')
            weights[str(path)] = sha(path)
    for name in ('ntw-adaptive-rl-v3b', 'ntw-adaptive-arena-v8-distill', 'ntw-champion'):
        path = Path(entrants[name]['path'])
        weights[str(path)] = sha(path)
    return dict(players=4, deck_size=104, turns=10, focal_seat=0, models=MODELS, proxy_pools=POOLS,
                games_per_model_per_pool=GAMES, batch_size=BATCH, total_games=GAMES * len(MODELS) * len(POOLS),
                bootstrap_replicates=BOOTSTRAPS, cpu_threads=2, device='cpu', seed_namespace=OUT.name,
                seed_scheme='SHA256 namespace and batch tag, 64 bits with high bit set; new proxy-development namespace.',
                pairing='Same initial deals, without-replacement opponent types and reset action RNG across all models and pools.',
                reference='original-ppo4', no_interim_selection=True,
                source_sha256={str(p.relative_to(ROOT)): sha(p) for p in paths}, weights_sha256=weights,
                evaluation_manifest_sha256=sha(evaluation), torch_version=torch.__version__, numpy_version=np.__version__,
                scope='Four-player development games against learned opponent proxies. Ordinary paired intervals, no formal certification or accepted upgrade.')


def freeze():
    value = specification()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        assert load(path) == value, 'Keep the existing diagnostic frozen'
    else:
        dump(path, value)
        for name in value['source_sha256']:
            target = OUT / 'source-snapshot' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
    for name, digest in value['source_sha256'].items():
        assert sha(OUT / 'source-snapshot' / name) == digest
    return value


def replay(record):
    game = SmallGame(record['games'], 4, record['deal_seed'])
    game.rows = game.rows[:, ::-1].copy()
    assert len(record['actions']) == 10
    for values in record['actions']:
        cards = np.asarray(values, dtype=np.int64)
        assert np.all((game.hands == cards[:, :, None]).sum(axis=-1) == 1)
        game.step(cards)
    np.testing.assert_array_equal(game.scores, np.asarray(record['scores']))


@torch.inference_mode()
def simulate(planner, model, pool, name, block, size=BATCH):
    tag = f'diagnostic-{block}'
    torch.manual_seed(t.seed(f'actions-{block}'))
    table, groups = t.environment(planner, 4, size, tag)
    actions = []
    for turn in range(10):
        common, action = table.features()
        features = torch.cat((common[:, 0, None, :].expand(-1, 10 - turn, -1), action[:, 0]), dim=-1)
        indices = t.opponents(planner, table, common, action, groups)
        indices[:, 0] = model(features).argmin(-1)
        cards = table.hands.gather(2, indices[:, :, None]).squeeze(-1)
        actions.append(cards.cpu().tolist())
        table.step(cards)
    record = dict(pool=pool, model=name, block=block, games=size, deal_seed=t.seed(tag),
                  action_seed=t.seed(f'actions-{block}'), actions=actions, scores=table.scores.cpu().tolist())
    replay(record)
    return record


def prepare_models():
    planners = {name: t.Planner(dict(engine='torch', device='cpu', opponents='actual-proxy', proxy_directory=path)) for name, path in POOLS.items()}
    torch.set_num_threads(2)
    models = {name: t.e.load(ROOT / path) for name, path in MODELS.items()}
    return planners, models


def records_so_far():
    path = OUT / 'blocks.jsonl'
    if not path.exists():
        return {}
    records = {}
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            record = json.loads(line)
            key = (record['pool'], record['model'], record['block'])
            assert key not in records
            assert key[0] in POOLS and key[1] in MODELS and 0 <= key[2] < GAMES // BATCH
            assert record['games'] == BATCH
            assert record['deal_seed'] == t.seed(f'diagnostic-{key[2]}')
            assert record['action_seed'] == t.seed(f'actions-{key[2]}')
            replay(record)
            records[key] = record
    return records


def summarize(records, protocol):
    assert len(records) == len(POOLS) * len(MODELS) * (GAMES // BATCH)
    assert specification() == protocol
    results = {}
    for pool in POOLS:
        scores = {name: np.concatenate([np.asarray(records[(pool, name, b)]['scores']) for b in range(GAMES // BATCH)]) for name in MODELS}
        shares = {}
        for name, values in scores.items():
            winners = values == values.min(axis=1, keepdims=True)
            shares[name] = (winners / winners.sum(axis=1, keepdims=True))[:, 0]
        difference = np.stack([v - shares['original-ppo4'] for v in shares.values()], axis=-1)
        rng = np.random.default_rng(t.seed('bootstrap-' + pool))
        bootstrap = np.empty((BOOTSTRAPS, len(MODELS)))
        for start in range(0, BOOTSTRAPS, 100):
            bootstrap[start:start + 100] = difference[rng.integers(0, GAMES, (100, GAMES))].mean(axis=1) * 100
        rows = []
        for index, (name, values) in enumerate(shares.items()):
            rows.append(dict(model=name, games=GAMES, win_rate=float(values.mean()),
                             delta_pp=float(difference[:, index].mean() * 100),
                             development_delta_ci95_pp=np.quantile(bootstrap[:, index], [.025, .975]).tolist(),
                             mean_bullheads=float(scores[name][:, 0].mean())))
        results[pool] = rows
    value = dict(status='complete', completed_at=datetime.now().astimezone().isoformat(), results=results,
                 games=protocol['total_games'], protocol_sha256=sha(OUT / 'protocol.json'),
                 trace_sha256=sha(OUT / 'blocks.jsonl'), audit='all games independently replayed in SmallGame',
                 scope=protocol['scope'])
    dump(OUT / 'results.json', value)
    print(json.dumps(value), flush=True)


def main(check):
    t.e.DEVICE = torch.device('cpu')
    t.CONFIG['seed_namespace'] = OUT.name
    assert not torch.cuda.is_available()
    if check:
        specification()
        planners, models = prepare_models()
        first = None
        for pool, planner in planners.items():
            for name, model in models.items():
                record = simulate(planner, model, pool, name, -1, 8)
                if first is None:
                    first = record
        repeated = simulate(planners[first['pool']], models[first['model']], first['pool'], first['model'], -1, 8)
        assert first == repeated
        print(json.dumps(dict(status='passed', replayed_games=56, six_configurations_checked=True,
                              exact_repeat=True, device='cpu', output_written=False)), flush=True)
        return
    protocol = freeze()
    if (OUT / 'results.json').exists():
        print('Diagnostic already complete.', flush=True)
        return
    records = records_so_far()
    planners, models = prepare_models()
    started = time.perf_counter()
    for pool, planner in planners.items():
        for name, model in models.items():
            for block in range(GAMES // BATCH):
                key = (pool, name, block)
                if key in records:
                    continue
                record = simulate(planner, model, pool, name, block)
                with (OUT / 'blocks.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(record) + '\n')
                records[key] = record
                if len(records) % 8 == 0:
                    print(json.dumps(dict(phase='four-player-proxy-diagnostic', batches=len(records),
                                          total_batches=192, seconds=time.perf_counter() - started)), flush=True)
    summarize(records_so_far(), protocol)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    main(parser.parse_args().check)
