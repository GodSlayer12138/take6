"""Refine the four-player continuation from completed strong-search teachers."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from datetime import datetime
import io
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
import explore_small_strategies as e
import train_progressive_continuation as teacher
from train_progressive_opponents_v3 import sha, dump, atomic_save

ROOT = e.ROOT
CAMPAIGN = ROOT / 'artifacts/progressive-upgrades'
OUT = CAMPAIGN / 'continuation-v3-fourp'
INITIAL = CAMPAIGN / 'ppo-specialist-4-v1/model.pt'
SOURCES = {'development-004': ['specialist-prior-1024'],
           'development-006': ['hybrid-wide-specialist'],
           'development-007': ['pair512-specialist2048']}
CONFIG = dict(seed=91237001, players=4, epochs=20, batch=256, lr=5e-5,
              weight_decay=.001, device='cpu', cpu_threads=2, row_permutation_augmentation=True,
              selection='Minimum whole-deal validation cross entropy, including epoch zero; ties retain earlier epoch.',
              information='Own current hand and public state, imitating the completed development search action.',
              role='Development training only; no certificate or incomplete development data. Not a game-strength result.')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def protocol():
    sources = {}
    for run, names in SOURCES.items():
        folder = CAMPAIGN / run
        manifest, result = load(folder / 'manifest.json'), load(folder / 'results.json')
        assert manifest['purpose'] == result['purpose'] == 'development' and result['audit'] == 'passed'
        assert all(name in manifest['versions'] for name in names)
        assert result['trace_sha256'] == sha(folder / 'blocks.jsonl')
        assert result['games'] == manifest['deals'] * 7 * len(manifest['versions'])
        sources[run] = dict(teachers=names, manifest_sha256=sha(folder / 'manifest.json'),
                            result_sha256=sha(folder / 'results.json'), trace_sha256=result['trace_sha256'])
    paths = [Path(__file__), ROOT / 'training/train_progressive_continuation.py',
             ROOT / 'training/train_progressive_opponents_v3.py', ROOT / 'training/explore_small_strategies.py',
             ROOT / 'training/small_player_env.py', ROOT / 'training/train_policy.py']
    return dict(**CONFIG, completed_sources=sources, initial_checkpoint=str(INITIAL), initial_sha256=sha(INITIAL),
                source_sha256={str(p.relative_to(ROOT)): sha(p) for p in paths},
                torch_version=torch.__version__, numpy_version=np.__version__)


def build():
    data, original = teacher.build_data(SOURCES)
    mask = data['players'] == 4
    data = {k: v[mask] for k, v in data.items()}
    assert len(data['action']) == (64 + 256 + 256) * 4 * 9
    assert bool((data['players'] == 4).all())
    assert bool(data['mask'][torch.arange(len(data['action'])), data['action']].all())
    assert not set(data['deal_seed'][data['validation']].tolist()) & set(data['deal_seed'][~data['validation']].tolist())
    provenance = dict(original_builder=original, selected_players=[4], states=len(data['action']),
                      unique_deals=len(data['deal_seed'].unique()), validation_states=int(data['validation'].sum()),
                      selected_teacher_games=2304, only_completed_development=True,
                      split='Original continuation-v1 hash, grouped by whole deal; all rotations stay together.')
    return data, provenance


def snapshot():
    value = protocol()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        assert load(path) == value, 'Existing training protocol must stay frozen'
    else:
        dump(path, value)
        for name in value['source_sha256']:
            target = OUT / 'source-snapshot' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
    for name, digest in value['source_sha256'].items():
        assert sha(OUT / 'source-snapshot' / name) == digest
    return value


def save_best(model, epoch, metrics):
    atomic_save(OUT / 'model.pt', dict(state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                                     epoch=epoch, validation=metrics, role='Four-player strong-search continuation distillation'))
    dump(OUT / 'model.json', e.serial(model))


def train(data):
    model = e.load(INITIAL)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    train_ids = torch.nonzero(~data['validation'], as_tuple=True)[0]
    val_ids = torch.nonzero(data['validation'], as_tuple=True)[0]
    last = OUT / 'last.pt'
    if last.exists():
        state = torch.load(last, map_location='cpu', weights_only=True)
        assert state['protocol_sha256'] == sha(OUT / 'protocol.json')
        initial, best, best_epoch = state['initial'], state['best'], state['best_epoch']
        best_state = state['best_state_dict']
        model.load_state_dict(best_state)
        save_best(model, best_epoch, best)
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['rng'])
        dump(OUT / f'epoch-{state["epoch"]:02d}.json', state['record'])
        start_epoch = state['epoch'] + 1
    else:
        initial = teacher.evaluate(model, data, val_ids)
        best, best_epoch, start_epoch = initial, 0, 1
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        dump(OUT / 'initial-validation.json', initial)
        save_best(model, 0, initial)
    start = time.perf_counter()
    for epoch in range(start_epoch, CONFIG['epochs'] + 1):
        losses = []
        for ids in train_ids[torch.randperm(len(train_ids))].split(CONFIG['batch']):
            permutation = e.PERMS[int(torch.randint(len(e.PERMS), (1,)))]
            features = torch.from_numpy(e.permute_features(data['features'][ids].numpy(), permutation))
            logits = (-model(features) * 10).masked_fill(~data['mask'][ids], -1e9)
            loss = torch.nn.functional.cross_entropy(logits, data['action'][ids])
            assert torch.isfinite(loss)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics = teacher.evaluate(model, data, val_ids)
        if metrics['nll'] < best['nll']:
            best, best_epoch = metrics, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save_best(model, epoch, best)
        row = dict(epoch=epoch, loss=float(np.mean(losses)), validation=metrics,
                   best_epoch=best_epoch, seconds=time.perf_counter() - start)
        atomic_save(last, dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(), rng=torch.get_rng_state(),
                              best_state_dict=best_state, initial=initial, best=best, best_epoch=best_epoch,
                              epoch=epoch, record=row, protocol_sha256=sha(OUT / 'protocol.json')))
        dump(OUT / f'epoch-{epoch:02d}.json', row)
        print(json.dumps(dict(phase='continuation-v3-fourp', **row)), flush=True)
    model.load_state_dict(torch.load(OUT / 'model.pt', map_location='cpu', weights_only=True)['state_dict'])
    assert teacher.evaluate(model, data, val_ids) == best
    history = [load(OUT / f'epoch-{epoch:02d}.json') for epoch in range(1, CONFIG['epochs'] + 1)]
    assert [row['epoch'] for row in history] == list(range(1, CONFIG['epochs'] + 1))
    (OUT / 'training.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in history), encoding='utf-8')
    result = dict(status='complete', completed_at=datetime.now().astimezone().isoformat(), initial=initial,
                  best=best, best_epoch=best_epoch, best_validation_nll=best['nll'], states=len(train_ids), validation_states=len(val_ids),
                  protocol_sha256=sha(OUT / 'protocol.json'), source_sha256=sha(__file__),
                  weights_sha256=sha(OUT / 'model.pt'), json_sha256=sha(OUT / 'model.json'),
                  interpretation='Development imitation only; no game-strength conclusion or accepted upgrade.')
    dump(OUT / 'complete.json', result)
    print(json.dumps(result), flush=True)


def check(data):
    model = e.load(INITIAL)
    ids = torch.nonzero(data['validation'], as_tuple=True)[0]
    metrics = teacher.evaluate(model, data, ids)
    assert np.isfinite(metrics['nll'])
    # The continuation uses Torch RNG for both batch order and row augmentation.
    buffer = io.BytesIO()
    torch.save({'rng': torch.get_rng_state()}, buffer)
    a, p = torch.randperm(512), torch.randint(24, (16,))
    buffer.seek(0)
    torch.set_rng_state(torch.load(buffer, weights_only=True)['rng'])
    assert torch.equal(a, torch.randperm(512)) and torch.equal(p, torch.randint(24, (16,)))
    print(json.dumps(dict(status='passed', four_player_states=len(data['action']),
                          whole_deal_split_disjoint=True, source_games_replayed=True,
                          rng_checkpoint_exact=True, initial_validation=metrics, output_written=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(CONFIG['cpu_threads'])
    torch.manual_seed(CONFIG['seed'])
    e.DEVICE = torch.device('cpu')
    assert not torch.cuda.is_available()
    if args.check:
        protocol()
        data, _ = build()
        check(data)
    else:
        snapshot()
        if (OUT / 'complete.json').exists():
            print('Four-player continuation v3 already complete.', flush=True)
        else:
            path = OUT / 'dataset.pt'
            if not (OUT / 'data-provenance.json').exists():
                data, provenance = build()
                atomic_save(path, data)
                dump(OUT / 'data-provenance.json', dict(provenance, dataset_sha256=sha(path), protocol_sha256=sha(OUT / 'protocol.json')))
            else:
                provenance = load(OUT / 'data-provenance.json')
                assert provenance['dataset_sha256'] == sha(path) and provenance['protocol_sha256'] == sha(OUT / 'protocol.json')
                data = torch.load(path, map_location='cpu', weights_only=True)
            train(data)
