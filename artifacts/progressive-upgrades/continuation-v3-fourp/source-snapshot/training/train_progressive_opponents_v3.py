"""CPU-only refinement of opponent proxies using completed development games.

The running replication and all certificate data are outside this experiment.
Historical tensors and new features are file-mapped to limit committed memory.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
from datetime import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn
import explore_small_strategies as e
from small_player_env import SmallGame

ROOT = e.ROOT
OUT = ROOT / 'artifacts/progressive-upgrades/opponent-proxies-v3-devshift'
OLD = ROOT / 'artifacts/progressive-upgrades/opponent-proxies-v2'
KINDS = ['dirv-10000', 'alpha-2000', 'mcs', 'champion-search']
RUNS = ['development-004', 'development-005', 'development-006', 'development-007']
CONFIG = dict(seed=91236001, games_per_player_count=2048, epochs=12, batch=256,
              lr=2e-5, weight_decay=1e-4, cpu_threads=2, device='cpu',
              training_source_mix=[.5, .5], validation_source_mix=[.5, .5],
              split='proxy-v1 whole original deal seed hash, validation when hash modulo 5 is zero',
              selection='Lowest equally weighted old/new validation NLL, including the initial v2 checkpoint; ties retain earlier epoch.',
              information='Acting opponent own hand and current public rows, seen cards and scores only; observed action is the target.',
              role='Development training only. No certificate trace or incomplete development run is used. No game-strength claim.')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def atomic_save(path, value):
    temporary = Path(str(path) + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def validation(seed):
    return int.from_bytes(hashlib.sha256(f'proxy-v1:{int(seed)}'.encode()).digest()[:4], 'little') % 5 == 0


def sources():
    provenance = {}
    selected = {3: [], 4: []}
    for name in RUNS:
        folder = ROOT / 'artifacts/progressive-upgrades' / name
        manifest, result = load(folder / 'manifest.json'), load(folder / 'results.json')
        assert manifest['purpose'] == result['purpose'] == 'development'
        assert result['audit'] == 'passed'
        digest = sha(folder / 'blocks.jsonl')
        assert result['trace_sha256'] == digest
        blocks, games = {3: set(), 4: set()}, 0
        with (folder / 'blocks.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                block = json.loads(line)
                p = block['players']
                assert p in blocks and block['block'] not in blocks[p]
                blocks[p].add(block['block'])
                assert len(block['records']) == p * len(manifest['versions'])
                games += len(block['records'])
                for record in block['records']:
                    assert record['deal_seed'] == block['deal_seed']
                    if any(kind in record['seats'] for kind in KINDS):
                        selected[p].append(record)
        assert all(len(ids) == manifest['deals'] for ids in blocks.values())
        assert games == result['games'] == manifest['deals'] * 7 * len(manifest['versions'])
        provenance[name] = dict(manifest_sha256=sha(folder / 'manifest.json'),
                                results_sha256=sha(folder / 'results.json'), trace_sha256=digest,
                                complete_games=games)
    rng = np.random.default_rng(CONFIG['seed'])
    for p in selected:
        records = selected[p]
        assert len(records) >= CONFIG['games_per_player_count']
        selected[p] = [records[i] for i in rng.permutation(len(records))[:CONFIG['games_per_player_count']]]
    return provenance, selected


def reconstruct(records, players):
    """Yield pre-action features; verify every source action and final score."""
    b, p = len(records), players
    cards = np.stack([np.random.default_rng(r['deal_seed']).permutation(104) + 1 for r in records])
    game = object.__new__(SmallGame)
    game.batch, game.players = b, p
    game.hands = np.sort(cards[:, :p * 10].reshape(b, p, 10), axis=-1)
    game.rows = np.zeros((b, 4, 6), dtype=np.int64)
    game.rows[:, :, 0] = cards[:, -1:-5:-1]
    game.lengths = np.ones((b, 4), dtype=np.int64)
    game.scores = np.zeros((b, p), dtype=np.float32)
    game.seen = np.zeros((b, 104), dtype=bool)
    game.seen[np.arange(b)[:, None], game.rows[:, :, 0] - 1] = True
    owners = np.array([[KINDS.index(k) if k in KINDS else -1 for k in r['seats']] for r in records])
    take = owners >= 0
    seeds = np.broadcast_to(np.array([r['deal_seed'] for r in records])[:, None], (b, p))[take]
    for turn in range(10):
        played = np.array([r['actions'][turn] for r in records]) + 1
        matches = game.hands == played[:, :, None]
        assert np.all(matches.sum(axis=-1) == 1), 'Illegal archived action'
        if turn < 9:
            h, n = 10 - turn, int(take.sum())
            features = np.zeros((n, 10, 270), dtype=np.float32)
            features[:, :h] = game.features()[take]
            yield dict(features=features, mask=np.broadcast_to(np.arange(10) < h, (n, 10)).copy(),
                       action=matches.argmax(axis=-1)[take], kind=owners[take], deal_seed=seeds)
        game.step(played)
    np.testing.assert_array_equal(game.scores, np.array([r['bullheads'] for r in records]))


def freeze(provenance):
    files = [Path(__file__), ROOT / 'training/explore_small_strategies.py',
             ROOT / 'training/small_player_env.py', ROOT / 'training/train_policy.py']
    value = dict(**CONFIG, completed_development_sources=provenance,
                 original_dataset_sha256=sha(OLD / 'dataset.pt'),
                 original_data_provenance_sha256=sha(OLD / 'data-provenance.json'),
                 initial_weights={k: sha(OLD / (k + '.pt')) for k in KINDS},
                 source_sha256={str(p.relative_to(ROOT)): sha(p) for p in files},
                 torch_version=torch.__version__, numpy_version=np.__version__)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        assert load(path) == value, 'Existing training protocol must stay frozen'
    else:
        dump(path, value)
        for file in files:
            target = OUT / 'source-snapshot' / file.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, target)
    for name, digest in value['source_sha256'].items():
        assert sha(OUT / 'source-snapshot' / name) == digest
    return value


def dataset(records):
    complete = OUT / 'data-provenance.json'
    feature_path, meta_path = OUT / 'new-features.npy', OUT / 'new-metadata.npz'
    if not complete.exists():
        n = sum(sum(k in KINDS for k in r['seats']) * 9 for group in records.values() for r in group)
        features = np.lib.format.open_memmap(feature_path, mode='w+', dtype=np.float32, shape=(n, 10, 270))
        values = {k: [] for k in ('mask', 'action', 'kind', 'deal_seed')}
        offset = 0
        for p, group in records.items():
            for first in range(0, len(group), 32):
                for batch in reconstruct(group[first:first + 32], p):
                    end = offset + len(batch['action'])
                    features[offset:end] = batch['features']
                    for key in values:
                        values[key].append(batch[key].copy())
                    offset = end
                if first % 512 == 0:
                    print(json.dumps(dict(phase='proxy-v3-data', players=p, games=min(first + 32, len(group)))), flush=True)
        assert offset == n
        features.flush()
        del features
        meta = {k: np.concatenate(v) for k, v in values.items()}
        meta['validation'] = np.array([validation(s) for s in meta['deal_seed']], dtype=bool)
        np.savez(meta_path, **meta)
        dump(complete, dict(states=n, unique_deals=len(np.unique(meta['deal_seed'])),
                            games_by_players={str(p): len(g) for p, g in records.items()},
                            validation_states=int(meta['validation'].sum()),
                            features_sha256=sha(feature_path), metadata_sha256=sha(meta_path),
                            protocol_sha256=sha(OUT / 'protocol.json'), all_selected_games_replayed=True,
                            no_certificate_or_incomplete_development_used=True))
    record = load(complete)
    assert record['protocol_sha256'] == sha(OUT / 'protocol.json')
    assert sha(feature_path) == record['features_sha256'] and sha(meta_path) == record['metadata_sha256']
    new = dict(np.load(meta_path, allow_pickle=False))
    new['features'] = np.load(feature_path, mmap_mode='r', allow_pickle=False)
    old = torch.load(OLD / 'dataset.pt', map_location='cpu', weights_only=True, mmap=True)
    for data in (old, new):
        seeds = np.asarray(data['deal_seed'])
        split = np.asarray(data['validation'])
        assert np.array_equal(split, np.array([validation(s) for s in seeds]))
        assert not set(seeds[split].tolist()) & set(seeds[~split].tolist())
    old_seeds, new_seeds = set(old['deal_seed'].tolist()), set(new['deal_seed'].tolist())
    assert not old_seeds & new_seeds, 'Fresh campaign development must have new deal seeds'
    return old, new


def tensors(data, ids):
    return tuple(torch.as_tensor(np.asarray(data[k])[ids].copy()) for k in ('features', 'mask', 'action'))


def logits(model, f, mask):
    return (-model(f) * 10).masked_fill(~mask, -1e9)


@torch.inference_mode()
def evaluate(model, data, ids):
    assert len(ids)
    loss, correct = 0., 0
    for first in range(0, len(ids), 512):
        f, mask, action = tensors(data, ids[first:first + 512])
        score = logits(model, f, mask)
        loss += float(nn.functional.cross_entropy(score, action, reduction='sum'))
        correct += int((score.argmax(-1) == action).sum())
    return dict(states=len(ids), nll=loss / len(ids), accuracy=correct / len(ids))


def metrics(model, data, indices):
    values = [evaluate(model, d, ids) for d, ids in zip(data, indices)]
    return dict(old=values[0], new=values[1], balanced_nll=sum(v['nll'] for v in values) / 2)


def save_best(kind, model, epoch, score):
    atomic_save(OUT / (kind + '.pt'), dict(state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                                        epoch=epoch, opponent=kind, validation=score))
    dump(OUT / (kind + '.json'), e.serial(model))


def train(data):
    summaries = {}
    for index, kind in enumerate(KINDS):
        torch.manual_seed(CONFIG['seed'] + index)
        rng = np.random.default_rng(CONFIG['seed'] + index)
        model = e.load(OLD / (kind + '.pt'))
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
        train_ids = [np.flatnonzero((np.asarray(d['kind']) == index) & ~np.asarray(d['validation'])) for d in data]
        validation_ids = [np.flatnonzero((np.asarray(d['kind']) == index) & np.asarray(d['validation'])) for d in data]
        assert all(len(ids) for ids in train_ids + validation_ids)
        resume = OUT / (kind + '-last.pt')
        if resume.exists():
            checkpoint = torch.load(resume, map_location='cpu', weights_only=True)
            assert checkpoint['protocol_sha256'] == sha(OUT / 'protocol.json')
            best_state = checkpoint['best_state_dict']
            model.load_state_dict(best_state)
            save_best(kind, model, checkpoint['best_epoch'], checkpoint['best'])
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            rng.bit_generator.state = checkpoint['numpy_rng']
            initial, best, best_epoch, start_epoch = checkpoint['initial'], checkpoint['best'], checkpoint['best_epoch'], checkpoint['epoch'] + 1
            dump(OUT / (kind + f'-epoch-{checkpoint["epoch"]:02d}.json'), checkpoint['training_record'])
        else:
            initial = metrics(model, data, validation_ids)
            best, best_epoch, start_epoch = initial, 0, 1
            dump(OUT / (kind + '-initial-validation.json'), initial)
            save_best(kind, model, 0, best)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        start = time.perf_counter()
        for epoch in range(start_epoch, CONFIG['epochs'] + 1):
            model.train()
            losses = []
            for _ in range(math.ceil(sum(map(len, train_ids)) / CONFIG['batch'])):
                batches = [tensors(d, rng.choice(ids, CONFIG['batch'] // 2, replace=True)) for d, ids in zip(data, train_ids)]
                f, mask, action = [torch.cat(parts) for parts in zip(*batches)]
                loss = nn.functional.cross_entropy(logits(model, f, mask), action)
                assert torch.isfinite(loss)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                losses.append(float(loss.detach()))
            model.eval()
            score = metrics(model, data, validation_ids)
            if score['balanced_nll'] < best['balanced_nll']:
                best, best_epoch = score, epoch
                save_best(kind, model, epoch, best)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            row = dict(opponent=kind, epoch=epoch, loss=float(np.mean(losses)), validation=score,
                       best_epoch=best_epoch, seconds=time.perf_counter() - start)
            atomic_save(resume, dict(state_dict=model.state_dict(), best_state_dict=best_state,
                                    optimizer=optimizer.state_dict(), numpy_rng=rng.bit_generator.state, training_record=row,
                                    epoch=epoch, initial=initial, best=best, best_epoch=best_epoch, protocol_sha256=sha(OUT / 'protocol.json')))
            # Each epoch has a separate record, so resumed runs do not append duplicate epochs.
            dump(OUT / (kind + f'-epoch-{epoch:02d}.json'), row)
            print(json.dumps(dict(phase='proxy-v3-train', **row)), flush=True)
        saved = torch.load(OUT / (kind + '.pt'), map_location='cpu', weights_only=True)
        assert saved['epoch'] == best_epoch and saved['validation'] == best
        model.load_state_dict(saved['state_dict'])
        np.testing.assert_allclose(metrics(model, data, validation_ids)['balanced_nll'], best['balanced_nll'], rtol=0, atol=1e-10)
        summaries[kind] = dict(initial=initial, best=best, best_epoch=best_epoch,
                               weights_sha256=sha(OUT / (kind + '.pt')), json_sha256=sha(OUT / (kind + '.json')))
    history = []
    for kind in KINDS:
        for epoch in range(1, CONFIG['epochs'] + 1):
            row = load(OUT / (kind + f'-epoch-{epoch:02d}.json'))
            assert row['opponent'] == kind and row['epoch'] == epoch
            history.append(row)
    (OUT / 'training.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in history), encoding='utf-8')
    dump(OUT / 'complete.json', dict(status='complete', completed_at=datetime.now().astimezone().isoformat(),
                                    protocol_sha256=sha(OUT / 'protocol.json'), models=summaries,
                                    states=sum(int((~np.asarray(d['validation'])).sum()) for d in data),
                                    validation_states=sum(int(np.asarray(d['validation']).sum()) for d in data),
                                    epochs_per_opponent=CONFIG['epochs'], training_records=len(history), source_sha256=sha(__file__),
                                    game_strength='Not evaluated; does not count as an accepted upgrade.'))
    print(json.dumps(dict(phase='proxy-v3-complete', models=summaries)), flush=True)


def check(records):
    old = torch.load(OLD / 'dataset.pt', map_location='cpu', weights_only=True, mmap=True)
    checked = 0
    for p, group in records.items():
        for batch in reconstruct(group[:4], p):
            assert np.isfinite(batch['features']).all()
            assert np.all(batch['mask'][np.arange(len(batch['action'])), batch['action']])
            checked += len(batch['action'])
    for index, kind in enumerate(KINDS):
        model = e.load(OLD / (kind + '.pt'))
        ids = np.flatnonzero(np.asarray(old['kind']) == index)[:16]
        f, mask, action = tensors(old, ids)
        loss = nn.functional.cross_entropy(logits(model, f, mask), action)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters())
        if index == 0:
            optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
            optimizer.step()
            rng = np.random.default_rng(12345)
            stream = io.BytesIO()
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), rng=rng.bit_generator.state), stream)
            stream.seek(0)
            restored = torch.load(stream, map_location='cpu', weights_only=True)
            other = e.load(OLD / (kind + '.pt'))
            other.load_state_dict(restored['model'])
            other_optimizer = torch.optim.AdamW(other.parameters(), lr=CONFIG['lr'])
            other_optimizer.load_state_dict(restored['optimizer'])
            other_rng = np.random.default_rng()
            other_rng.bit_generator.state = restored['rng']
            first, second = rng.choice(ids, 16), other_rng.choice(ids, 16)
            np.testing.assert_array_equal(first, second)
            for net, opt, chosen_ids in ((model, optimizer, first), (other, other_optimizer, second)):
                f, mask, action = tensors(old, chosen_ids)
                opt.zero_grad()
                nn.functional.cross_entropy(logits(net, f, mask), action).backward()
                opt.step()
            for a, b in zip(model.parameters(), other.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
    print(json.dumps(dict(status='passed', replayed_games=8, checked_new_actions=checked,
                          old_proxy_forward_backward_checks=4, exact_optimizer_resume_checks=1,
                          device=str(e.DEVICE), cuda_available=torch.cuda.is_available(),
                          output_written=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(CONFIG['cpu_threads'])
    e.DEVICE = torch.device('cpu')
    assert not torch.cuda.is_available()
    provenance, records = sources()
    if args.check:
        check(records)
    else:
        freeze(provenance)
        if (OUT / 'complete.json').exists():
            print('Opponent proxy v3 training already complete.', flush=True)
        else:
            train(dataset(records))
