"""Independently check the completed CPU proxy refinement and its provenance."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from datetime import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from train_policy import CandidateValueNet

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/progressive-upgrades/opponent-proxies-v3-devshift'
OLD = ROOT / 'artifacts/progressive-upgrades/opponent-proxies-v2'
KINDS = ['dirv-10000', 'alpha-2000', 'mcs', 'champion-search']


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


@torch.inference_mode()
def score(model, data, kind):
    ids = np.flatnonzero((np.asarray(data['kind']) == kind) & np.asarray(data['validation']))
    total, correct = 0., 0
    for first in range(0, len(ids), 257):
        batch = ids[first:first + 257]
        features = torch.from_numpy(np.asarray(data['features'])[batch].copy())
        valid = torch.from_numpy(np.asarray(data['mask'])[batch].copy())
        action = torch.from_numpy(np.asarray(data['action'])[batch].copy())
        logits = (-model(features) * 10).double().masked_fill(~valid, float('-inf'))
        losses = torch.logsumexp(logits, dim=1) - logits.gather(1, action[:, None]).squeeze(1)
        total += float(losses.sum())
        correct += int((logits.argmax(1) == action).sum())
    return dict(states=len(ids), nll=total / len(ids), accuracy=correct / len(ids))


def main():
    torch.set_num_threads(2)
    assert not torch.cuda.is_available()
    protocol, complete, provenance = [load(OUT / name) for name in ('protocol.json', 'complete.json', 'data-provenance.json')]
    assert complete['status'] == 'complete'
    assert protocol['device'] == 'cpu' and protocol['cpu_threads'] == 2
    assert protocol['epochs'] == 12 and protocol['training_source_mix'] == protocol['validation_source_mix'] == [.5, .5]
    assert complete['protocol_sha256'] == provenance['protocol_sha256'] == sha(OUT / 'protocol.json')
    assert sha(OLD / 'dataset.pt') == protocol['original_dataset_sha256']
    assert sha(OLD / 'data-provenance.json') == protocol['original_data_provenance_sha256']
    for name, digest in protocol['source_sha256'].items():
        assert sha(ROOT / name) == sha(OUT / 'source-snapshot' / name) == digest
    for run, source in protocol['completed_development_sources'].items():
        folder = ROOT / 'artifacts/progressive-upgrades' / run
        assert run in ('development-004', 'development-005', 'development-006', 'development-007')
        manifest, result = load(folder / 'manifest.json'), load(folder / 'results.json')
        assert manifest['purpose'] == result['purpose'] == 'development' and result['audit'] == 'passed'
        assert sha(folder / 'manifest.json') == source['manifest_sha256']
        assert sha(folder / 'results.json') == source['results_sha256']
        assert sha(folder / 'blocks.jsonl') == source['trace_sha256'] == result['trace_sha256']
        assert result['games'] == source['complete_games']
    assert sha(OUT / 'new-features.npy') == provenance['features_sha256']
    assert sha(OUT / 'new-metadata.npz') == provenance['metadata_sha256']
    old = torch.load(OLD / 'dataset.pt', map_location='cpu', weights_only=True, mmap=True)
    new = dict(np.load(OUT / 'new-metadata.npz', allow_pickle=False))
    new['features'] = np.load(OUT / 'new-features.npy', mmap_mode='r', allow_pickle=False)
    groups = {False: set(), True: set()}
    for data in (old, new):
        seeds, flags = np.asarray(data['deal_seed']), np.asarray(data['validation'])
        expected = np.array([int.from_bytes(hashlib.sha256(f'proxy-v1:{int(seed)}'.encode()).digest()[:4], 'little') % 5 == 0 for seed in seeds])
        assert np.array_equal(expected, flags)
        for flag in groups:
            groups[flag].update(seeds[flags == flag].tolist())
    assert not groups[True] & groups[False]
    assert not set(old['deal_seed'].tolist()) & set(new['deal_seed'].tolist())
    assert len(new['action']) == provenance['states']
    assert complete['states'] == sum(int((~np.asarray(d['validation'])).sum()) for d in (old, new))
    assert complete['validation_states'] == sum(int(np.asarray(d['validation']).sum()) for d in (old, new))
    history = [json.loads(line) for line in (OUT / 'training.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(history) == complete['training_records'] == 48
    reports = {}
    for index, kind in enumerate(KINDS):
        assert sha(OLD / (kind + '.pt')) == protocol['initial_weights'][kind]
        saved, published = torch.load(OUT / (kind + '.pt'), map_location='cpu', weights_only=True), complete['models'][kind]
        assert sha(OUT / (kind + '.pt')) == published['weights_sha256']
        assert sha(OUT / (kind + '.json')) == published['json_sha256']
        rows = [row for row in history if row['opponent'] == kind]
        assert [r['epoch'] for r in rows] == list(range(1, 13))
        for row in rows:
            assert row == load(OUT / (kind + f'-epoch-{row["epoch"]:02d}.json'))
        for item in [published['initial']] + [r['validation'] for r in rows]:
            np.testing.assert_allclose(item['balanced_nll'], (item['old']['nll'] + item['new']['nll']) / 2, rtol=0, atol=1e-12)
        choices = [(published['initial']['balanced_nll'], 0, published['initial'])]
        choices += [(r['validation']['balanced_nll'], r['epoch'], r['validation']) for r in rows]
        _, epoch, selected = min(choices, key=lambda item: (item[0], item[1]))
        assert epoch == saved['epoch'] == published['best_epoch']
        assert selected == saved['validation'] == published['best']
        model = CandidateValueNet().eval()
        model.load_state_dict(torch.load(OLD / (kind + '.pt'), map_location='cpu', weights_only=True)['state_dict'])
        for name, data in (('old', old), ('new', new)):
            initial = score(model, data, index)
            np.testing.assert_allclose(initial['nll'], published['initial'][name]['nll'], rtol=0, atol=1e-6)
            np.testing.assert_allclose(initial['accuracy'], published['initial'][name]['accuracy'], rtol=0, atol=1e-6)
        model.load_state_dict(saved['state_dict'])
        serialized = load(OUT / (kind + '.json'))
        assert serialized['format'] == 'ntw-neural-v1' and serialized['featureSize'] == 270
        assert serialized['targetScale'] == 10 and serialized['activation'] == 'relu' and len(serialized['layers']) == 4
        for layer, record in zip(model.layers, serialized['layers']):
            np.testing.assert_array_equal(layer.weight.detach().numpy(), np.asarray(record['weight'], dtype=np.float32))
            np.testing.assert_array_equal(layer.bias.detach().numpy(), np.asarray(record['bias'], dtype=np.float32))
        scores = dict(old=score(model, old, index), new=score(model, new, index))
        for name, value in scores.items():
            assert value['states'] == selected[name]['states']
            np.testing.assert_allclose(value['nll'], selected[name]['nll'], rtol=0, atol=1e-6)
            np.testing.assert_allclose(value['accuracy'], selected[name]['accuracy'], rtol=0, atol=1e-6)
        reports[kind] = dict(epoch=epoch, independent_validation=scores, exact_pt_json_weights=True)
    report = dict(status='passed', observed_at=datetime.now().astimezone().isoformat(), models=reports,
                  protocol_sha256=sha(OUT / 'protocol.json'), complete_sha256=sha(OUT / 'complete.json'), auditor_sha256=sha(__file__),
                  whole_deal_train_validation_disjoint=True, source_and_output_hashes_checked=True,
                  all_48_epoch_records_checked=True, certificate_and_incomplete_development_outcomes_read=False,
                  game_strength='No game-strength inference; this audit does not accept an upgrade.')
    (OUT / 'independent-audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
