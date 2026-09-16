"""Independent provenance, selection and numerical audit of continuation v3."""
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
from train_progressive_opponents_v3 import sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/progressive-upgrades/continuation-v3-fourp'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


@torch.inference_mode()
def evaluate(model, data, indices):
    total, correct = 0., 0
    for ids in indices.split(257):
        logits = (-model(data['features'][ids]) * 10).double().masked_fill(~data['mask'][ids], float('-inf'))
        target = data['action'][ids]
        total += float((torch.logsumexp(logits, 1) - logits.gather(1, target[:, None]).squeeze(1)).sum())
        correct += int((logits.argmax(1) == target).sum())
    return dict(states=len(indices), nll=total / len(indices), accuracy=correct / len(indices))


def main():
    torch.set_num_threads(2)
    assert not torch.cuda.is_available()
    protocol, provenance, complete = [load(OUT / name) for name in ('protocol.json', 'data-provenance.json', 'complete.json')]
    assert complete['status'] == 'complete'
    assert complete['protocol_sha256'] == provenance['protocol_sha256'] == sha(OUT / 'protocol.json')
    assert protocol['players'] == 4 and protocol['epochs'] == 20 and protocol['device'] == 'cpu'
    assert protocol['cpu_threads'] == 2 and protocol['row_permutation_augmentation']
    for path, digest in protocol['source_sha256'].items():
        assert sha(ROOT / path) == sha(OUT / 'source-snapshot' / path) == digest
    expected = {'development-004': ['specialist-prior-1024'], 'development-006': ['hybrid-wide-specialist'],
                'development-007': ['pair512-specialist2048']}
    assert set(protocol['completed_sources']) == set(expected)
    for run, values in protocol['completed_sources'].items():
        folder = ROOT / 'artifacts/progressive-upgrades' / run
        manifest, result = load(folder / 'manifest.json'), load(folder / 'results.json')
        assert manifest['purpose'] == result['purpose'] == 'development' and result['audit'] == 'passed'
        assert values['teachers'] == expected[run] and all(t in manifest['versions'] for t in values['teachers'])
        assert sha(folder / 'manifest.json') == values['manifest_sha256']
        assert sha(folder / 'results.json') == values['result_sha256']
        assert sha(folder / 'blocks.jsonl') == values['trace_sha256'] == result['trace_sha256']
        assert provenance['original_builder']['sources'][run] == dict(trace_sha256=values['trace_sha256'], teachers=values['teachers'])
    assert sha(OUT / 'dataset.pt') == provenance['dataset_sha256']
    data = torch.load(OUT / 'dataset.pt', map_location='cpu', weights_only=True)
    assert len(data['action']) == provenance['states'] == 20736
    assert bool((data['players'] == 4).all()) and provenance['selected_players'] == [4]
    assert len(data['deal_seed'].unique()) == provenance['unique_deals'] == 576
    expected_flags = torch.tensor([int.from_bytes(hashlib.sha256(f'continuation-v1:{seed}'.encode()).digest()[:4], 'little') % 5 == 0 for seed in data['deal_seed'].tolist()])
    assert torch.equal(expected_flags, data['validation'])
    train = torch.nonzero(~data['validation'], as_tuple=True)[0]
    validation = torch.nonzero(data['validation'], as_tuple=True)[0]
    assert not set(data['deal_seed'][train].tolist()) & set(data['deal_seed'][validation].tolist())
    assert len(train) == complete['states'] == 17028 and len(validation) == complete['validation_states'] == 3708
    assert sha(protocol['initial_checkpoint']) == protocol['initial_sha256']
    assert sha(OUT / 'model.pt') == complete['weights_sha256'] and sha(OUT / 'model.json') == complete['json_sha256']
    history = [json.loads(line) for line in (OUT / 'training.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [r['epoch'] for r in history] == list(range(1, 21))
    for row in history:
        assert row == load(OUT / f'epoch-{row["epoch"]:02d}.json')
    choices = [(complete['initial']['nll'], 0, complete['initial'])] + [(r['validation']['nll'], r['epoch'], r['validation']) for r in history]
    _, epoch, selected = min(choices, key=lambda item: (item[0], item[1]))
    weights = torch.load(OUT / 'model.pt', map_location='cpu', weights_only=True)
    assert epoch == weights['epoch'] == complete['best_epoch'] and selected == weights['validation'] == complete['best']
    model = CandidateValueNet().eval()
    independent = {}
    for name, path, published in [('initial', Path(protocol['initial_checkpoint']), complete['initial']),
                                   ('selected', OUT / 'model.pt', complete['best'])]:
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['state_dict'])
        score = evaluate(model, data, validation)
        np.testing.assert_allclose(score['nll'], published['nll'], rtol=0, atol=1e-6)
        np.testing.assert_allclose(score['accuracy'], published['accuracy'], rtol=0, atol=1e-6)
        independent[name] = score
    serialized = load(OUT / 'model.json')
    assert serialized['format'] == 'ntw-neural-v1' and serialized['featureSize'] == 270
    assert serialized['targetScale'] == 10 and serialized['activation'] == 'relu' and len(serialized['layers']) == 4
    for layer, values in zip(model.layers, serialized['layers']):
        np.testing.assert_array_equal(layer.weight.detach().numpy(), np.asarray(values['weight'], dtype=np.float32))
        np.testing.assert_array_equal(layer.bias.detach().numpy(), np.asarray(values['bias'], dtype=np.float32))
    result = dict(status='passed', observed_at=datetime.now().astimezone().isoformat(), selected_epoch=epoch,
                  independent_validation=independent, all_twenty_epochs_checked=True,
                  whole_deal_train_validation_disjoint=True, only_four_player_states_used=True,
                  source_and_output_hashes_checked=True, exact_pt_json_weights=True,
                  certificate_or_incomplete_development_outcomes_read=False,
                  auditor_sha256=sha(__file__), protocol_sha256=sha(OUT / 'protocol.json'), complete_sha256=sha(OUT / 'complete.json'),
                  interpretation='Training audit only; does not establish a game win-rate improvement.')
    (OUT / 'independent-audit.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
