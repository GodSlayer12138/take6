"""Original take6 actor weights, restricted deserialization, FP32 inference.

TensorFlow Dense kernels are transposed for torch.nn.functional.linear.
Only the focal hand and public observations enter the 635-element input.
"""
from __future__ import annotations
from collections import OrderedDict
import hashlib
import io
import json
from pathlib import Path
import pickle
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'external/take6'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ArrayUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ('collections', 'OrderedDict'): OrderedDict,
            ('numpy', 'ndarray'): np.ndarray,
            ('numpy', 'dtype'): np.dtype,
            ('numpy.core.multiarray', '_reconstruct'): np._core.multiarray._reconstruct,
            ('numpy.core.multiarray', 'scalar'): np._core.multiarray.scalar,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f'Unexpected pickle global: {module}.{name}')
        return allowed[module, name]


def encode(hand, rows, seen, scores, seat, expert):
    """One-based cards; absolute-seat public scores, reordered per ScoreWrapper."""
    if not 1 <= len(hand) <= 10 or len(rows) != 4 or not 0 <= seat < len(scores):
        raise ValueError('Invalid public state')
    flat = sum(rows, [])
    if (len(set(hand)) != len(hand) or set(hand) & set(seen) or
            not set(flat) <= set(seen) or any(not 1 <= c <= 104 for c in hand + seen)):
        raise ValueError('Inconsistent public cards')
    x = np.zeros(635, dtype=np.float32)
    x[np.asarray(hand) - 1] = 1
    for i, row in enumerate(rows):
        x[104 + i * 104 + np.asarray(row) - 1] = 1
    x[520 + np.asarray(seen) - 1] = 1
    x[624:634] = -1
    order = [seat] + [i for i in range(len(scores)) if i != seat]
    x[624:624 + len(scores)] = np.minimum(1, np.asarray(scores, dtype=np.float32)[order] / 66)
    x[634] = float(expert)
    padded = np.asarray([0] * (10 - len(hand)) + sorted(hand), dtype=np.int64)
    return x, (padded > 0).astype(np.float32), padded


class Take6Actor:
    def __init__(self, players):
        relative = f'trained-anns/{players}-players-' + ('expert' if players == 2 else 'standard')
        path = UPSTREAM / relative
        provenance = json.loads((UPSTREAM / 'provenance.json').read_text())
        expected = next(r['sha256'] for r in provenance['files'] if r['path'] == relative)
        if sha(path) != expected:
            raise ValueError('Upstream weight digest mismatch')
        checkpoint = ArrayUnpickler(io.BytesIO(path.read_bytes())).load()
        weights = checkpoint['weights']
        shapes = [(635, 256), (256, 256), (256, 256), (256, 10)]
        names = ['fc_1', 'fc_2', 'fc_3', 'fc_out']
        self.arrays = []
        self.layers = []
        for name, shape in zip(names, shapes):
            kernel, bias = [weights[f'learner/{name}/{key}'] for key in ('kernel', 'bias')]
            assert kernel.shape == shape and bias.shape == (shape[1],)
            assert kernel.dtype == bias.dtype == np.float32
            assert np.isfinite(kernel).all() and np.isfinite(bias).all()
            self.arrays.append((kernel, bias))
            self.layers.append((torch.from_numpy(kernel.T.copy()), torch.from_numpy(bias.copy())))
        self.metadata = dict(path=str(path.relative_to(ROOT)), sha256=expected,
            actor_parameters=sum(k.size + b.size for k, b in self.arrays),
            actor_and_critic_parameters=sum(a.size for a in weights.values()),
            stored_global_timestep=int(checkpoint['global_timestep']),
            inference='Original FP32 actor weights, three 256-wide ReLU layers, masked softmax, seeded categorical sampling; PyTorch CPU port, not original TensorFlow runtime.')

    @torch.inference_mode()
    def logits(self, x):
        value = torch.from_numpy(x)
        for i, (weight, bias) in enumerate(self.layers):
            value = torch.nn.functional.linear(value, weight, bias)
            if i < 3:
                value = torch.relu(value)
        return value.numpy()

    @staticmethod
    def probabilities(logits, mask):
        masked = logits + np.where(mask > 0, np.float32(0), np.finfo(np.float32).min)
        exp = np.exp(masked - np.max(masked))
        return exp / exp.sum()

    def choose(self, hand, rows, seen, scores, seat, expert, rng):
        x, mask, padded = encode(hand, rows, seen, scores, seat, expert)
        probs = self.probabilities(self.logits(x), mask)
        card = int(padded[rng.choice(10, p=probs)])
        if card not in hand:
            raise RuntimeError('Invalid masked action')
        return card
