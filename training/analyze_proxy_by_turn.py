"""Report compact proxy accuracy by turn on a game-held-out split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from train_compact_opponent_proxy import CompactProxy, read_records, tensorize


def load_model(path: Path) -> CompactProxy:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    model = CompactProxy()
    with torch.no_grad():
        for layer, source in zip(model.layers, manifest["layers"]):
            layer.weight.copy_(torch.tensor(source["weight"]))
            layer.bias.copy_(torch.tensor(source["bias"]))
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--strategy", required=True)
    args = parser.parse_args()
    records = read_records(args.data, args.strategy)
    heldout = [record for record in records if int(record["gameIndex"]) % 5 == 0]
    model = load_model(args.model)
    print(f"model={args.model} strategy={args.strategy} heldout={len(heldout)}")
    for turn in range(10):
        subset = [record for record in heldout if int(record["turn"]) == turn]
        actions, mask, chosen = tensorize(subset).tensors
        with torch.inference_mode():
            logits = model(actions).masked_fill(~mask, -1e9)
            top1 = (logits.argmax(1) == chosen).float().mean().item()
            top2 = (logits.topk(2, 1).indices == chosen[:, None]).any(1).float().mean().item()
            nll = torch.nn.functional.cross_entropy(logits, chosen).item()
            temperatures = (0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0)
            calibrated = [(temperature, torch.nn.functional.cross_entropy(logits / temperature, chosen).item()) for temperature in temperatures]
            best_temperature, best_nll = min(calibrated, key=lambda item: item[1])
        print(f"turn={turn} top1={top1:.3%} top2={top2:.3%} nll={nll:.4f} bestT={best_temperature:.1f} calibratedNll={best_nll:.4f}")


if __name__ == "__main__":
    main()
