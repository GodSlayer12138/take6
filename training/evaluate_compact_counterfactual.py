"""Evaluate an exported compact counterfactual policy on held-out records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train_compact_counterfactual import evaluate, read_records, tensorize
from train_contextual_opponent_proxy import ContextualProxy, FEATURE_SIZE_V2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--by-turn", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.model.read_text(encoding="utf-8"))
    if manifest.get("featureSize") != FEATURE_SIZE_V2:
        raise ValueError(f"incompatible feature size: {manifest.get('featureSize')}")
    device = torch.device("cuda")
    model = ContextualProxy(FEATURE_SIZE_V2, tuple(manifest["widths"])).to(device)
    with torch.no_grad():
        for layer, exported in zip(model.layers, manifest["layers"], strict=True):
            layer.weight.copy_(torch.tensor(exported["weight"], device=device))
            layer.bias.copy_(torch.tensor(exported["bias"], device=device))
    records = read_records(args.data)
    loader = DataLoader(tensorize(records), batch_size=2048, pin_memory=True)
    print(f"states={len(records)} metrics={evaluate(model, loader, device)}")
    if args.by_turn:
        for turn in range(10):
            turn_records = [record for record in records if int(record["turn"]) == turn]
            turn_loader = DataLoader(tensorize(turn_records), batch_size=2048, pin_memory=True)
            print(f"turn={turn} states={len(turn_records)} metrics={evaluate(model, turn_loader, device)}")


if __name__ == "__main__":
    main()
