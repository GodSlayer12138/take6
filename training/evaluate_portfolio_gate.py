"""Evaluate a saved public-state portfolio gate on held-out arena games."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pretrain_policy_v2 import encode_state
from train_portfolio_gate import Gate, read


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()

    strategies, records = read(args.data)
    manifest = json.loads(args.model.read_text(encoding="utf-8"))
    if strategies != manifest["strategies"]:
        raise ValueError("strategy order mismatch")
    model = Gate(len(strategies))
    with torch.no_grad():
        for layer, source in zip(model.layers, manifest["layers"]):
            layer.weight.copy_(torch.tensor(source["weight"]))
            layer.bias.copy_(torch.tensor(source["bias"]))
    model.eval()

    states = torch.tensor(
        [encode_state({"turn": 0, **record}, record["playedCards"]) for record in records],
        dtype=torch.float32,
    )
    with torch.inference_mode():
        selected = model(states).argmax(1).tolist()

    metrics = ("win", "score", "rank", "reward")
    baseline = {key: 0.0 for key in metrics}
    chosen = {key: 0.0 for key in metrics}
    oracle_win = 0.0
    counts = [0 for _ in strategies]
    per_strategy = [{key: 0.0 for key in metrics} for _ in strategies]
    for record, index in zip(records, selected):
        counts[index] += 1
        for key in metrics:
            baseline[key] += float(record["results"][0][key])
            chosen[key] += float(record["results"][index][key])
        oracle_win += max(float(result["win"]) for result in record["results"])
        for strategy_index, result in enumerate(record["results"]):
            for key in metrics:
                per_strategy[strategy_index][key] += float(result[key])
    count = len(records)
    output = {
        "games": count,
        "baseline": {key: value / count for key, value in baseline.items()},
        "chosen": {key: value / count for key, value in chosen.items()},
        "delta": {key: (chosen[key] - baseline[key]) / count for key in metrics},
        "oracleWin": oracle_win / count,
        "perStrategy": {
            strategy: {key: value / count for key, value in values.items()}
            for strategy, values in zip(strategies, per_strategy)
        },
        "selectionCounts": dict(zip(strategies, counts)),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
