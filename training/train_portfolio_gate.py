"""Train a public-initial-state gate over full-game arena strategies."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pretrain_policy_v2 import encode_state
from vectorized_ppo import STATE_SIZE


def read(path: Path) -> tuple[list[str], list[dict]]:
    strategies: list[str] | None = None
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if "meta" in value:
                strategies = list(value["meta"]["strategies"])
            else:
                records.append(value)
    if not strategies:
        raise ValueError(f"missing portfolio metadata: {path}")
    return strategies, records


def tensorize(records: list[dict], count: int) -> TensorDataset:
    states = torch.tensor([encode_state({"turn": 0, **record}, record["playedCards"]) for record in records], dtype=torch.float32)
    rewards = torch.tensor([[float(result["reward"]) for result in record["results"]] for record in records], dtype=torch.float32)
    if rewards.shape[1] != count:
        raise ValueError("strategy count mismatch")
    advantages = rewards - rewards.mean(dim=1, keepdim=True)
    return TensorDataset(states, advantages, rewards)


class Gate(nn.Module):
    def __init__(self, outputs: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList((nn.Linear(STATE_SIZE, 128), nn.Linear(128, 64), nn.Linear(64, outputs)))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        value = torch.nn.functional.silu(self.layers[0](state))
        value = torch.nn.functional.silu(self.layers[1](value))
        return self.layers[2](value)


@torch.inference_mode()
def evaluate(model: Gate, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    chosen_sum = baseline_sum = oracle_sum = 0.0
    correct = count = 0
    mse = 0.0
    for states, advantages, rewards in loader:
        states, advantages, rewards = states.to(device), advantages.to(device), rewards.to(device)
        prediction = model(states)
        chosen = prediction.argmax(1)
        chosen_sum += rewards.gather(1, chosen[:, None]).sum().item()
        baseline_sum += rewards[:, 0].sum().item()
        oracle_sum += rewards.max(1).values.sum().item()
        correct += (chosen == rewards.argmax(1)).sum().item()
        mse += torch.nn.functional.mse_loss(prediction, advantages, reduction="sum").item()
        count += states.shape[0]
    return {
        "chosenReward": chosen_sum / count,
        "baselineReward": baseline_sum / count,
        "gain": (chosen_sum - baseline_sum) / count,
        "oracleGain": (oracle_sum - baseline_sum) / count,
        "accuracy": correct / count,
        "mse": mse / count,
    }


def compact(tensor: torch.Tensor) -> list:
    values = tensor.detach().cpu().tolist()
    if values and isinstance(values[0], list):
        return [[round(float(value), 7) for value in row] for row in values]
    return [round(float(value), 7) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=2026082605)
    args = parser.parse_args()
    executable = Path(sys.executable).resolve()
    if executable.parent.name.lower() != "ntw-ai" or not torch.cuda.is_available():
        raise RuntimeError(f"portfolio gate requires ntw-ai CUDA, got {executable}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    strategies: list[str] | None = None
    train_records: list[dict] = []
    for path in args.train:
        source_strategies, records = read(path)
        if strategies is not None and source_strategies != strategies:
            raise ValueError("strategy order mismatch")
        strategies = source_strategies
        train_records.extend(records)
    validation_strategies, validation_records = read(args.validation)
    if strategies != validation_strategies:
        raise ValueError("validation strategy order mismatch")
    assert strategies is not None
    train_loader = DataLoader(tensorize(train_records, len(strategies)), batch_size=256, shuffle=True, pin_memory=True)
    validation_loader = DataLoader(tensorize(validation_records, len(strategies)), batch_size=1024, pin_memory=True)
    model = Gate(len(strategies)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best = None
    best_gain = -float("inf")
    best_metrics: dict[str, float] = {}
    print(f"[portfolio-gate] executable={executable} gpu={torch.cuda.get_device_name(0)} train={len(train_records)} validation={len(validation_records)}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for states, advantages, _rewards in train_loader:
            states, advantages = states.to(device), advantages.to(device)
            prediction = model(states)
            loss = torch.nn.functional.smooth_l1_loss(prediction, advantages)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        metrics = evaluate(model, validation_loader, device)
        if metrics["gain"] > best_gain:
            best_gain = metrics["gain"]
            best_metrics = metrics
            best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(f"[portfolio-gate] epoch={epoch:03d} gain={metrics['gain']:+.4f} oracle={metrics['oracleGain']:+.4f} accuracy={metrics['accuracy']:.3%} mse={metrics['mse']:.4f}")
    assert best is not None
    model.load_state_dict(best)
    manifest = {
        "format": "ntw-portfolio-gate-v1",
        "stateSize": STATE_SIZE,
        "strategies": strategies,
        "activation": "silu",
        "layers": [{"weight": compact(layer.weight), "bias": compact(layer.bias)} for layer in model.layers],
        "validation": best_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"[portfolio-gate] wrote={args.output} metrics={best_metrics}")


if __name__ == "__main__":
    main()
