"""Train tiny action-only opponent policies for fast JS rollouts."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pretrain_policy_v2 import ACTION_SIZE, HAND_SIZE, encode_action


class CompactProxy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList((
            nn.Linear(ACTION_SIZE, 48),
            nn.Linear(48, 24),
            nn.Linear(24, 1),
        ))

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        value = actions
        for layer in self.layers[:-1]:
            value = torch.nn.functional.silu(layer(value))
        return self.layers[-1](value).squeeze(-1)


def read_records(path: Path, strategy: str) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("strategy") == strategy and record.get("deckMode") == "adaptive":
                records.append(record)
    return records


def tensorize(records: list[dict]) -> TensorDataset:
    actions = torch.zeros((len(records), HAND_SIZE, ACTION_SIZE), dtype=torch.float32)
    mask = torch.zeros((len(records), HAND_SIZE), dtype=torch.bool)
    chosen = torch.zeros(len(records), dtype=torch.long)
    for row, record in enumerate(records):
        hand = list(record["hand"])
        for index, card in enumerate(hand):
            actions[row, index] = torch.tensor(encode_action(record, int(card)))
            mask[row, index] = True
        chosen[row] = hand.index(int(record["chosenCard"]))
    return TensorDataset(actions, mask, chosen)


@torch.inference_mode()
def evaluate(model: CompactProxy, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    correct = top2 = count = 0
    nll = 0.0
    for actions, mask, chosen in loader:
        actions, mask, chosen = actions.to(device), mask.to(device), chosen.to(device)
        logits = model(actions).masked_fill(~mask, -1e9)
        correct += (logits.argmax(1) == chosen).sum().item()
        top2 += (logits.topk(2, 1).indices == chosen[:, None]).any(1).sum().item()
        nll += torch.nn.functional.cross_entropy(logits, chosen, reduction="sum").item()
        count += chosen.numel()
    return {"top1": correct / count, "top2": top2 / count, "nll": nll / count}


def compact(tensor: torch.Tensor) -> list:
    values = tensor.detach().cpu().tolist()
    if values and isinstance(values[0], list):
        return [[round(float(value), 7) for value in row] for row in values]
    return [round(float(value), 7) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--strategy", choices=("champion", "external_mcs"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=2026082107)
    args = parser.parse_args()
    executable = Path(sys.executable).resolve()
    if executable.parent.name.lower() != "ntw-ai":
        raise RuntimeError(f"refusing to train outside ntw-ai: {executable}")
    if not torch.cuda.is_available():
        raise RuntimeError("compact proxy training requires CUDA")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    records = [record for path in args.data for record in read_records(path, args.strategy)]
    games = sorted({int(record["gameIndex"]) for record in records})
    validation_games = set(games[::5])
    train = [record for record in records if int(record["gameIndex"]) not in validation_games]
    validation = [record for record in records if int(record["gameIndex"]) in validation_games]
    print(f"[compact-proxy] executable={executable} gpu={torch.cuda.get_device_name(0)}")
    print(f"[compact-proxy] strategy={args.strategy} train={len(train)} validation={len(validation)}")
    train_loader = DataLoader(tensorize(train), batch_size=1024, shuffle=True, pin_memory=True)
    validation_loader = DataLoader(tensorize(validation), batch_size=2048, shuffle=False, pin_memory=True)
    model = CompactProxy().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-5)
    best = None
    best_nll = float("inf")
    best_metrics = {}
    for epoch in range(1, args.epochs + 1):
        model.train()
        for actions, mask, chosen in train_loader:
            actions, mask, chosen = actions.to(device), mask.to(device), chosen.to(device)
            logits = model(actions).masked_fill(~mask, -1e9)
            loss = torch.nn.functional.cross_entropy(logits, chosen)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        metrics = evaluate(model, validation_loader, device)
        if metrics["nll"] < best_nll:
            best_nll = metrics["nll"]
            best_metrics = metrics
            best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(f"[compact-proxy] epoch={epoch:03d} top1={metrics['top1']:.3%} top2={metrics['top2']:.3%} nll={metrics['nll']:.4f}")
    if best is None:
        raise RuntimeError("no compact proxy checkpoint")
    model.load_state_dict(best)
    manifest = {
        "format": "ntw-compact-opponent-v1",
        "strategy": args.strategy,
        "actionSize": ACTION_SIZE,
        "activation": "silu",
        "layers": [{"weight": compact(layer.weight), "bias": compact(layer.bias)} for layer in model.layers],
        "validation": best_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"[compact-proxy] wrote={args.output} metrics={best_metrics}")


if __name__ == "__main__":
    main()
