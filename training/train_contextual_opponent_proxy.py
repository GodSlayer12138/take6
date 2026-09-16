"""Train a small contextual opponent proxy for information-set rollouts."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pretrain_policy_v2 import HAND_SIZE, bull_heads, encode_action, row_penalty

FEATURE_SIZE_V1 = 44
FEATURE_SIZE_V2 = 143


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def encode_contextual(record: dict, card: int) -> list[float]:
    hand = list(record["hand"])
    scores = list(record.get("scores") or [0] * 5)
    opponents = scores[1:] or [0]
    mean = sum(hand) / len(hand)
    variance = sum((value - mean) ** 2 for value in hand) / len(hand)
    hand_heads = sum(bull_heads(value) for value in hand) / len(hand)
    score_rank = sum(value < scores[0] for value in opponents) / 4
    rows = record["rows"]
    row_context = [value for row in rows for value in (row[-1] / 54, len(row) / 5, clamp01(row_penalty(row) / 25))]
    features = [
        *encode_action(record, card),
        (10 - len(hand)) / 10,
        len(hand) / 10,
        min(hand) / 54,
        max(hand) / 54,
        mean / 54,
        math.sqrt(variance) / 54,
        hand_heads / 7,
        clamp01(scores[0] / 50),
        clamp01(min(opponents) / 50),
        clamp01(sum(opponents) / len(opponents) / 50),
        clamp01(max(opponents) / 50),
        score_rank,
        *row_context,
        min(row[-1] for row in rows) / 54,
        max(row[-1] for row in rows) / 54,
        len(record["seenCards"]) / 54,
    ]
    if len(features) != FEATURE_SIZE_V1:
        raise RuntimeError(f"feature mismatch: {len(features)}")
    return features


def encode_contextual_v2(record: dict, card: int) -> list[float]:
    hand_mask = [0.0] * 54
    for value in record["hand"]:
        hand_mask[int(value) - 1] = 1.0
    seen_mask = [0.0] * 54
    for value in record["seenCards"]:
        seen_mask[int(value) - 1] = 1.0
    rows = record["rows"]
    row_context = [
        value
        for row in rows
        for value in (row[-1] / 54, len(row) / 5, clamp01(row_penalty(row) / 25))
    ]
    scores = list(record.get("scores") or [0] * 5)
    while len(scores) < 5:
        scores.append(0)
    features = [
        *encode_action(record, card),
        *hand_mask,
        *seen_mask,
        *row_context,
        *(clamp01(value / 50) for value in scores[:5]),
        (10 - len(record["hand"])) / 10,
    ]
    if len(features) != FEATURE_SIZE_V2:
        raise RuntimeError(f"v2 feature mismatch: {len(features)}")
    return features


class ContextualProxy(nn.Module):
    def __init__(self, feature_size: int = FEATURE_SIZE_V1, widths: tuple[int, int] | None = None) -> None:
        super().__init__()
        widths = widths or ((96, 48) if feature_size == FEATURE_SIZE_V2 else (64, 32))
        self.layers = nn.ModuleList((nn.Linear(feature_size, widths[0]), nn.Linear(widths[0], widths[1]), nn.Linear(widths[1], 1)))

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


def tensorize(records: list[dict], feature_version: int = 1) -> TensorDataset:
    feature_size = FEATURE_SIZE_V2 if feature_version == 2 else FEATURE_SIZE_V1
    encoder = encode_contextual_v2 if feature_version == 2 else encode_contextual
    actions = torch.zeros((len(records), HAND_SIZE, feature_size), dtype=torch.float32)
    mask = torch.zeros((len(records), HAND_SIZE), dtype=torch.bool)
    chosen = torch.zeros(len(records), dtype=torch.long)
    for row, record in enumerate(records):
        hand = list(record["hand"])
        for index, card in enumerate(hand):
            actions[row, index] = torch.tensor(encoder(record, int(card)))
            mask[row, index] = True
        chosen[row] = hand.index(int(record["chosenCard"]))
    return TensorDataset(actions, mask, chosen)


@torch.inference_mode()
def evaluate(model: ContextualProxy, loader: DataLoader, device: torch.device) -> dict[str, float]:
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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--feature-version", type=int, choices=(1, 2), default=1)
    parser.add_argument("--hidden1", type=int)
    parser.add_argument("--hidden2", type=int)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--seed", type=int, default=2026082108)
    args = parser.parse_args()
    executable = Path(sys.executable).resolve()
    if executable.parent.name.lower() != "ntw-ai":
        raise RuntimeError(f"refusing to train outside ntw-ai: {executable}")
    if not torch.cuda.is_available():
        raise RuntimeError("contextual proxy training requires CUDA")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    records = []
    for source_index, path in enumerate(args.data):
        for record in read_records(path, args.strategy):
            record["_sourceIndex"] = source_index
            records.append(record)
    games = sorted({(int(record["_sourceIndex"]), int(record["gameIndex"])) for record in records})
    validation_games = set(games[::5])
    train = [record for record in records if (int(record["_sourceIndex"]), int(record["gameIndex"])) not in validation_games]
    validation = [record for record in records if (int(record["_sourceIndex"]), int(record["gameIndex"])) in validation_games]
    print(f"[context-proxy] executable={executable} gpu={torch.cuda.get_device_name(0)}")
    print(f"[context-proxy] strategy={args.strategy} train={len(train)} validation={len(validation)}")
    train_loader = DataLoader(tensorize(train, args.feature_version), batch_size=1024, shuffle=True, pin_memory=True)
    validation_loader = DataLoader(tensorize(validation, args.feature_version), batch_size=2048, shuffle=False, pin_memory=True)
    feature_size = FEATURE_SIZE_V2 if args.feature_version == 2 else FEATURE_SIZE_V1
    widths = None
    if args.hidden1 is not None or args.hidden2 is not None:
        if args.hidden1 is None or args.hidden2 is None or min(args.hidden1, args.hidden2) < 1:
            raise ValueError("--hidden1 and --hidden2 must both be positive")
        widths = (args.hidden1, args.hidden2)
    model = ContextualProxy(feature_size, widths).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=3e-5)
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
            print(f"[context-proxy] epoch={epoch:03d} top1={metrics['top1']:.3%} top2={metrics['top2']:.3%} nll={metrics['nll']:.4f}")
    model.load_state_dict(best)
    manifest = {
        "format": f"ntw-contextual-opponent-v{args.feature_version}",
        "strategy": args.strategy,
        "featureSize": feature_size,
        "widths": list(widths or ((96, 48) if feature_size == FEATURE_SIZE_V2 else (64, 32))),
        "activation": "silu",
        "layers": [{"weight": compact(layer.weight), "bias": compact(layer.bias)} for layer in model.layers],
        "validation": best_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"[context-proxy] wrote={args.output} metrics={best_metrics}")


if __name__ == "__main__":
    main()
