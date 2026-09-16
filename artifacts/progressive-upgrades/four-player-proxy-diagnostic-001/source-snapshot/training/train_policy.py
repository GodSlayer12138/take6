"""Distill the high-budget information-set search into a browser MLP.

All Python commands for this project are expected to run in the ``ntw-ai``
Conda environment. The exported JSON contains only dense weights and is safe
for the browser worker to load.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


FEATURE_VERSION = 1
FEATURE_SIZE = 270
TARGET_SCALE = 10.0


def bull_heads(card: int) -> int:
    if card == 55:
        return 7
    if card % 11 == 0:
        return 5
    if card % 10 == 0:
        return 3
    if card % 10 == 5:
        return 2
    return 1


def row_penalty(row: list[int]) -> int:
    return sum(bull_heads(card) for card in row)


def target_row_index(rows: list[list[int]], card: int) -> int:
    best_index = -1
    best_tail = -1
    for index, row in enumerate(rows):
        tail = row[-1]
        if best_tail < tail < card:
            best_tail = tail
            best_index = index
    return best_index


def cheapest_row_index(rows: list[list[int]]) -> int:
    return min(range(4), key=lambda index: (row_penalty(rows[index]), len(rows[index])))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def placement_features(record: dict, card: int) -> list[float]:
    rows = record["rows"]
    hand = record["hand"]
    deck_size = record["deckSize"]
    player_count = record["playerCount"]
    row_index = target_row_index(rows, card)
    too_low = row_index < 0
    if too_low:
        row_index = cheapest_row_index(rows)
    row = rows[row_index]
    tail = row[-1]
    captures = too_low or len(row) >= 5
    immediate = row_penalty(row) if captures else 0
    gap = 0 if too_low else max(0, card - tail - 1)
    open_slots = 0 if too_low else max(0, 5 - len(row))

    known = set(record["seenCards"]) | set(hand)
    unknown = 0
    interval = 0
    for candidate in range(1, deck_size + 1):
        if candidate in known:
            continue
        unknown += 1
        if not too_low and tail < candidate < card:
            interval += 1
    interval_fraction = interval / max(1, unknown)
    traffic = interval_fraction * max(0, player_count - 1)
    min_tail = min(row_[-1] for row_ in rows)
    trapped = sum(candidate != card and candidate < min_tail for candidate in hand)
    sorted_hand = sorted(hand)
    rank = sorted_hand.index(card)
    left_gap = card - sorted_hand[rank - 1] if rank > 0 else card
    right_gap = sorted_hand[rank + 1] - card if rank + 1 < len(sorted_hand) else deck_size + 1 - card
    row_one_hot = [1.0 if index == row_index else 0.0 for index in range(4)]
    values = [
        card / 104,
        card / deck_size,
        bull_heads(card) / 7,
        rank / max(1, len(sorted_hand) - 1),
        left_gap / 104,
        right_gap / 104,
        *row_one_hot,
        gap / 104,
        open_slots / 5,
        immediate / 25,
        float(too_low),
        float(captures),
        interval_fraction,
        traffic / 9,
        trapped / 9,
    ]
    return [clamp01(value) for value in values]


def encode_candidate(record: dict, card: int) -> list[float]:
    hand_mask = [0.0] * 104
    for value in record["hand"]:
        hand_mask[value - 1] = 1.0
    seen_mask = [0.0] * 104
    for value in record["seenCards"]:
        seen_mask[value - 1] = 1.0
    board: list[float] = []
    for row in record["rows"]:
        board.extend(value / 104 for value in row)
        board.extend([0.0] * (6 - len(row)))
    row_summaries: list[float] = []
    for row in record["rows"]:
        row_summaries.extend((len(row) / 5, clamp01(row_penalty(row) / 25), row[-1] / 104))
    scores = record.get("scores") or [0] * record["playerCount"]
    own_score = scores[0]
    other_scores = scores[1:]
    score_summaries = [
        clamp01(own_score / 50),
        clamp01((min(other_scores) if other_scores else 0) / 50),
        clamp01((sum(other_scores) / max(1, len(other_scores))) / 50),
        clamp01((max(other_scores) if other_scores else 0) / 50),
    ]
    globals_ = [
        record["deckSize"] / 104,
        record["playerCount"] / 10,
        len(record["hand"]) / 10,
        (10 - len(record["hand"])) / 10,
    ]
    features = [
        *hand_mask,
        *seen_mask,
        *board,
        *row_summaries,
        *score_summaries,
        *globals_,
        *placement_features(record, card),
    ]
    if len(features) != FEATURE_SIZE:
        raise ValueError(f"Feature contract mismatch: {len(features)} != {FEATURE_SIZE}")
    return features


def read_records(paths: Iterable[Path], limit: int | None = None, player_count: int | None = None, deck_mode: str | None = None) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if "meta" in record:
                    continue
                if player_count is not None and record["playerCount"] != player_count:
                    continue
                if deck_mode is not None and record["deckMode"] != deck_mode:
                    continue
                records.append(record)
                if limit and len(records) >= limit:
                    return records
    return records


def tensorize(records: list[dict]) -> TensorDataset:
    count = len(records)
    features = torch.zeros((count, 10, FEATURE_SIZE), dtype=torch.float32)
    targets = torch.zeros((count, 10), dtype=torch.float32)
    mask = torch.zeros((count, 10), dtype=torch.bool)
    for row_index, record in enumerate(records):
        labels = record["labels"]
        for candidate_index, (card, utility) in enumerate(labels):
            features[row_index, candidate_index] = torch.tensor(encode_candidate(record, card))
            targets[row_index, candidate_index] = float(utility) / TARGET_SCALE
            mask[row_index, candidate_index] = True
    return TensorDataset(features, targets, mask)


class CandidateValueNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(FEATURE_SIZE, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 1),
        ])

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            values = torch.relu(layer(values))
        return self.layers[-1](values).squeeze(-1)


def distillation_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    temperature = 1.35
    invalid = torch.finfo(prediction.dtype).min
    predicted_logits = (-prediction * TARGET_SCALE / temperature).masked_fill(~mask, invalid)
    target_logits = (-target * TARGET_SCALE / temperature).masked_fill(~mask, invalid)
    target_probs = torch.softmax(target_logits, dim=1)
    ranking = -(target_probs * torch.log_softmax(predicted_logits, dim=1)).sum(dim=1).mean()
    regression = torch.nn.functional.smooth_l1_loss(prediction[mask], target[mask])
    return ranking + 0.35 * regression, ranking, regression


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    decisions = 0
    loss_states = 0
    correct = 0
    top2 = 0
    regret_sum = 0.0
    loss_sum = 0.0
    for features, targets, mask in loader:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        prediction = model(features).masked_fill(~mask, float("inf"))
        chosen = prediction.argmin(dim=1)
        expert = targets.masked_fill(~mask, float("inf")).argmin(dim=1)
        decision_mask = mask.sum(dim=1) > 1
        correct += ((chosen == expert) & decision_mask).sum().item()
        top2_indices = prediction.topk(k=2, dim=1, largest=False).indices
        top2 += ((top2_indices == expert.unsqueeze(1)).any(dim=1) & decision_mask).sum().item()
        selected_target = targets.gather(1, chosen.unsqueeze(1)).squeeze(1)
        expert_target = targets.gather(1, expert.unsqueeze(1)).squeeze(1)
        regret_sum += (((selected_target - expert_target) * TARGET_SCALE) * decision_mask).sum().item()
        loss, _, _ = distillation_loss(model(features), targets, mask)
        loss_sum += loss.item() * features.shape[0]
        decisions += decision_mask.sum().item()
        loss_states += features.shape[0]
    return {
        "loss": loss_sum / max(1, loss_states),
        "accuracy": correct / max(1, decisions),
        "top2": top2 / max(1, decisions),
        "regret": regret_sum / max(1, decisions),
    }


def export_browser_model(model: CandidateValueNet, output: Path, metrics: dict[str, float], sources: list[str]) -> None:
    def compact(tensor: torch.Tensor) -> list:
        values = tensor.detach().cpu().tolist()
        if values and isinstance(values[0], list):
            return [[round(float(value), 7) for value in row] for row in values]
        return [round(float(value), 7) for value in values]

    layers = []
    for index, layer in enumerate(model.layers):
        layers.append({
            "source": f"layers.{index}",
            "weight": compact(layer.weight),
            "bias": compact(layer.bias),
        })
    manifest = {
        "format": "ntw-neural-v1",
        "name": "牧场冠军 GPU",
        "featureVersion": FEATURE_VERSION,
        "featureSize": FEATURE_SIZE,
        "targetScale": TARGET_SCALE,
        "activation": "relu",
        "layers": layers,
        "validation": metrics,
        "trainingSources": sources,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--validation", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=Path("src/game/models/ntw-champion.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/models/ntw-champion.pt"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--player-count", type=int)
    parser.add_argument("--deck-mode", choices=("adaptive", "classic"))
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device} {torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}")

    train_records = read_records(args.train, args.limit, args.player_count, args.deck_mode)
    if args.validation:
        validation_records = read_records(args.validation, args.limit, args.player_count, args.deck_mode)
    else:
        split = max(1, int(len(train_records) * 0.9))
        generator = random.Random(args.seed)
        generator.shuffle(train_records)
        validation_records = train_records[split:]
        train_records = train_records[:split]
    print(f"[train] tensorizing {len(train_records)} train / {len(validation_records)} validation states")
    train_dataset = tensorize(train_records)
    validation_dataset = tensorize(validation_records)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=0, pin_memory=pin_memory)

    model = CandidateValueNet().to(device)
    if args.initial_checkpoint:
        initial = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
        if initial.get("featureVersion") != FEATURE_VERSION or initial.get("featureSize") != FEATURE_SIZE:
            raise RuntimeError("Initial checkpoint feature contract is incompatible")
        model.load_state_dict(initial["state_dict"])
        print(f"[train] initialized from {args.initial_checkpoint}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.08)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_regret = math.inf
    best_state = None
    best_metrics: dict[str, float] = {}
    stale = 0
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_rank = 0.0
        running_regression = 0.0
        examples = 0
        for features, targets, mask in train_loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                prediction = model(features)
                loss, ranking, regression = distillation_loss(prediction, targets, mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            count = features.shape[0]
            running_loss += loss.item() * count
            running_rank += ranking.item() * count
            running_regression += regression.item() * count
            examples += count
        scheduler.step()
        metrics = evaluate(model, validation_loader, device)
        elapsed = time.perf_counter() - started
        print(
            f"[train] epoch={epoch:03d} loss={running_loss/examples:.4f} "
            f"rank={running_rank/examples:.4f} reg={running_regression/examples:.4f} "
            f"val_acc={metrics['accuracy']:.3%} top2={metrics['top2']:.3%} "
            f"regret={metrics['regret']:.4f} time={elapsed:.1f}s"
        )
        if metrics["regret"] < best_regret - 1e-5:
            best_regret = metrics["regret"]
            best_metrics = metrics
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"[train] early stop after {epoch} epochs")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "featureVersion": FEATURE_VERSION,
        "featureSize": FEATURE_SIZE,
        "metrics": best_metrics,
        "sources": [str(path) for path in args.train],
    }, args.checkpoint)
    export_browser_model(model, args.output, best_metrics, [str(path) for path in args.train])
    print(f"[train] best={best_metrics}")
    print(f"[train] checkpoint={args.checkpoint}")
    print(f"[train] browser_model={args.output}")


if __name__ == "__main__":
    main()
