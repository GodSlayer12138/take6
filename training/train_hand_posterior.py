"""Train a public-history posterior over opponents' remaining cards."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pretrain_policy_v2 import encode_state
from train_contextual_opponent_proxy import compact
from vectorized_ppo import CARDS, PLAYERS, STATE_SIZE

OPPONENTS = PLAYERS - 1


class HandPosterior(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList((
            nn.Linear(STATE_SIZE, 512),
            nn.Linear(512, 256),
            nn.Linear(256, CARDS * OPPONENTS),
        ))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        value = states
        for layer in self.layers[:-1]:
            value = torch.nn.functional.silu(layer(value))
        return self.layers[-1](value).reshape(-1, CARDS, OPPONENTS)


def read_examples(path: Path) -> tuple[list[tuple[list[float], list[int]]], list[tuple[list[float], list[int]]]]:
    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if "meta" not in record and record.get("deckMode") == "adaptive":
                groups[(int(record["gameIndex"]), int(record["turn"]))].append(record)
    train: list[tuple[list[float], list[int]]] = []
    validation: list[tuple[list[float], list[int]]] = []
    for (game_index, _turn), records in groups.items():
        if len(records) != PLAYERS:
            continue
        by_player = {int(record["player"]): record for record in records}
        focal = next((record for record in records if str(record.get("strategy", "")).startswith("neural_hybrid_v2")), None)
        if focal is None:
            continue
        focal_player = int(focal["player"])
        targets = [-1] * CARDS
        for offset in range(1, PLAYERS):
            player = (focal_player + offset) % PLAYERS
            for card in by_player[player]["hand"]:
                targets[int(card) - 1] = offset - 1
        state = encode_state(focal, focal.get("playedCards") or [[], [], [], [], []])
        (validation if game_index % 5 == 0 else train).append((state, targets))
    return train, validation


def tensorize(examples: list[tuple[list[float], list[int]]]) -> TensorDataset:
    states = torch.tensor([state for state, _targets in examples], dtype=torch.float32)
    targets = torch.tensor([targets for _state, targets in examples], dtype=torch.long)
    return TensorDataset(states, targets)


@torch.inference_mode()
def evaluate(model: HandPosterior, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    top1 = top2 = count = 0
    nll = 0.0
    by_turn = defaultdict(lambda: [0, 0, 0.0])
    for states, targets in loader:
        states, targets = states.to(device), targets.to(device)
        logits = model(states)
        mask = targets >= 0
        valid_logits = logits[mask]
        valid_targets = targets[mask]
        top1 += (valid_logits.argmax(1) == valid_targets).sum().item()
        top2 += (valid_logits.topk(2, 1).indices == valid_targets[:, None]).any(1).sum().item()
        nll += torch.nn.functional.cross_entropy(valid_logits, valid_targets, reduction="sum").item()
        count += valid_targets.numel()
        turns = torch.round(states[:, -1] * 10).long()
        for turn in turns.unique().tolist():
            state_mask = turns == turn
            card_mask = mask[state_mask]
            turn_logits = logits[state_mask][card_mask]
            turn_targets = targets[state_mask][card_mask]
            entry = by_turn[int(turn)]
            entry[0] += (turn_logits.argmax(1) == turn_targets).sum().item()
            entry[1] += turn_targets.numel()
            entry[2] += torch.nn.functional.cross_entropy(turn_logits, turn_targets, reduction="sum").item()
    metrics = {"top1": top1 / count, "top2": top2 / count, "nll": nll / count}
    metrics["byTurn"] = {str(turn): {"top1": values[0] / values[1], "nll": values[2] / values[1]} for turn, values in sorted(by_turn.items())}
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--seed", type=int, default=2026082644)
    args = parser.parse_args()
    executable = Path(sys.executable).resolve()
    if executable.parent.name.lower() != "ntw-ai" or not torch.cuda.is_available():
        raise RuntimeError(f"posterior training requires ntw-ai CUDA, got {executable}")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    train, validation = read_examples(args.data)
    train_loader = DataLoader(tensorize(train), batch_size=args.batch_size, shuffle=True, pin_memory=True)
    validation_loader = DataLoader(tensorize(validation), batch_size=args.batch_size * 2, pin_memory=True)
    model = HandPosterior().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=5e-5)
    best = None
    best_metrics = {}
    best_nll = float("inf")
    print(f"[hand-posterior] executable={executable} gpu={torch.cuda.get_device_name(0)} train={len(train)} validation={len(validation)}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for states, targets in train_loader:
            states, targets = states.to(device), targets.to(device)
            logits = model(states)
            mask = targets >= 0
            loss = torch.nn.functional.cross_entropy(logits[mask], targets[mask])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        metrics = evaluate(model, validation_loader, device)
        if metrics["nll"] < best_nll:
            best_nll = metrics["nll"]
            best_metrics = metrics
            best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(f"[hand-posterior] epoch={epoch:03d} top1={metrics['top1']:.3%} top2={metrics['top2']:.3%} nll={metrics['nll']:.4f}")
    assert best is not None
    model.load_state_dict(best)
    manifest = {
        "format": "ntw-hand-posterior-v1",
        "cards": CARDS,
        "opponents": OPPONENTS,
        "stateSize": STATE_SIZE,
        "activation": "silu",
        "layers": [{"weight": compact(layer.weight), "bias": compact(layer.bias)} for layer in model.layers],
        "validation": best_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"[hand-posterior] wrote={args.output} metrics={best_metrics}")


if __name__ == "__main__":
    main()
