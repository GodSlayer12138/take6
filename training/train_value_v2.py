"""Fit a public-information terminal-cost value model from exact JS arena games."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from pretrain_policy_v2 import encode_state
from train_policy_v2_offline import read_trajectories
from vectorized_ppo import FEATURE_VERSION, FORMAT, STATE_SIZE, PolicyValueNet, export_model


def tensorize(games: list[list[dict]], rank_weight: float, win_weight: float, objective: str = "cost") -> TensorDataset:
    rows: list[tuple[dict, list[list[int]]]] = []
    for trajectory in games:
        own_history: list[int] = []
        for index, record in enumerate(trajectory):
            histories = record.get("playedCards") or [own_history.copy(), [], [], [], []]
            rows.append((record, histories))
            if index + 1 < len(trajectory):
                removed = set(record["hand"]) - set(trajectory[index + 1]["hand"])
                if len(removed) == 1:
                    own_history.append(removed.pop())
    states = torch.zeros((len(rows), STATE_SIZE), dtype=torch.float32)
    targets = torch.zeros(len(rows), dtype=torch.float32)
    turns = torch.zeros(len(rows), dtype=torch.long)
    for index, (record, histories) in enumerate(rows):
        states[index] = torch.tensor(encode_state(record, histories))
        current_score = float((record.get("scores") or [0])[0])
        future_penalty = max(0.0, float(record["finalScore"]) - current_score)
        cost = (
            future_penalty
            + (float(record["finalRank"]) - 1.0) * rank_weight
            + (1.0 - float(record["winShare"])) * win_weight
        )
        targets[index] = float(record["winShare"]) if objective == "win" else -cost
        turns[index] = int(record["turn"])
    return TensorDataset(states, targets, turns)


def value_forward(model: PolicyValueNet, states: torch.Tensor) -> torch.Tensor:
    context = states
    for layer in model.state_layers:
        context = F.silu(layer(context))
    return model.value_layers[1](F.silu(model.value_layers[0](context))).squeeze(-1)


@torch.inference_mode()
def evaluate(model: PolicyValueNet, loader: DataLoader, device: torch.device, objective: str) -> dict[str, float]:
    model.eval()
    squared = absolute = count = 0.0
    by_turn_squared = [0.0] * 10
    by_turn_count = [0] * 10
    for states, targets, turns in loader:
        states, targets = states.to(device), targets.to(device)
        prediction = value_forward(model, states)
        calibrated = prediction.sigmoid() if objective == "win" else prediction
        error = calibrated - targets
        squared += error.square().sum().item()
        absolute += error.abs().sum().item()
        count += targets.numel()
        for turn in range(10):
            selected = turns == turn
            if selected.any():
                by_turn_squared[turn] += error.detach().cpu()[selected].square().sum().item()
                by_turn_count[turn] += selected.sum().item()
    result = {
        "mse": squared / max(1, count),
        "mae": absolute / max(1, count),
        **{f"turn{turn}Mse": by_turn_squared[turn] / max(1, by_turn_count[turn]) for turn in range(10)},
    }
    if objective == "win":
        result["bce"] = sum(
            F.binary_cross_entropy_with_logits(value_forward(model, states.to(device)), targets.to(device), reduction="sum").item()
            for states, targets, _turns in loader
        ) / max(1, count)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--validation", type=Path, nargs="+", required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--rank-weight", type=float, default=0.8)
    parser.add_argument("--win-weight", type=float, default=4.0)
    parser.add_argument("--objective", choices=("cost", "win"), default="cost")
    parser.add_argument("--seed", type=int, default=2026082405)
    args = parser.parse_args()

    executable = Path(sys.executable).resolve()
    if executable.parent.name.lower() != "ntw-ai":
        raise RuntimeError(f"refusing to train outside ntw-ai: {executable}")
    if not torch.cuda.is_available():
        raise RuntimeError("value training requires CUDA")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    train_games = read_trajectories(args.train)
    validation_games = read_trajectories(args.validation)
    random.shuffle(train_games)
    train_loader = DataLoader(tensorize(train_games, args.rank_weight, args.win_weight, args.objective), batch_size=args.batch_size, shuffle=True, pin_memory=True)
    validation_loader = DataLoader(tensorize(validation_games, args.rank_weight, args.win_weight, args.objective), batch_size=args.batch_size * 2, pin_memory=True)
    print(f"[value-v2] executable={executable} torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}")
    print(f"[value-v2] trainGames={len(train_games)} validationGames={len(validation_games)}")

    payload = torch.load(args.initial, map_location="cpu", weights_only=False)
    if payload.get("format") != FORMAT or payload.get("featureVersion") != FEATURE_VERSION:
        raise RuntimeError(f"incompatible initial checkpoint: {args.initial}")
    model = PolicyValueNet().to(device)
    model.load_state_dict(payload["state_dict"])
    for parameter in model.action_layers.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=2e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05)
    best_state = None
    best_metrics: dict[str, float] = {}
    best_mse = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = examples = 0.0
        for states, targets, _turns in train_loader:
            states, targets = states.to(device), targets.to(device)
            prediction = value_forward(model, states)
            loss = F.binary_cross_entropy_with_logits(prediction, targets) if args.objective == "win" else F.smooth_l1_loss(prediction, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * targets.numel()
            examples += targets.numel()
        scheduler.step()
        metrics = evaluate(model, validation_loader, device, args.objective)
        if metrics["mse"] < best_mse:
            best_mse = metrics["mse"]
            best_metrics = metrics
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0:
            extra = f" valBce={metrics['bce']:.4f}" if args.objective == "win" else ""
            print(f"[value-v2] epoch={epoch:03d} loss={total/examples:.4f} valMse={metrics['mse']:.4f} valMae={metrics['mae']:.4f}{extra}")
    if best_state is None:
        raise RuntimeError("value training produced no checkpoint")
    model.load_state_dict(best_state)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "optimizer": None, "format": FORMAT, "featureVersion": FEATURE_VERSION, "update": 0, "metrics": best_metrics, "training": f"real-js-terminal-{args.objective}"}, args.checkpoint)
    export_model(model, args.output, 0, best_metrics)
    print(f"[value-v2] checkpoint={args.checkpoint}")
    print(f"[value-v2] browser_model={args.output}")
    print(json.dumps(best_metrics, sort_keys=True))


if __name__ == "__main__":
    main()
