"""Offline advantage-weighted training for the public-history Policy v2."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from pretrain_policy_v2 import encode_action, encode_state
from vectorized_ppo import ACTION_SIZE, FORMAT, HAND_SIZE, STATE_SIZE, PolicyValueNet, export_model


def read_trajectories(paths: list[Path], strategy: str | None = None) -> list[list[dict]]:
    games: list[list[dict]] = []
    for path in paths:
        grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if "meta" in record or record.get("deckMode") != "adaptive":
                    continue
                if "chosenCard" not in record or "reward" not in record:
                    continue
                if strategy and record.get("strategy") != strategy:
                    continue
                grouped[(int(record["gameIndex"]), int(record.get("player", 0)))].append(record)
        games.extend(sorted(records, key=lambda item: item["turn"]) for records in grouped.values())
    return games


def tensorize(
    games: list[list[dict]],
    win_weight: float,
    rank_weight: float,
    score_weight: float,
) -> TensorDataset:
    records: list[tuple[dict, list[list[int]]]] = []
    for trajectory in games:
        own_history: list[int] = []
        for index, record in enumerate(trajectory):
            histories = record.get("playedCards") or [own_history.copy(), [], [], [], []]
            records.append((record, histories))
            if index + 1 < len(trajectory):
                removed = set(record["hand"]) - set(trajectory[index + 1]["hand"])
                if len(removed) == 1:
                    own_history.append(removed.pop())

    count = len(records)
    states = torch.zeros((count, STATE_SIZE), dtype=torch.float32)
    actions = torch.zeros((count, HAND_SIZE, ACTION_SIZE), dtype=torch.float32)
    masks = torch.zeros((count, HAND_SIZE), dtype=torch.bool)
    chosen = torch.zeros(count, dtype=torch.long)
    rewards = torch.zeros(count, dtype=torch.float32)
    behavior = torch.ones(count, dtype=torch.float32)
    turns = torch.zeros(count, dtype=torch.long)
    for index, (record, histories) in enumerate(records):
        states[index] = torch.tensor(encode_state(record, histories))
        hand = list(record["hand"])
        for candidate, card in enumerate(hand):
            actions[index, candidate] = torch.tensor(encode_action(record, int(card)))
            masks[index, candidate] = True
        chosen[index] = hand.index(int(record["chosenCard"]))
        if all(key in record for key in ("winShare", "finalRank", "finalScore")):
            rewards[index] = (
                float(record["winShare"]) * win_weight
                - (float(record["finalRank"]) - 1.0) * rank_weight
                - float(record["finalScore"]) * score_weight
            )
        else:
            rewards[index] = float(record["reward"])
        behavior[index] = max(0.02, float(record.get("behaviorProbability") or 1.0 / len(hand)))
        turns[index] = int(record["turn"])
    return TensorDataset(states, actions, masks, chosen, rewards, behavior, turns)


@torch.inference_mode()
def evaluate(model: PolicyValueNet, loader: DataLoader, device: torch.device, beta: float) -> dict[str, float]:
    model.eval()
    totals = defaultdict(float)
    examples = 0
    for state, action, mask, chosen, reward, _behavior, _turn in loader:
        state, action, mask = state.to(device), action.to(device), mask.to(device)
        chosen, reward = chosen.to(device), reward.to(device)
        logits, value = model(state, action)
        logits = logits.masked_fill(~mask, -1e9)
        log_prob = F.log_softmax(logits, dim=1).gather(1, chosen[:, None]).squeeze(1)
        advantage = reward - value
        weight = torch.exp(advantage / beta).clamp(0.15, 6.0)
        batch = state.shape[0]
        totals["nll"] += (-log_prob).sum().item()
        totals["awr"] += (-(weight * log_prob)).sum().item()
        totals["valueMse"] += F.mse_loss(value, reward, reduction="sum").item()
        totals["weight"] += weight.sum().item()
        examples += batch
    return {key: value / max(1, examples) for key, value in totals.items()}


def save(model: PolicyValueNet, checkpoint: Path, output: Path, epoch: int, metrics: dict[str, float]) -> None:
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": state,
        "optimizer": None,
        "format": FORMAT,
        "featureVersion": 2,
        "update": epoch,
        "metrics": metrics,
        "training": "offline-advantage-weighted",
    }, checkpoint)
    export_model(model, output, epoch, metrics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--validation-data",
        type=Path,
        nargs="+",
        help="Optional whole-file validation set that is never shuffled into training.",
    )
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--imitation-weight", type=float, default=0.04)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026082105)
    parser.add_argument("--win-weight", type=float, default=5.0)
    parser.add_argument("--rank-weight", type=float, default=0.42)
    parser.add_argument("--score-weight", type=float, default=0.075)
    parser.add_argument("--algorithm", choices=("awr", "reinforce"), default="awr")
    parser.add_argument("--strategy")
    parser.add_argument("--importance-clip", type=float, default=4.0)
    args = parser.parse_args()

    executable = Path(sys.executable).resolve()
    if executable.parent.name.lower() != "ntw-ai":
        raise RuntimeError(f"refusing to train outside ntw-ai: {executable}")
    if not torch.cuda.is_available():
        raise RuntimeError("offline Policy v2 training requires CUDA")
    device = torch.device("cuda")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(f"[offline-v2] executable={executable}")
    print(f"[offline-v2] torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}")

    games = read_trajectories(args.data, args.strategy)
    random.shuffle(games)
    if args.validation_data:
        train_games = games
        validation_games = read_trajectories(args.validation_data, args.strategy)
    else:
        split = max(1, math.floor(len(games) * 0.9))
        train_games = games[:split]
        validation_games = games[split:]
    if not train_games or not validation_games:
        raise RuntimeError(
            f"empty split: train={len(train_games)} validation={len(validation_games)}"
        )
    train_loader = DataLoader(
        tensorize(train_games, args.win_weight, args.rank_weight, args.score_weight),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        tensorize(validation_games, args.win_weight, args.rank_weight, args.score_weight),
        batch_size=args.batch_size * 2,
        pin_memory=True,
    )
    print(
        f"[offline-v2] games={len(train_games) + len(validation_games)} "
        f"train={len(train_games)} validation={len(validation_games)} "
        f"externalValidation={bool(args.validation_data)}"
    )

    payload = torch.load(args.initial, map_location="cpu", weights_only=False)
    model = PolicyValueNet().to(device)
    model.load_state_dict(payload["state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.1)
    latest: dict[str, float] = {}
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = defaultdict(float)
        examples = 0
        for state, action, mask, chosen, reward, behavior, _turn in train_loader:
            state, action, mask = state.to(device), action.to(device), mask.to(device)
            chosen, reward = chosen.to(device), reward.to(device)
            logits, value = model(state, action)
            logits = logits.masked_fill(~mask, -1e9)
            log_probs = F.log_softmax(logits, dim=1)
            selected_log_prob = log_probs.gather(1, chosen[:, None]).squeeze(1)
            entropy = -(log_probs.exp() * log_probs.masked_fill(~mask, 0)).sum(dim=1).mean()
            advantage = (reward - value.detach()).clamp(-8, 8)
            if args.algorithm == "reinforce":
                normalized_advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(0.2)
                importance = torch.exp(selected_log_prob.detach() - behavior.to(device).log()).clamp(
                    1.0 / args.importance_clip,
                    args.importance_clip,
                )
                awr_loss = -(importance * normalized_advantage * selected_log_prob).mean()
            else:
                weight = torch.exp(advantage / args.beta).clamp(0.15, 6.0)
                awr_loss = -(weight * selected_log_prob).mean()
            imitation_loss = -selected_log_prob.mean()
            value_loss = F.smooth_l1_loss(value, reward)
            loss = awr_loss + args.imitation_weight * imitation_loss + args.value_weight * value_loss - args.entropy_weight * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch = state.shape[0]
            totals["loss"] += loss.item() * batch
            totals["entropy"] += entropy.item() * batch
            examples += batch
        scheduler.step()
        if epoch == 1 or epoch % 5 == 0:
            latest = evaluate(model, validation_loader, device, args.beta)
            print(
                f"[offline-v2] epoch={epoch:03d} loss={totals['loss']/examples:.4f} "
                f"entropy={totals['entropy']/examples:.3f} valAwr={latest['awr']:.4f} "
                f"valNll={latest['nll']:.4f} valueMse={latest['valueMse']:.4f} weight={latest['weight']:.3f}"
            )
        if epoch % args.save_every == 0:
            numbered_pt = args.checkpoint.with_name(f"{args.checkpoint.stem}-e{epoch:03d}{args.checkpoint.suffix}")
            numbered_json = args.output.with_name(f"{args.output.stem}-e{epoch:03d}{args.output.suffix}")
            save(model, numbered_pt, numbered_json, epoch, latest)
    save(model, args.checkpoint, args.output, args.epochs, latest)
    print(f"[offline-v2] checkpoint={args.checkpoint}")
    print(f"[offline-v2] browser_model={args.output}")


if __name__ == "__main__":
    main()
