"""Distill a real arena opponent into a CUDA Policy-v2 proxy."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from pretrain_policy_v2 import encode_action, encode_state
from vectorized_ppo import ACTION_SIZE, FORMAT, HAND_SIZE, STATE_SIZE, PolicyValueNet, export_model


def read_records(paths: list[Path], strategy: str) -> list[dict]:
    records: list[dict] = []
    for source_index, path in enumerate(paths):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("strategy") == strategy and record.get("deckMode") == "adaptive":
                    # gameIndex is local to each generated data file. Preserve the
                    # source so distinct games with the same local index cannot
                    # leak across the game-held-out train/validation split.
                    record["_sourceIndex"] = source_index
                    records.append(record)
    return records


def tensorize(records: list[dict]) -> TensorDataset:
    count = len(records)
    states = torch.zeros((count, STATE_SIZE), dtype=torch.float32)
    actions = torch.zeros((count, HAND_SIZE, ACTION_SIZE), dtype=torch.float32)
    masks = torch.zeros((count, HAND_SIZE), dtype=torch.bool)
    chosen = torch.zeros(count, dtype=torch.long)
    for index, record in enumerate(records):
        histories = record.get("playedCards") or [[], [], [], [], []]
        states[index] = torch.tensor(encode_state(record, histories))
        hand = list(record["hand"])
        for candidate, card in enumerate(hand):
            actions[index, candidate] = torch.tensor(encode_action(record, int(card)))
            masks[index, candidate] = True
        chosen[index] = hand.index(int(record["chosenCard"]))
    return TensorDataset(states, actions, masks, chosen)


@torch.inference_mode()
def evaluate(model: PolicyValueNet, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    correct = 0
    top2 = 0
    nll = 0.0
    count = 0
    for state, action, mask, chosen in loader:
        state, action, mask, chosen = state.to(device), action.to(device), mask.to(device), chosen.to(device)
        logits, _value = model(state, action)
        logits = logits.masked_fill(~mask, -1e9)
        nll += F.cross_entropy(logits, chosen, reduction="sum").item()
        correct += (logits.argmax(dim=1) == chosen).sum().item()
        top2 += (logits.topk(2, dim=1).indices == chosen[:, None]).any(dim=1).sum().item()
        count += state.shape[0]
    return {"accuracy": correct / count, "top2": top2 / count, "nll": nll / count}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--strategy", choices=("champion", "external_mcs"), required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2026082108)
    args = parser.parse_args()

    executable = Path(sys.executable).resolve()
    if executable.parent.name.lower() != "ntw-ai" or not torch.cuda.is_available():
        raise RuntimeError(f"opponent distillation requires ntw-ai CUDA, got {executable}")
    device = torch.device("cuda")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    records = read_records(args.data, args.strategy)
    games = sorted({(int(record["_sourceIndex"]), int(record["gameIndex"])) for record in records})
    random.shuffle(games)
    validation_games = set(games[max(1, int(len(games) * 0.9)):])
    train = [
        record for record in records
        if (int(record["_sourceIndex"]), int(record["gameIndex"])) not in validation_games
    ]
    validation = [
        record for record in records
        if (int(record["_sourceIndex"]), int(record["gameIndex"])) in validation_games
    ]
    print(f"[proxy] executable={executable} gpu={torch.cuda.get_device_name(0)} strategy={args.strategy}")
    print(
        f"[proxy] sources={len(args.data)} games={len(games)} records={len(records)} "
        f"train={len(train)} validation={len(validation)}"
    )
    train_loader = DataLoader(tensorize(train), batch_size=args.batch_size, shuffle=True, pin_memory=True)
    validation_loader = DataLoader(tensorize(validation), batch_size=args.batch_size * 2, pin_memory=True)

    payload = torch.load(args.initial, map_location="cpu", weights_only=False)
    model = PolicyValueNet().to(device)
    model.load_state_dict(payload["state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.1)
    best_accuracy = -1.0
    best_state = None
    best_metrics: dict[str, float] = {}
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        examples = 0
        for state, action, mask, chosen in train_loader:
            state, action, mask, chosen = state.to(device), action.to(device), mask.to(device), chosen.to(device)
            logits, _value = model(state, action)
            logits = logits.masked_fill(~mask, -1e9)
            log_probs = F.log_softmax(logits, dim=1)
            entropy = -(log_probs.exp() * log_probs.masked_fill(~mask, 0)).sum(dim=1).mean()
            loss = F.nll_loss(log_probs, chosen) - args.entropy_weight * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * state.shape[0]
            examples += state.shape[0]
        scheduler.step()
        metrics = evaluate(model, validation_loader, device)
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_metrics = metrics
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"[proxy] epoch={epoch:03d} loss={loss_sum/examples:.4f} "
                f"accuracy={metrics['accuracy']:.3%} top2={metrics['top2']:.3%} nll={metrics['nll']:.4f}"
            )
    if best_state is None:
        raise RuntimeError("proxy training produced no checkpoint")
    model.load_state_dict(best_state)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "optimizer": None,
        "format": FORMAT,
        "featureVersion": 2,
        "update": 0,
        "metrics": best_metrics,
        "training": f"behavior-clone:{args.strategy}",
    }, args.checkpoint)
    export_model(model, args.output, 0, best_metrics)
    print(f"[proxy] checkpoint={args.checkpoint}")


if __name__ == "__main__":
    main()
