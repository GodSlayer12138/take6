"""Train a reduced Policy-v2 architecture on counterfactual action costs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from distill_policy_v2_compact import CompactPolicyV2, compact
from pretrain_policy_v2 import encode_action, encode_state
from vectorized_ppo import ACTION_SIZE, CARDS, HAND_SIZE, STATE_SIZE
from counterfactual_loss import optimal_set_loss, pairwise_ranking_loss


def read_records(paths: list[Path]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if "meta" not in record and record.get("deckMode") == "adaptive":
                    records.append(record)
    return records


def tensorize(records: list[dict]) -> TensorDataset:
    states = torch.zeros((len(records), STATE_SIZE), dtype=torch.float32)
    actions = torch.zeros((len(records), HAND_SIZE, ACTION_SIZE), dtype=torch.float32)
    costs = torch.zeros((len(records), HAND_SIZE), dtype=torch.float32)
    masks = torch.zeros((len(records), HAND_SIZE), dtype=torch.bool)
    for row, record in enumerate(records):
        states[row] = torch.tensor(encode_state(record, record.get("playedCards") or [[], [], [], [], []]))
        label_map = {int(card): float(cost) for card, cost in record["labels"]}
        for column, card in enumerate(record["hand"]):
            actions[row, column] = torch.tensor(encode_action(record, int(card)))
            costs[row, column] = label_map[int(card)]
            masks[row, column] = True
    return TensorDataset(states, actions, costs, masks)


@torch.inference_mode()
def evaluate(model: CompactPolicyV2, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    top1 = top2 = count = 0
    regret = 0.0
    for states, actions, costs, masks in loader:
        states, actions, costs, masks = states.to(device), actions.to(device), costs.to(device), masks.to(device)
        logits = model(states, actions).masked_fill(~masks, -1e9)
        predicted = logits.argmax(1)
        expert = costs.masked_fill(~masks, 1e9).argmin(1)
        decision = masks.sum(1) > 1
        top1 += ((predicted == expert) & decision).sum().item()
        top2 += ((logits.topk(2, 1).indices == expert[:, None]).any(1) & decision).sum().item()
        regret += ((costs.gather(1, predicted[:, None]) - costs.gather(1, expert[:, None])).squeeze(1) * decision).sum().item()
        count += decision.sum().item()
    decisions = max(1, count)
    return {"top1": top1 / decisions, "top2": top2 / decisions, "regret": regret / decisions}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--validation", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--regression-weight", type=float, default=0.0)
    parser.add_argument("--pairwise-weight", type=float, default=0.0)
    parser.add_argument("--pairwise-temperature", type=float, default=1.0)
    parser.add_argument("--optimal-weight", type=float, default=0.0)
    parser.add_argument("--state-widths", type=int, nargs=2, default=(192, 128))
    parser.add_argument("--action-widths", type=int, nargs=2, default=(96, 32))
    parser.add_argument("--seed", type=int, default=2026082619)
    args = parser.parse_args()
    auxiliary_weight = args.regression_weight + args.pairwise_weight + args.optimal_weight
    if min(args.regression_weight, args.pairwise_weight, args.optimal_weight) < 0 or auxiliary_weight > 1:
        raise ValueError("regression, pairwise, and optimal weights must be non-negative and sum to at most 1")
    if args.temperature <= 0 or args.pairwise_temperature <= 0:
        raise ValueError("temperatures must be positive")
    executable = Path(sys.executable).resolve()
    if executable.parent.name.lower() != "ntw-ai" or not torch.cuda.is_available():
        raise RuntimeError(f"structured counterfactual training requires ntw-ai CUDA, got {executable}")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    train_records = read_records(args.train)
    validation_records = read_records(args.validation)
    train_loader = DataLoader(tensorize(train_records), batch_size=args.batch_size, shuffle=True, pin_memory=True)
    validation_loader = DataLoader(tensorize(validation_records), batch_size=args.batch_size * 2, pin_memory=True)
    state_widths = tuple(args.state_widths)
    action_widths = tuple(args.action_widths)
    model = CompactPolicyV2(state_widths, action_widths).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=5e-5)
    best = None
    best_metrics: dict[str, float] = {}
    best_regret = float("inf")
    print(
        f"[structured-counterfactual] executable={executable} gpu={torch.cuda.get_device_name(0)} "
        f"train={len(train_records)} validation={len(validation_records)}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        for states, actions, costs, masks in train_loader:
            states, actions, costs, masks = states.to(device), actions.to(device), costs.to(device), masks.to(device)
            targets = torch.softmax(-costs.masked_fill(~masks, 1e9) / args.temperature, dim=1)
            logits = model(states, actions).masked_fill(~masks, -1e9)
            ranking_loss = -(targets * torch.log_softmax(logits, dim=1).masked_fill(~masks, 0)).sum(1).mean()
            counts = masks.sum(1, keepdim=True).clamp_min(1)
            target_mean = costs.masked_fill(~masks, 0).sum(1, keepdim=True) / counts
            logit_mean = logits.masked_fill(~masks, 0).sum(1, keepdim=True) / counts
            target_values = -(costs - target_mean) / 5.0
            predicted_values = logits - logit_mean
            regression_loss = torch.nn.functional.smooth_l1_loss(predicted_values[masks], target_values[masks])
            pairwise_loss = pairwise_ranking_loss(logits, costs, masks, args.pairwise_temperature)
            best_set_loss = optimal_set_loss(logits, costs, masks)
            listwise_weight = 1.0 - auxiliary_weight
            loss = (
                ranking_loss * listwise_weight
                + regression_loss * args.regression_weight
                + pairwise_loss * args.pairwise_weight
                + best_set_loss * args.optimal_weight
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        metrics = evaluate(model, validation_loader, device)
        if metrics["regret"] < best_regret:
            best_regret = metrics["regret"]
            best_metrics = metrics
            best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[structured-counterfactual] epoch={epoch:03d} top1={metrics['top1']:.3%} "
                f"top2={metrics['top2']:.3%} regret={metrics['regret']:.4f}"
            )
    assert best is not None
    model.load_state_dict(best)
    manifest = {
        "format": "ntw-policy-v2-compact",
        "featureVersion": 2,
        "cards": CARDS,
        "stateSize": STATE_SIZE,
        "actionSize": ACTION_SIZE,
        "activation": "silu",
        "strategy": "counterfactual_rollout_full_history",
        "stateWidths": list(state_widths),
        "actionWidths": list(action_widths),
        "temperature": args.temperature,
        "regressionWeight": args.regression_weight,
        "pairwiseWeight": args.pairwise_weight,
        "pairwiseTemperature": args.pairwise_temperature,
        "optimalWeight": args.optimal_weight,
        "stateLayers": [{"weight": compact(layer.weight), "bias": compact(layer.bias)} for layer in model.state_layers],
        "actionLayers": [{"weight": compact(layer.weight), "bias": compact(layer.bias)} for layer in model.action_layers],
        "validation": best_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"[structured-counterfactual] wrote={args.output} metrics={best_metrics}")


if __name__ == "__main__":
    main()
