"""Train a compact rollout policy from public-state counterfactual costs."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from train_contextual_opponent_proxy import ContextualProxy, FEATURE_SIZE_V2, compact, encode_contextual_v2
from counterfactual_loss import optimal_set_loss, pairwise_ranking_loss
from vectorized_ppo import HAND_SIZE


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
    features = torch.zeros((len(records), HAND_SIZE, FEATURE_SIZE_V2), dtype=torch.float32)
    costs = torch.zeros((len(records), HAND_SIZE), dtype=torch.float32)
    masks = torch.zeros((len(records), HAND_SIZE), dtype=torch.bool)
    for row, record in enumerate(records):
        label_map = {int(card): float(cost) for card, cost in record["labels"]}
        for column, card in enumerate(record["hand"]):
            features[row, column] = torch.tensor(encode_contextual_v2(record, int(card)))
            costs[row, column] = label_map[int(card)]
            masks[row, column] = True
    return TensorDataset(features, costs, masks)


@torch.inference_mode()
def evaluate(model: ContextualProxy, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    top1 = top2 = decisions = 0
    regret = 0.0
    for features, costs, masks in loader:
        features, costs, masks = features.to(device), costs.to(device), masks.to(device)
        logits = model(features).masked_fill(~masks, -1e9)
        predicted = logits.argmax(1)
        expert = costs.masked_fill(~masks, 1e9).argmin(1)
        decision = masks.sum(1) > 1
        selected_cost = costs.gather(1, predicted[:, None]).squeeze(1)
        best_cost = costs.gather(1, expert[:, None]).squeeze(1)
        top1 += ((predicted == expert) & decision).sum().item()
        top2 += ((logits.topk(2, 1).indices == expert[:, None]).any(1) & decision).sum().item()
        regret += ((selected_cost - best_cost) * decision).sum().item()
        decisions += decision.sum().item()
    count = max(1, decisions)
    return {"top1": top1 / count, "top2": top2 / count, "regret": regret / count}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--validation", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--regression-weight", type=float, default=0.0)
    parser.add_argument("--pairwise-weight", type=float, default=0.0)
    parser.add_argument("--pairwise-temperature", type=float, default=1.0)
    parser.add_argument("--optimal-weight", type=float, default=0.0)
    parser.add_argument("--hidden1", type=int, default=192)
    parser.add_argument("--hidden2", type=int, default=96)
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--anchor-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026082613)
    args = parser.parse_args()
    auxiliary_weight = args.regression_weight + args.pairwise_weight + args.optimal_weight + args.anchor_weight
    if min(args.regression_weight, args.pairwise_weight, args.optimal_weight, args.anchor_weight) < 0 or auxiliary_weight > 1:
        raise ValueError("regression, pairwise, optimal, and anchor weights must be non-negative and sum to at most 1")
    if args.temperature <= 0 or args.pairwise_temperature <= 0:
        raise ValueError("temperatures must be positive")
    if min(args.hidden1, args.hidden2) < 1:
        raise ValueError("hidden widths must be positive")
    executable = Path(sys.executable).resolve()
    if executable.parent.name.lower() != "ntw-ai" or not torch.cuda.is_available():
        raise RuntimeError(f"compact counterfactual training requires ntw-ai CUDA, got {executable}")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    train_records = read_records(args.train)
    validation_records = read_records(args.validation)
    train_loader = DataLoader(tensorize(train_records), batch_size=args.batch_size, shuffle=True, pin_memory=True)
    validation_loader = DataLoader(tensorize(validation_records), batch_size=args.batch_size * 2, pin_memory=True)
    widths = (args.hidden1, args.hidden2)
    model = ContextualProxy(FEATURE_SIZE_V2, widths).to(device)
    anchor_model = None
    if args.initial:
        manifest = json.loads(args.initial.read_text(encoding="utf-8"))
        if manifest.get("featureSize") != FEATURE_SIZE_V2 or tuple(manifest.get("widths") or ()) != widths:
            raise ValueError(f"incompatible initial compact model: {args.initial}")
        with torch.no_grad():
            for layer, exported in zip(model.layers, manifest["layers"], strict=True):
                layer.weight.copy_(torch.tensor(exported["weight"], device=device))
                layer.bias.copy_(torch.tensor(exported["bias"], device=device))
        print(f"[compact-counterfactual] initial={args.initial}")
        if args.anchor_weight > 0:
            anchor_model = copy.deepcopy(model).eval()
            for parameter in anchor_model.parameters():
                parameter.requires_grad_(False)
    elif args.anchor_weight > 0:
        raise ValueError("--anchor-weight requires --initial")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=5e-5)
    best = None
    best_metrics: dict[str, float] = {}
    best_regret = float("inf")
    best_epoch: int | None = None
    print(
        f"[compact-counterfactual] executable={executable} gpu={torch.cuda.get_device_name(0)} "
        f"train={len(train_records)} validation={len(validation_records)}"
    )
    if args.initial:
        initial_metrics = evaluate(model, validation_loader, device)
        best_regret = initial_metrics["regret"]
        best_metrics = initial_metrics
        best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        best_epoch = 0
        print(f"[compact-counterfactual] epoch=000 metrics={initial_metrics}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for features, costs, masks in train_loader:
            features, costs, masks = features.to(device), costs.to(device), masks.to(device)
            targets = torch.softmax(-costs.masked_fill(~masks, 1e9) / args.temperature, dim=1)
            logits = model(features).masked_fill(~masks, -1e9)
            log_probs = torch.log_softmax(logits, dim=1).masked_fill(~masks, 0)
            ranking_loss = -(targets * log_probs).sum(1).mean()
            counts = masks.sum(1, keepdim=True).clamp_min(1)
            target_mean = costs.masked_fill(~masks, 0).sum(1, keepdim=True) / counts
            logit_mean = logits.masked_fill(~masks, 0).sum(1, keepdim=True) / counts
            target_values = -(costs - target_mean) / 5.0
            predicted_values = logits - logit_mean
            regression_loss = torch.nn.functional.smooth_l1_loss(predicted_values[masks], target_values[masks])
            pairwise_loss = pairwise_ranking_loss(logits, costs, masks, args.pairwise_temperature)
            best_set_loss = optimal_set_loss(logits, costs, masks)
            if anchor_model is not None:
                with torch.no_grad():
                    anchor_logits = anchor_model(features).masked_fill(~masks, -1e9)
                    anchor_probs = torch.softmax(anchor_logits, dim=1)
                anchor_loss = -(anchor_probs * torch.log_softmax(logits, dim=1).masked_fill(~masks, 0)).sum(1).mean()
            else:
                anchor_loss = logits.new_zeros(())
            listwise_weight = 1.0 - auxiliary_weight
            loss = (
                ranking_loss * listwise_weight
                + regression_loss * args.regression_weight
                + pairwise_loss * args.pairwise_weight
                + best_set_loss * args.optimal_weight
                + anchor_loss * args.anchor_weight
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        metrics = evaluate(model, validation_loader, device)
        if metrics["regret"] < best_regret:
            best_regret = metrics["regret"]
            best_metrics = metrics
            best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[compact-counterfactual] epoch={epoch:03d} top1={metrics['top1']:.3%} "
                f"top2={metrics['top2']:.3%} regret={metrics['regret']:.4f}"
            )
    assert best is not None
    model.load_state_dict(best)
    manifest = {
        "format": "ntw-contextual-opponent-v2",
        "strategy": "counterfactual_rollout",
        "featureSize": FEATURE_SIZE_V2,
        "activation": "silu",
        "widths": list(widths),
        "temperature": args.temperature,
        "regressionWeight": args.regression_weight,
        "pairwiseWeight": args.pairwise_weight,
        "pairwiseTemperature": args.pairwise_temperature,
        "optimalWeight": args.optimal_weight,
        "anchorWeight": args.anchor_weight,
        "selectionEpoch": best_epoch,
        "layers": [{"weight": compact(layer.weight), "bias": compact(layer.bias)} for layer in model.layers],
        "validation": best_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"[compact-counterfactual] wrote={args.output} metrics={best_metrics}")


if __name__ == "__main__":
    main()
