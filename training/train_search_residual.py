"""Learn a public-information residual correction for low-budget search."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from counterfactual_loss import pairwise_ranking_loss
from pretrain_policy_v2 import encode_action, encode_state
from train_contextual_opponent_proxy import compact
from vectorized_ppo import ACTION_SIZE, HAND_SIZE, STATE_SIZE


DYNAMIC_SIZE = 10
BASELINE_SCALE = 5.0


def read_records(paths: list[Path]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if "meta" not in record:
                    records.append(record)
    return records


def dynamic_features(entry: dict, entries: list[dict]) -> list[float]:
    count = len(entries)
    utility = float(entry["utility"])
    search_utility = float(entry.get("searchUtility", utility))
    neural_utility = float(entry.get("neuralUtility", 0.0))
    mean_utility = sum(float(value["utility"]) for value in entries) / count
    mean_search = sum(float(value.get("searchUtility", value["utility"])) for value in entries) / count
    mean_neural = sum(float(value.get("neuralUtility", 0.0)) for value in entries) / count
    clip = lambda value: max(-3.0, min(3.0, value))
    return [
        clip((utility - mean_utility) / 5.0),
        clip((search_utility - mean_search) / 5.0),
        clip((neural_utility - mean_neural) / 5.0),
        min(2.0, float(entry.get("expectedPenalty", 0.0)) / 15.0),
        min(2.0, float(entry.get("immediatePenalty", 0.0)) / 15.0),
        min(3.0, float(entry.get("risk", 0.0)) / 10.0),
        clip((float(entry.get("cvar", 0.0)) - float(entry.get("expectedPenalty", 0.0))) / 10.0),
        max(0.0, min(1.0, float(entry.get("disasterRate", 0.0)))),
        float(entry.get("rank", 0)) / max(1, count - 1),
        min(2.0, float(entry.get("samples", 18)) / 18.0),
    ]


def tensorize(records: list[dict]) -> TensorDataset:
    count = len(records)
    states = torch.zeros((count, STATE_SIZE), dtype=torch.float32)
    actions = torch.zeros((count, HAND_SIZE, ACTION_SIZE), dtype=torch.float32)
    dynamics = torch.zeros((count, HAND_SIZE, DYNAMIC_SIZE), dtype=torch.float32)
    baseline = torch.zeros((count, HAND_SIZE), dtype=torch.float32)
    costs = torch.zeros((count, HAND_SIZE), dtype=torch.float32)
    masks = torch.zeros((count, HAND_SIZE), dtype=torch.bool)
    for row, record in enumerate(records):
        played = record.get("playedCards") or [[], [], [], [], []]
        states[row] = torch.tensor(encode_state(record, played))
        labels = {int(card): float(cost) for card, cost in record["labels"]}
        entries = record["searchCandidates"]
        mean_utility = sum(float(entry["utility"]) for entry in entries) / len(entries)
        for column, entry in enumerate(entries):
            card = int(entry["card"])
            actions[row, column] = torch.tensor(encode_action(record, card))
            dynamics[row, column] = torch.tensor(dynamic_features(entry, entries))
            baseline[row, column] = -(float(entry["utility"]) - mean_utility) / BASELINE_SCALE
            costs[row, column] = labels[card]
            masks[row, column] = True
    return TensorDataset(states, actions, dynamics, baseline, costs, masks)


class SearchResidual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.state_layers = nn.ModuleList((nn.Linear(STATE_SIZE, 192), nn.Linear(192, 128)))
        self.delta_layers = nn.ModuleList(
            (nn.Linear(128 + ACTION_SIZE + DYNAMIC_SIZE, 96), nn.Linear(96, 48), nn.Linear(48, 1))
        )
        nn.init.zeros_(self.delta_layers[-1].weight)
        nn.init.zeros_(self.delta_layers[-1].bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor, dynamic: torch.Tensor) -> torch.Tensor:
        context = state
        for layer in self.state_layers:
            context = torch.nn.functional.silu(layer(context))
        context = context[:, None, :].expand(-1, action.shape[1], -1)
        value = torch.cat((context, action, dynamic), dim=2)
        for layer in self.delta_layers[:-1]:
            value = torch.nn.functional.silu(layer(value))
        return self.delta_layers[-1](value).squeeze(2)


@torch.inference_mode()
def evaluate(model: SearchResidual, loader: DataLoader, device: torch.device, alphas: tuple[float, ...]) -> dict:
    model.eval()
    totals = {alpha: {"regret": 0.0, "exact": 0} for alpha in alphas}
    baseline_regret = oracle_regret = 0.0
    decisions = 0
    for state, action, dynamic, baseline, costs, mask in loader:
        state, action, dynamic = state.to(device), action.to(device), dynamic.to(device)
        baseline, costs, mask = baseline.to(device), costs.to(device), mask.to(device)
        delta = model(state, action, dynamic).masked_fill(~mask, 0)
        best_index = costs.masked_fill(~mask, 1e9).argmin(1)
        best_cost = costs.gather(1, best_index[:, None]).squeeze(1)
        decision = mask.sum(1) > 1
        baseline_index = baseline.masked_fill(~mask, -1e9).argmax(1)
        baseline_cost = costs.gather(1, baseline_index[:, None]).squeeze(1)
        baseline_regret += ((baseline_cost - best_cost) * decision).sum().item()
        oracle_regret += ((costs.masked_fill(~mask, 1e9).min(1).values - best_cost) * decision).sum().item()
        for alpha in alphas:
            corrected = (baseline + delta * alpha).masked_fill(~mask, -1e9)
            selected = corrected.argmax(1)
            selected_cost = costs.gather(1, selected[:, None]).squeeze(1)
            totals[alpha]["regret"] += ((selected_cost - best_cost) * decision).sum().item()
            totals[alpha]["exact"] += ((selected == best_index) & decision).sum().item()
        decisions += decision.sum().item()
    count = max(1, decisions)
    by_alpha = {
        str(alpha): {"regret": values["regret"] / count, "exact": values["exact"] / count}
        for alpha, values in totals.items()
    }
    best_alpha = min(alphas, key=lambda alpha: (by_alpha[str(alpha)]["regret"], -by_alpha[str(alpha)]["exact"]))
    return {
        "baselineRegret": baseline_regret / count,
        "oracleRegret": oracle_regret / count,
        "bestAlpha": best_alpha,
        "bestRegret": by_alpha[str(best_alpha)]["regret"],
        "bestExact": by_alpha[str(best_alpha)]["exact"],
        "byAlpha": by_alpha,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--validation", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026082662)
    args = parser.parse_args()

    executable = Path(sys.executable).resolve()
    if executable.parent.name.lower() != "ntw-ai" or not torch.cuda.is_available():
        raise RuntimeError(f"search residual training requires ntw-ai CUDA, got {executable}")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    train_records = read_records(args.train)
    validation_records = read_records(args.validation)
    train_loader = DataLoader(tensorize(train_records), batch_size=args.batch_size, shuffle=True, pin_memory=True)
    validation_loader = DataLoader(tensorize(validation_records), batch_size=args.batch_size * 2, pin_memory=True)
    model = SearchResidual().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    alphas = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
    best_state = None
    best_metrics = None
    best_regret = math.inf
    print(
        f"[search-residual] executable={executable} gpu={torch.cuda.get_device_name(0)} "
        f"train={len(train_records)} validation={len(validation_records)}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        for state, action, dynamic, baseline, costs, mask in train_loader:
            state, action, dynamic = state.to(device), action.to(device), dynamic.to(device)
            baseline, costs, mask = baseline.to(device), costs.to(device), mask.to(device)
            delta = model(state, action, dynamic)
            corrected = (baseline + delta).masked_fill(~mask, -1e9)
            target_probs = torch.softmax((-costs / args.temperature).masked_fill(~mask, -1e9), dim=1)
            listwise = -(target_probs * torch.log_softmax(corrected, dim=1).masked_fill(~mask, 0)).sum(1).mean()
            expert = costs.masked_fill(~mask, 1e9).argmin(1)
            hard = torch.nn.functional.cross_entropy(corrected, expert)
            counts = mask.sum(1, keepdim=True).clamp_min(1)
            target_mean = costs.masked_fill(~mask, 0).sum(1, keepdim=True) / counts
            target_values = -(costs - target_mean) / BASELINE_SCALE
            corrected_mean = corrected.masked_fill(~mask, 0).sum(1, keepdim=True) / counts
            regression = torch.nn.functional.smooth_l1_loss((corrected - corrected_mean)[mask], target_values[mask])
            pairwise = pairwise_ranking_loss(corrected, costs, mask, 1.0)
            residual_penalty = delta[mask].square().mean()
            loss = listwise * 0.44 + hard * 0.16 + regression * 0.18 + pairwise * 0.20 + residual_penalty * 0.02
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        metrics = evaluate(model, validation_loader, device, alphas)
        if metrics["bestRegret"] < best_regret:
            best_regret = metrics["bestRegret"]
            best_metrics = metrics
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0:
            print(f"[search-residual] epoch={epoch:03d} metrics={metrics}")
    if best_state is None or best_metrics is None:
        raise RuntimeError("search residual produced no checkpoint")
    model.load_state_dict(best_state)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "ntw-search-residual-v1", "state_dict": best_state, "metrics": best_metrics}, args.checkpoint)
    manifest = {
        "format": "ntw-search-residual-v1",
        "stateSize": STATE_SIZE,
        "actionSize": ACTION_SIZE,
        "dynamicSize": DYNAMIC_SIZE,
        "baselineScale": BASELINE_SCALE,
        "blend": best_metrics["bestAlpha"],
        "activation": "silu",
        "stateLayers": [{"weight": compact(layer.weight), "bias": compact(layer.bias)} for layer in model.state_layers],
        "deltaLayers": [{"weight": compact(layer.weight), "bias": compact(layer.bias)} for layer in model.delta_layers],
        "validation": best_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"[search-residual] wrote={args.output} metrics={best_metrics}")


if __name__ == "__main__":
    main()
