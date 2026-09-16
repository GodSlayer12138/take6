"""Train a public-information gate between released search and C1 policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pretrain_policy_v2 import encode_state
from train_contextual_opponent_proxy import compact
from vectorized_ppo import STATE_SIZE


EXTRA_SIZE = 6
GATE_SIZE = STATE_SIZE + EXTRA_SIZE


class SearchGate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList((nn.Linear(GATE_SIZE, 128), nn.Linear(128, 64), nn.Linear(64, 1)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        value = features
        for layer in self.layers[:-1]:
            value = torch.nn.functional.silu(layer(value))
        return self.layers[-1](value).squeeze(1)


def read_records(paths: list[Path]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if "meta" not in record:
                    records.append(record)
    return records


def tensorize(records: list[dict]) -> TensorDataset:
    features = torch.zeros((len(records), GATE_SIZE), dtype=torch.float32)
    advantages = torch.zeros(len(records), dtype=torch.float32)
    outcomes = torch.zeros((len(records), 4), dtype=torch.float32)
    for index, record in enumerate(records):
        costs = {int(card): float(cost) for card, cost in record["labels"]}
        search_card = int(record["searchCard"])
        counterfactual_card = int(record["counterfactualCard"])
        search_cost = costs[search_card]
        counterfactual_cost = costs[counterfactual_card]
        best_cost = min(costs.values())
        state = encode_state(record, record.get("playedCards") or [[], [], [], [], []])
        extra = (
            search_card / 54,
            counterfactual_card / 54,
            max(-2.0, min(2.0, float(record["searchMargin"]) / 5)),
            max(-2.0, min(2.0, float(record["counterfactualMargin"]) / 5)),
            float(search_card == counterfactual_card),
            (counterfactual_card - search_card) / 54,
        )
        features[index] = torch.tensor([*state, *extra])
        advantages[index] = (search_cost - counterfactual_cost) / 5
        outcomes[index] = torch.tensor((search_cost, counterfactual_cost, best_cost, float(len(costs) > 1)))
    return TensorDataset(features, advantages, outcomes)


@torch.inference_mode()
def evaluate(model: SearchGate, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = {"gate": 0.0, "search": 0.0, "counterfactual": 0.0, "oracle": 0.0}
    use_counterfactual = correct = disagreements = count = 0
    for features, advantages, outcomes in loader:
        features, advantages, outcomes = features.to(device), advantages.to(device), outcomes.to(device)
        prediction = model(features)
        select_counterfactual = prediction > 0
        search_cost, counterfactual_cost, best_cost, decision = outcomes.unbind(1)
        selected = torch.where(select_counterfactual, counterfactual_cost, search_cost)
        totals["gate"] += ((selected - best_cost) * decision).sum().item()
        totals["search"] += ((search_cost - best_cost) * decision).sum().item()
        totals["counterfactual"] += ((counterfactual_cost - best_cost) * decision).sum().item()
        totals["oracle"] += ((torch.minimum(search_cost, counterfactual_cost) - best_cost) * decision).sum().item()
        use_counterfactual += (select_counterfactual & decision.bool()).sum().item()
        correct += (((select_counterfactual == (advantages > 0)) | advantages.abs().lt(1e-7)) & decision.bool()).sum().item()
        disagreements += ((features[:, STATE_SIZE + 4] < 0.5) & decision.bool()).sum().item()
        count += decision.sum().item()
    count = max(1, int(count))
    return {
        "regret": totals["gate"] / count,
        "searchRegret": totals["search"] / count,
        "counterfactualRegret": totals["counterfactual"] / count,
        "oracleRegret": totals["oracle"] / count,
        "accuracy": correct / count,
        "counterfactualRate": use_counterfactual / count,
        "disagreementRate": disagreements / count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--validation", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=2026082650)
    args = parser.parse_args()

    executable = Path(sys.executable).resolve()
    if executable.parent.name.lower() != "ntw-ai" or not torch.cuda.is_available():
        raise RuntimeError(f"search gate requires ntw-ai CUDA, got {executable}")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    train_records = read_records(args.train)
    validation_records = read_records(args.validation)
    train_loader = DataLoader(tensorize(train_records), batch_size=512, shuffle=True, pin_memory=True)
    validation_loader = DataLoader(tensorize(validation_records), batch_size=2048, pin_memory=True)
    model = SearchGate().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best_state = None
    best_metrics: dict[str, float] = {}
    best_regret = float("inf")
    print(f"[search-gate] executable={executable} gpu={torch.cuda.get_device_name(0)} train={len(train_records)} validation={len(validation_records)}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for features, advantages, _outcomes in train_loader:
            features, advantages = features.to(device), advantages.to(device)
            prediction = model(features)
            regression = torch.nn.functional.smooth_l1_loss(prediction, advantages)
            target = (advantages > 0).float()
            weights = (advantages.abs() * 5).clamp(0.1, 4)
            classification = (
                torch.nn.functional.binary_cross_entropy_with_logits(prediction * 3, target, reduction="none") * weights
            ).sum() / weights.sum()
            loss = regression * 0.55 + classification * 0.45
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        metrics = evaluate(model, validation_loader, device)
        if metrics["regret"] < best_regret:
            best_regret = metrics["regret"]
            best_metrics = metrics
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(f"[search-gate] epoch={epoch:03d} metrics={metrics}")
    assert best_state is not None
    model.load_state_dict(best_state)
    manifest = {
        "format": "ntw-search-counterfactual-gate-v1",
        "stateSize": STATE_SIZE,
        "extraSize": EXTRA_SIZE,
        "activation": "silu",
        "layers": [{"weight": compact(layer.weight), "bias": compact(layer.bias)} for layer in model.layers],
        "validation": best_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"[search-gate] wrote={args.output} metrics={best_metrics}")


if __name__ == "__main__":
    main()
