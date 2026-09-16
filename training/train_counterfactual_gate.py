"""Train a public-state gate between two counterfactual rollout policies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pretrain_policy_v2 import encode_state
from train_compact_counterfactual import read_records, tensorize
from train_contextual_opponent_proxy import ContextualProxy, FEATURE_SIZE_V2, compact
from vectorized_ppo import STATE_SIZE

EXTRA_SIZE = 6
GATE_SIZE = STATE_SIZE + EXTRA_SIZE


class Gate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList((nn.Linear(GATE_SIZE, 128), nn.Linear(128, 64), nn.Linear(64, 1)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        value = features
        for layer in self.layers[:-1]:
            value = torch.nn.functional.silu(layer(value))
        return self.layers[-1](value).squeeze(1)


def load_compact(path: Path, device: torch.device) -> tuple[ContextualProxy, dict]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    model = ContextualProxy(FEATURE_SIZE_V2, tuple(manifest["widths"])).to(device)
    with torch.no_grad():
        for layer, exported in zip(model.layers, manifest["layers"], strict=True):
            layer.weight.copy_(torch.tensor(exported["weight"], device=device))
            layer.bias.copy_(torch.tensor(exported["bias"], device=device))
    return model.eval(), manifest


@torch.inference_mode()
def build_gate_dataset(records: list[dict], new_model: ContextualProxy, old_model: ContextualProxy, device: torch.device) -> TensorDataset:
    action_features, costs, masks = tensorize(records).tensors
    states = torch.tensor([
        encode_state(record, record.get("playedCards") or [[], [], [], [], []])
        for record in records
    ], dtype=torch.float32)
    output_features = torch.zeros((len(records), GATE_SIZE), dtype=torch.float32)
    advantages = torch.zeros(len(records), dtype=torch.float32)
    regrets = torch.zeros((len(records), 4), dtype=torch.float32)
    loader = DataLoader(TensorDataset(action_features, costs, masks, states), batch_size=2048)
    offset = 0
    for actions, batch_costs, batch_masks, batch_states in loader:
        size = actions.shape[0]
        actions, batch_costs, batch_masks = actions.to(device), batch_costs.to(device), batch_masks.to(device)
        new_logits = new_model(actions).masked_fill(~batch_masks, -1e9)
        old_logits = old_model(actions).masked_fill(~batch_masks, -1e9)
        new_order = new_logits.argsort(1, descending=True)
        old_order = old_logits.argsort(1, descending=True)
        new_choice = new_order[:, 0]
        old_choice = old_order[:, 0]
        new_cost = batch_costs.gather(1, new_choice[:, None]).squeeze(1)
        old_cost = batch_costs.gather(1, old_choice[:, None]).squeeze(1)
        best_cost = batch_costs.masked_fill(~batch_masks, 1e9).min(1).values
        new_margin = new_logits.gather(1, new_order[:, :1]).squeeze(1) - new_logits.gather(1, new_order[:, 1:2]).squeeze(1)
        old_margin = old_logits.gather(1, old_order[:, :1]).squeeze(1) - old_logits.gather(1, old_order[:, 1:2]).squeeze(1)
        cards = actions[:, :, 0] * 54
        new_card = cards.gather(1, new_choice[:, None]).squeeze(1)
        old_card = cards.gather(1, old_choice[:, None]).squeeze(1)
        extra = torch.stack((
            new_card / 54,
            old_card / 54,
            (new_margin / 5).clamp(-2, 2),
            (old_margin / 5).clamp(-2, 2),
            (new_choice == old_choice).float(),
            (new_card - old_card) / 54,
        ), dim=1).cpu()
        output_features[offset:offset + size] = torch.cat((batch_states, extra), dim=1)
        advantages[offset:offset + size] = ((old_cost - new_cost) / 5).cpu()
        regrets[offset:offset + size] = torch.stack((new_cost, old_cost, best_cost, batch_masks.sum(1).float()), dim=1).cpu()
        offset += size
    return TensorDataset(output_features, advantages, regrets)


@torch.inference_mode()
def evaluate(model: Gate, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    selected_regret = new_regret = old_regret = oracle_regret = 0.0
    correct = count = 0
    for features, advantages, regrets in loader:
        features, advantages, regrets = features.to(device), advantages.to(device), regrets.to(device)
        prediction = model(features)
        use_new = prediction > 0
        new_cost, old_cost, best_cost, choices = regrets.unbind(1)
        decision = choices > 1
        selected = torch.where(use_new, new_cost, old_cost)
        selected_regret += ((selected - best_cost) * decision).sum().item()
        new_regret += ((new_cost - best_cost) * decision).sum().item()
        old_regret += ((old_cost - best_cost) * decision).sum().item()
        oracle_regret += ((torch.minimum(new_cost, old_cost) - best_cost) * decision).sum().item()
        correct += (((use_new == (advantages > 0)) | advantages.abs().lt(1e-7)) & decision).sum().item()
        count += decision.sum().item()
    count = max(1, count)
    return {
        "regret": selected_regret / count,
        "newRegret": new_regret / count,
        "oldRegret": old_regret / count,
        "oracleRegret": oracle_regret / count,
        "accuracy": correct / count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--validation", type=Path, nargs="+", required=True)
    parser.add_argument("--new-model", type=Path, required=True)
    parser.add_argument("--old-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--seed", type=int, default=2026082640)
    args = parser.parse_args()
    executable = Path(sys.executable).resolve()
    if executable.parent.name.lower() != "ntw-ai" or not torch.cuda.is_available():
        raise RuntimeError(f"gate training requires ntw-ai CUDA, got {executable}")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    new_model, _ = load_compact(args.new_model, device)
    old_model, _ = load_compact(args.old_model, device)
    train_records = read_records(args.train)
    validation_records = read_records(args.validation)
    train_data = build_gate_dataset(train_records, new_model, old_model, device)
    validation_data = build_gate_dataset(validation_records, new_model, old_model, device)
    train_loader = DataLoader(train_data, batch_size=1024, shuffle=True, pin_memory=True)
    validation_loader = DataLoader(validation_data, batch_size=2048, pin_memory=True)
    model = Gate().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=5e-5)
    best = None
    best_metrics = {}
    best_regret = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for features, advantages, _regrets in train_loader:
            features, advantages = features.to(device), advantages.to(device)
            prediction = model(features)
            regression = torch.nn.functional.smooth_l1_loss(prediction, advantages)
            target = (advantages > 0).float()
            weights = (advantages.abs() * 5).clamp(0.1, 3)
            classification = (torch.nn.functional.binary_cross_entropy_with_logits(prediction * 3, target, reduction="none") * weights).sum() / weights.sum()
            loss = regression * 0.6 + classification * 0.4
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        metrics = evaluate(model, validation_loader, device)
        if metrics["regret"] < best_regret:
            best_regret = metrics["regret"]
            best_metrics = metrics
            best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(f"[counterfactual-gate] epoch={epoch:03d} metrics={metrics}")
    assert best is not None
    model.load_state_dict(best)
    manifest = {
        "format": "ntw-counterfactual-gated",
        "modelPaths": [args.new_model.name, args.old_model.name],
        "weights": [1, 1],
        "stateSize": STATE_SIZE,
        "extraSize": EXTRA_SIZE,
        "activation": "silu",
        "gateLayers": [{"weight": compact(layer.weight), "bias": compact(layer.bias)} for layer in model.layers],
        "validation": best_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"[counterfactual-gate] wrote={args.output} metrics={best_metrics}")


if __name__ == "__main__":
    main()
