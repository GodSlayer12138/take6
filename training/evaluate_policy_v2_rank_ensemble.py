"""Evaluate rank blends of two Policy-v2 checkpoints on counterfactual costs."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pretrain_policy_v2 import prepare, read_games, tensorize
from vectorized_ppo import PolicyValueNet


def load_model(path: Path, device: torch.device) -> PolicyValueNet:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = PolicyValueNet().to(device)
    model.load_state_dict(payload["state_dict"])
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, required=True)
    parser.add_argument("--weights", type=int, nargs="+", default=(0, 25, 50, 75, 100))
    args = parser.parse_args()
    device = torch.device("cuda")
    prepared = prepare(read_games(args.data, None))
    loader = DataLoader(tensorize(prepared), batch_size=1024, pin_memory=True)
    primary = load_model(args.primary, device)
    secondary = load_model(args.secondary, device)
    totals = {weight: {"count": 0, "regret": 0.0, "c3": 0.0, "c4": 0.0, "optimal3": 0} for weight in args.weights}
    with torch.inference_mode():
        for states, actions, costs, masks in loader:
            states, actions, costs, masks = states.to(device), actions.to(device), costs.to(device), masks.to(device)
            primary_logits, _ = primary(states, actions)
            secondary_logits, _ = secondary(states, actions)
            primary_logits = primary_logits.masked_fill(~masks, -1e9)
            secondary_logits = secondary_logits.masked_fill(~masks, -1e9)
            primary_ranks = torch.argsort(torch.argsort(-primary_logits, dim=1), dim=1).float()
            secondary_ranks = torch.argsort(torch.argsort(-secondary_logits, dim=1), dim=1).float()
            legal_costs = costs.masked_fill(~masks, 1e9)
            best = legal_costs.min(1).values
            decisions = masks.sum(1) > 1
            for weight in args.weights:
                fraction = weight / 100
                blended = -(primary_ranks * (1 - fraction) + secondary_ranks * fraction).masked_fill(~masks, 1e9)
                selected = legal_costs.gather(1, blended.argmax(1, keepdim=True)).squeeze(1)
                candidate3 = legal_costs.gather(1, blended.topk(3, 1).indices).min(1).values
                candidate4 = legal_costs.gather(1, blended.topk(4, 1).indices).min(1).values
                entry = totals[weight]
                entry["count"] += decisions.sum().item()
                entry["regret"] += ((selected - best) * decisions).sum().item()
                entry["c3"] += ((candidate3 - best) * decisions).sum().item()
                entry["c4"] += ((candidate4 - best) * decisions).sum().item()
                entry["optimal3"] += ((candidate3 - best).abs().lt(1e-6) & decisions).sum().item()
    for weight, entry in totals.items():
        count = max(1, entry["count"])
        print(
            f"secondaryWeight={weight:3d} regret={entry['regret']/count:.4f} "
            f"c3={entry['c3']/count:.4f} c4={entry['c4']/count:.4f} "
            f"optimal3={entry['optimal3']/count:.2%}"
        )


if __name__ == "__main__":
    main()
