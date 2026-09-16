"""Evaluate a Policy-v2 checkpoint against held-out counterfactual costs."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pretrain_policy_v2 import evaluate, prepare, read_games, tensorize
from vectorized_ppo import PolicyValueNet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    executable = Path(__import__("sys").executable).resolve()
    if executable.parent.name.lower() != "ntw-ai" or not torch.cuda.is_available():
        raise RuntimeError(f"evaluation requires ntw-ai CUDA, got {executable}")
    device = torch.device("cuda")
    prepared = prepare(read_games(args.data, None))
    loader = DataLoader(tensorize(prepared), batch_size=args.batch_size, pin_memory=True)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = PolicyValueNet().to(device)
    model.load_state_dict(payload["state_dict"])
    print(f"states={len(prepared)} metrics={evaluate(model, loader, device, 1.0)}")


if __name__ == "__main__":
    main()
