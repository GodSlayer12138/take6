from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from vectorized_ppo import FEATURE_VERSION, FORMAT, PolicyValueNet, export_model


def load_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != FORMAT or payload.get("featureVersion") != FEATURE_VERSION:
        raise ValueError(f"incompatible policy checkpoint: {path}")
    return payload["state_dict"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Interpolate compatible Policy v2 checkpoints.")
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--right-weight", type=float, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    executable = Path(sys.executable).resolve()
    if executable.parent.name.lower() != "ntw-ai":
        raise RuntimeError(f"refusing to export outside ntw-ai: {executable}")
    if not 0.0 <= args.right_weight <= 1.0:
        raise ValueError("--right-weight must be in [0, 1]")

    left = load_state(args.left)
    right = load_state(args.right)
    if left.keys() != right.keys():
        raise ValueError("checkpoint parameter sets differ")

    alpha = args.right_weight
    state = {key: left[key].lerp(right[key], alpha) for key in left}
    model = PolicyValueNet()
    model.load_state_dict(state)
    metrics = {
        "interpolationRightWeight": alpha,
        "left": str(args.left),
        "right": str(args.right),
    }
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": state,
        "optimizer": None,
        "format": FORMAT,
        "featureVersion": FEATURE_VERSION,
        "update": 0,
        "metrics": metrics,
        "training": "checkpoint-interpolation",
    }, args.checkpoint)
    export_model(model, args.output, 0, metrics)
    print(f"[interpolate-v2] executable={executable} right_weight={alpha:.3f} output={args.output}")


if __name__ == "__main__":
    main()
