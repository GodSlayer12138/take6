"""Export a trusted rl-6-nimmt policy state_dict to browser JSON.

The upstream repository does not publish trained weights. This converter is
for a separately obtained PolicyMCSAgent / PUCTAgent actor.state_dict(). It
uses PyTorch's weights-only loader and deliberately refuses arbitrary pickle
objects.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def ordered_policy_weights(state: dict) -> list[tuple[str, object]]:
    """Return trunk layers followed by the scalar policy head.

    Lexicographic state_dict ordering is wrong here because ``head_nets`` sorts
    before ``latent_net`` even though it executes last.
    """
    latent: list[tuple[int, str, object]] = []
    head: list[tuple[int, str, object]] = []
    for key, value in state.items():
        if not key.endswith("weight") or getattr(value, "ndim", 0) != 2:
            continue
        latent_match = re.search(r"(?:^|\.)latent_net\.(\d+)\.weight$", key)
        head_match = re.search(r"(?:^|\.)head_nets\.0\.(\d+)\.weight$", key)
        if latent_match:
            latent.append((int(latent_match.group(1)), key, value))
        elif head_match:
            head.append((int(head_match.group(1)), key, value))
    if not latent or not head:
        raise ValueError("Expected MultiHeadedMLP latent_net and head_nets.0 weights")
    latent.sort(key=lambda item: item[0])
    head.sort(key=lambda item: item[0])
    return [(key, value) for _, key, value in [*latent, *head]]


def main() -> None:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Install PyTorch in the conversion environment first") from error

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="actor.state_dict() saved with torch.save")
    parser.add_argument("output", type=Path)
    parser.add_argument("--cards", type=int, default=104)
    parser.add_argument("--activation", choices=("relu", "tanh"), default="relu")
    args = parser.parse_args()

    try:
        state = torch.load(args.input, map_location="cpu", weights_only=True)
    except TypeError as error:
        raise RuntimeError("This converter requires a PyTorch version with weights_only=True") from error

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError("Expected a state_dict mapping, not a pickled Agent/Tournament object")

    weights = ordered_policy_weights(state)
    layers = []
    for weight_key, weight in weights:
        bias_key = weight_key[:-6] + "bias"
        if bias_key not in state:
            raise KeyError(f"Missing {bias_key}")
        layers.append({
            "source": weight_key,
            "weight": weight.detach().cpu().tolist(),
            "bias": state[bias_key].detach().cpu().tolist(),
        })

    if not layers or len(layers[-1]["bias"]) != 1:
        raise ValueError("Expected an action-conditioned policy ending in one scalar logit")
    if len(layers[0]["weight"][0]) != 48:
        raise ValueError("Expected 48 inputs: candidate action + 47-value upstream state")

    manifest = {
        "format": "rl-6-nimmt-browser-v1",
        "source": "johannbrehmer/rl-6-nimmt",
        "architecture": "action-conditioned-policy",
        "cards": args.cards,
        "stateLength": 47,
        "activation": args.activation,
        "layers": layers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Exported {len(layers)} layers to {args.output}")


if __name__ == "__main__":
    main()
