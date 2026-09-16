"""Train a full-public-history action proxy for a known arena opponent."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from distill_policy_v2_compact import CompactPolicyV2, compact
from pretrain_policy_v2 import encode_action, encode_state
from vectorized_ppo import ACTION_SIZE, CARDS, HAND_SIZE, STATE_SIZE


def read_records(paths: list[Path], strategy: str) -> list[dict]:
    records: list[dict] = []
    for source_index, path in enumerate(paths):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("strategy") == strategy and record.get("deckMode") == "adaptive":
                    record["_sourceIndex"] = source_index
                    records.append(record)
    return records


def tensorize(records: list[dict]) -> TensorDataset:
    states = torch.zeros((len(records), STATE_SIZE), dtype=torch.float32)
    actions = torch.zeros((len(records), HAND_SIZE, ACTION_SIZE), dtype=torch.float32)
    masks = torch.zeros((len(records), HAND_SIZE), dtype=torch.bool)
    chosen = torch.zeros(len(records), dtype=torch.long)
    for row, record in enumerate(records):
        histories = record.get("playedCards") or [[], [], [], [], []]
        states[row] = torch.tensor(encode_state(record, histories))
        hand = list(record["hand"])
        for column, card in enumerate(hand):
            actions[row, column] = torch.tensor(encode_action(record, int(card)))
            masks[row, column] = True
        chosen[row] = hand.index(int(record["chosenCard"]))
    return TensorDataset(states, actions, masks, chosen)


@torch.inference_mode()
def evaluate(model: CompactPolicyV2, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    top1 = top2 = count = 0
    nll = 0.0
    temperature_nll = {value: 0.0 for value in (0.5, 0.7, 1.0, 1.3, 1.6, 2.0)}
    for states, actions, masks, chosen in loader:
        states, actions, masks, chosen = states.to(device), actions.to(device), masks.to(device), chosen.to(device)
        logits = model(states, actions).masked_fill(~masks, -1e9)
        top1 += (logits.argmax(1) == chosen).sum().item()
        top2 += (logits.topk(2, 1).indices == chosen[:, None]).any(1).sum().item()
        nll += torch.nn.functional.cross_entropy(logits, chosen, reduction="sum").item()
        for temperature in temperature_nll:
            temperature_nll[temperature] += torch.nn.functional.cross_entropy(logits / temperature, chosen, reduction="sum").item()
        count += chosen.numel()
    calibrated_temperature, calibrated_sum = min(temperature_nll.items(), key=lambda item: item[1])
    return {
        "top1": top1 / count,
        "top2": top2 / count,
        "nll": nll / count,
        "temperature": calibrated_temperature,
        "calibratedNll": calibrated_sum / count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--strategy", choices=("champion", "external_mcs"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--state-widths", type=int, nargs=2, default=(192, 128))
    parser.add_argument("--action-widths", type=int, nargs=2, default=(96, 32))
    parser.add_argument("--seed", type=int, default=2026082609)
    args = parser.parse_args()

    executable = Path(sys.executable).resolve()
    if executable.parent.name.lower() != "ntw-ai" or not torch.cuda.is_available():
        raise RuntimeError(f"structured opponent training requires ntw-ai CUDA, got {executable}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    records = read_records(args.data, args.strategy)
    validation = [record for record in records if (int(record["_sourceIndex"]), int(record["gameIndex"]))[1] % 5 == 0]
    train = [record for record in records if (int(record["_sourceIndex"]), int(record["gameIndex"]))[1] % 5 != 0]
    device = torch.device("cuda")
    train_loader = DataLoader(tensorize(train), batch_size=512, shuffle=True, pin_memory=True)
    validation_loader = DataLoader(tensorize(validation), batch_size=1024, pin_memory=True)
    state_widths = tuple(args.state_widths)
    action_widths = tuple(args.action_widths)
    model = CompactPolicyV2(state_widths, action_widths).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=5e-5)
    best = None
    best_metrics: dict[str, float] = {}
    best_nll = float("inf")
    print(
        f"[structured-opponent] executable={executable} gpu={torch.cuda.get_device_name(0)} "
        f"strategy={args.strategy} train={len(train)} validation={len(validation)}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        for states, actions, masks, chosen in train_loader:
            states, actions, masks, chosen = states.to(device), actions.to(device), masks.to(device), chosen.to(device)
            logits = model(states, actions).masked_fill(~masks, -1e9)
            loss = torch.nn.functional.cross_entropy(logits, chosen, label_smoothing=0.01)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        metrics = evaluate(model, validation_loader, device)
        if metrics["nll"] < best_nll:
            best_nll = metrics["nll"]
            best_metrics = metrics
            best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[structured-opponent] epoch={epoch:03d} top1={metrics['top1']:.3%} "
                f"top2={metrics['top2']:.3%} nll={metrics['nll']:.4f} "
                f"temperature={metrics['temperature']:.1f}"
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
        "strategy": args.strategy,
        "stateWidths": list(state_widths),
        "actionWidths": list(action_widths),
        "stateLayers": [{"weight": compact(layer.weight), "bias": compact(layer.bias)} for layer in model.state_layers],
        "actionLayers": [{"weight": compact(layer.weight), "bias": compact(layer.bias)} for layer in model.action_layers],
        "validation": best_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"[structured-opponent] wrote={args.output} metrics={best_metrics}")


if __name__ == "__main__":
    main()
