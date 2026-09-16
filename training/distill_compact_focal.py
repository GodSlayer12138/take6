"""Distill the full public-history Policy-v2 into a fast contextual rollout policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from pretrain_policy_v2 import encode_action, encode_state
from train_contextual_opponent_proxy import ContextualProxy, FEATURE_SIZE_V2, compact, encode_contextual_v2
from vectorized_ppo import ACTION_SIZE, HAND_SIZE, PolicyValueNet, STATE_SIZE


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
    count = len(records)
    states = torch.zeros((count, STATE_SIZE), dtype=torch.float32)
    actions = torch.zeros((count, HAND_SIZE, ACTION_SIZE), dtype=torch.float32)
    compact_features = torch.zeros((count, HAND_SIZE, FEATURE_SIZE_V2), dtype=torch.float32)
    masks = torch.zeros((count, HAND_SIZE), dtype=torch.bool)
    for index, record in enumerate(records):
        histories = record.get("playedCards") or [[], [], [], [], []]
        states[index] = torch.tensor(encode_state(record, histories))
        for candidate, card in enumerate(record["hand"]):
            actions[index, candidate] = torch.tensor(encode_action(record, int(card)))
            compact_features[index, candidate] = torch.tensor(encode_contextual_v2(record, int(card)))
            masks[index, candidate] = True
    return TensorDataset(states, actions, compact_features, masks)


@torch.inference_mode()
def evaluate(teacher: PolicyValueNet, student: ContextualProxy, loader: DataLoader, device: torch.device, temperature: float) -> dict[str, float]:
    teacher.eval()
    student.eval()
    top1 = top2 = count = 0
    kl_sum = 0.0
    for states, actions, features, masks in loader:
        states, actions, features, masks = states.to(device), actions.to(device), features.to(device), masks.to(device)
        teacher_logits, _ = teacher(states, actions)
        teacher_logits = teacher_logits.masked_fill(~masks, -1e9)
        student_logits = student(features).masked_fill(~masks, -1e9)
        targets = torch.softmax(teacher_logits / temperature, dim=1)
        log_probs = torch.log_softmax(student_logits / temperature, dim=1)
        teacher_choice = teacher_logits.argmax(1)
        top1 += (student_logits.argmax(1) == teacher_choice).sum().item()
        top2 += (student_logits.topk(2, 1).indices == teacher_choice[:, None]).any(1).sum().item()
        kl_sum += (-(targets * log_probs.masked_fill(~masks, 0)).sum(1)).sum().item()
        count += states.shape[0]
    return {"top1": top1 / count, "top2": top2 / count, "crossEntropy": kl_sum / count}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--validation", type=Path, nargs="+", required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--hard-label", action="store_true")
    parser.add_argument("--seed", type=int, default=2026082606)
    args = parser.parse_args()
    executable = Path(sys.executable).resolve()
    if executable.parent.name.lower() != "ntw-ai" or not torch.cuda.is_available():
        raise RuntimeError(f"compact focal distillation requires ntw-ai CUDA, got {executable}")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    train_records = read_records(args.train)
    validation_records = read_records(args.validation)
    print(f"[compact-focal] executable={executable} gpu={torch.cuda.get_device_name(0)} train={len(train_records)} validation={len(validation_records)}")
    train_loader = DataLoader(tensorize(train_records), batch_size=512, shuffle=True, pin_memory=True)
    validation_loader = DataLoader(tensorize(validation_records), batch_size=1024, pin_memory=True)
    payload = torch.load(args.teacher, map_location="cpu", weights_only=False)
    teacher = PolicyValueNet().to(device)
    teacher.load_state_dict(payload["state_dict"])
    teacher.eval()
    student = ContextualProxy(FEATURE_SIZE_V2, (192, 96)).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate, weight_decay=3e-5)
    best = None
    best_selection = float("inf")
    best_metrics: dict[str, float] = {}
    for epoch in range(1, args.epochs + 1):
        student.train()
        for states, actions, features, masks in train_loader:
            states, actions, features, masks = states.to(device), actions.to(device), features.to(device), masks.to(device)
            with torch.no_grad():
                teacher_logits, _ = teacher(states, actions)
                masked_teacher = teacher_logits.masked_fill(~masks, -1e9)
                targets = torch.softmax(masked_teacher / args.temperature, dim=1)
            student_logits = student(features).masked_fill(~masks, -1e9)
            if args.hard_label:
                loss = torch.nn.functional.cross_entropy(student_logits, masked_teacher.argmax(1))
            else:
                log_probs = torch.log_softmax(student_logits / args.temperature, dim=1)
                loss = -(targets * log_probs.masked_fill(~masks, 0)).sum(1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        metrics = evaluate(teacher, student, validation_loader, device, args.temperature)
        selection = -metrics["top1"] if args.hard_label else metrics["crossEntropy"]
        if selection < best_selection:
            best_selection = selection
            best_metrics = metrics
            best = {key: value.detach().cpu().clone() for key, value in student.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(f"[compact-focal] epoch={epoch:03d} top1={metrics['top1']:.3%} top2={metrics['top2']:.3%} crossEntropy={metrics['crossEntropy']:.4f}")
    assert best is not None
    student.load_state_dict(best)
    manifest = {
        "format": "ntw-contextual-opponent-v2",
        "strategy": "policy_v2_distilled_rollout",
        "featureSize": FEATURE_SIZE_V2,
        "activation": "silu",
        "widths": [192, 96],
        "distillation": "hard" if args.hard_label else f"soft-t{args.temperature:g}",
        "layers": [{"weight": compact(layer.weight), "bias": compact(layer.bias)} for layer in student.layers],
        "validation": best_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"[compact-focal] wrote={args.output} metrics={best_metrics}")


if __name__ == "__main__":
    main()
