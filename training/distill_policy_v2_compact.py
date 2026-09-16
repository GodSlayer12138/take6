"""Distill Policy-v2 into a smaller network without dropping public history.

Unlike the contextual rollout proxy, this student keeps the complete 420
dimensional public state, including per-seat played-card masks.  The state
trunk and action head are narrower than the teacher, so one rollout decision
is substantially cheaper while remaining information-compatible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pretrain_policy_v2 import encode_action, encode_state
from vectorized_ppo import ACTION_SIZE, CARDS, HAND_SIZE, STATE_SIZE, PolicyValueNet


class CompactPolicyV2(nn.Module):
    def __init__(self, state_widths: tuple[int, int], action_widths: tuple[int, int]) -> None:
        super().__init__()
        self.state_layers = nn.ModuleList((
            nn.Linear(STATE_SIZE, state_widths[0]),
            nn.Linear(state_widths[0], state_widths[1]),
        ))
        self.action_layers = nn.ModuleList((
            nn.Linear(state_widths[1] + ACTION_SIZE, action_widths[0]),
            nn.Linear(action_widths[0], action_widths[1]),
            nn.Linear(action_widths[1], 1),
        ))

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        context = states
        for layer in self.state_layers:
            context = torch.nn.functional.silu(layer(context))
        expanded = context.unsqueeze(1).expand(-1, actions.shape[1], -1)
        values = torch.cat((expanded, actions), dim=-1)
        for layer in self.action_layers[:-1]:
            values = torch.nn.functional.silu(layer(values))
        return self.action_layers[-1](values).squeeze(-1)


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
    states = torch.zeros((len(records), STATE_SIZE), dtype=torch.float32)
    actions = torch.zeros((len(records), HAND_SIZE, ACTION_SIZE), dtype=torch.float32)
    masks = torch.zeros((len(records), HAND_SIZE), dtype=torch.bool)
    for row, record in enumerate(records):
        histories = record.get("playedCards") or [[], [], [], [], []]
        states[row] = torch.tensor(encode_state(record, histories))
        for column, card in enumerate(record["hand"]):
            actions[row, column] = torch.tensor(encode_action(record, int(card)))
            masks[row, column] = True
    return TensorDataset(states, actions, masks)


@torch.inference_mode()
def evaluate(
    teacher: PolicyValueNet,
    student: CompactPolicyV2,
    loader: DataLoader,
    device: torch.device,
    temperature: float,
) -> dict[str, float]:
    teacher.eval()
    student.eval()
    top1 = top2 = count = 0
    cross_entropy = 0.0
    for states, actions, masks in loader:
        states, actions, masks = states.to(device), actions.to(device), masks.to(device)
        teacher_logits, _ = teacher(states, actions)
        teacher_logits = teacher_logits.masked_fill(~masks, -1e9)
        student_logits = student(states, actions).masked_fill(~masks, -1e9)
        targets = torch.softmax(teacher_logits / temperature, dim=1)
        teacher_choice = teacher_logits.argmax(1)
        top1 += (student_logits.argmax(1) == teacher_choice).sum().item()
        top2 += (student_logits.topk(2, 1).indices == teacher_choice[:, None]).any(1).sum().item()
        cross_entropy += (-(targets * torch.log_softmax(student_logits / temperature, dim=1).masked_fill(~masks, 0)).sum(1)).sum().item()
        count += states.shape[0]
    return {
        "top1": top1 / count,
        "top2": top2 / count,
        "crossEntropy": cross_entropy / count,
    }


def compact(tensor: torch.Tensor) -> list:
    values = tensor.detach().cpu().tolist()
    if values and isinstance(values[0], list):
        return [[round(float(value), 7) for value in row] for row in values]
    return [round(float(value), 7) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--validation", type=Path, nargs="+", required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--state-widths", type=int, nargs=2, default=(192, 128))
    parser.add_argument("--action-widths", type=int, nargs=2, default=(96, 32))
    parser.add_argument("--seed", type=int, default=2026082608)
    args = parser.parse_args()

    executable = Path(sys.executable).resolve()
    if executable.parent.name.lower() != "ntw-ai" or not torch.cuda.is_available():
        raise RuntimeError(f"structured distillation requires ntw-ai CUDA, got {executable}")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    train_records = read_records(args.train)
    validation_records = read_records(args.validation)
    train_loader = DataLoader(tensorize(train_records), batch_size=512, shuffle=True, pin_memory=True)
    validation_loader = DataLoader(tensorize(validation_records), batch_size=1024, pin_memory=True)

    payload = torch.load(args.teacher, map_location="cpu", weights_only=False)
    teacher = PolicyValueNet().to(device)
    teacher.load_state_dict(payload["state_dict"])
    teacher.eval()
    state_widths = tuple(args.state_widths)
    action_widths = tuple(args.action_widths)
    student = CompactPolicyV2(state_widths, action_widths).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate, weight_decay=3e-5)
    best = None
    best_metrics: dict[str, float] = {}
    best_cross_entropy = float("inf")
    print(
        f"[structured-student] executable={executable} gpu={torch.cuda.get_device_name(0)} "
        f"train={len(train_records)} validation={len(validation_records)} "
        f"state={state_widths} action={action_widths}"
    )
    for epoch in range(1, args.epochs + 1):
        student.train()
        for states, actions, masks in train_loader:
            states, actions, masks = states.to(device), actions.to(device), masks.to(device)
            with torch.no_grad():
                teacher_logits, _ = teacher(states, actions)
                targets = torch.softmax(teacher_logits.masked_fill(~masks, -1e9) / args.temperature, dim=1)
            logits = student(states, actions).masked_fill(~masks, -1e9)
            loss = -(targets * torch.log_softmax(logits / args.temperature, dim=1).masked_fill(~masks, 0)).sum(1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        metrics = evaluate(teacher, student, validation_loader, device, args.temperature)
        if metrics["crossEntropy"] < best_cross_entropy:
            best_cross_entropy = metrics["crossEntropy"]
            best_metrics = metrics
            best = {key: value.detach().cpu().clone() for key, value in student.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[structured-student] epoch={epoch:03d} top1={metrics['top1']:.3%} "
                f"top2={metrics['top2']:.3%} crossEntropy={metrics['crossEntropy']:.4f}"
            )

    assert best is not None
    student.load_state_dict(best)
    manifest = {
        "format": "ntw-policy-v2-compact",
        "featureVersion": 2,
        "cards": CARDS,
        "stateSize": STATE_SIZE,
        "actionSize": ACTION_SIZE,
        "activation": "silu",
        "stateWidths": list(state_widths),
        "actionWidths": list(action_widths),
        "stateLayers": [{"weight": compact(layer.weight), "bias": compact(layer.bias)} for layer in student.state_layers],
        "actionLayers": [{"weight": compact(layer.weight), "bias": compact(layer.bias)} for layer in student.action_layers],
        "validation": best_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"[structured-student] wrote={args.output} metrics={best_metrics}")


if __name__ == "__main__":
    main()
