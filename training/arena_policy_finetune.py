"""Advantage-weighted update from trajectories against the real JS arena pool."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from train_policy import (
    FEATURE_SIZE,
    FEATURE_VERSION,
    TARGET_SCALE,
    CandidateValueNet,
    encode_candidate,
    export_browser_model,
)


def read_records(path: Path) -> list[dict]:
    records = []
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            value = json.loads(line)
            if 'meta' not in value:
                records.append(value)
    return records


def tensorize(records: list[dict], win_weight: float, rank_weight: float, score_weight: float) -> TensorDataset:
    count = len(records)
    features = torch.zeros((count, 10, FEATURE_SIZE), dtype=torch.float32)
    mask = torch.zeros((count, 10), dtype=torch.bool)
    chosen = torch.zeros(count, dtype=torch.long)
    rewards = torch.tensor([
        float(record['winShare']) * win_weight
        - (float(record['finalRank']) - 1) * rank_weight
        - float(record['finalScore']) * score_weight
        for record in records
    ], dtype=torch.float32)
    turns = torch.tensor([int(record['turn']) for record in records], dtype=torch.long)
    for row_index, record in enumerate(records):
        for candidate_index, card in enumerate(record['hand']):
            features[row_index, candidate_index] = torch.tensor(encode_candidate(record, card))
            mask[row_index, candidate_index] = True
            if card == record['chosenCard']:
                chosen[row_index] = candidate_index
    advantages = torch.zeros_like(rewards)
    for turn in range(10):
        selected = turns == turn
        values = rewards[selected]
        advantages[selected] = (values - values.mean()) / values.std(unbiased=False).clamp_min(0.15)
    return TensorDataset(features, mask, chosen, advantages)


def load_model(path: Path, device: torch.device) -> CandidateValueNet:
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('featureVersion') != FEATURE_VERSION or payload.get('featureSize') != FEATURE_SIZE:
        raise RuntimeError('checkpoint feature contract is incompatible')
    model = CandidateValueNet()
    model.load_state_dict(payload['state_dict'])
    return model.to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--initial-checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--learning-rate', type=float, default=8e-6)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--entropy-weight', type=float, default=0.004)
    parser.add_argument('--kl-weight', type=float, default=0.25)
    parser.add_argument('--bc-weight', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=2026082202)
    parser.add_argument('--win-weight', type=float, default=5.0)
    parser.add_argument('--rank-weight', type=float, default=0.42)
    parser.add_argument('--score-weight', type=float, default=0.075)
    args = parser.parse_args()

    executable = Path(sys.executable).resolve()
    if executable.parent.name.lower() != 'ntw-ai':
        raise RuntimeError(f'refusing to train outside ntw-ai: {executable}')
    if not torch.cuda.is_available():
        raise RuntimeError('arena policy fine-tuning requires CUDA')
    device = torch.device('cuda')
    print(f'[arena-train] executable={executable}')
    print(f'[arena-train] torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    records = read_records(args.data)
    wins = sum(record['winShare'] for record in records[::10]) / max(1, len(records) // 10)
    print(f'[arena-train] tensorizing {len(records)} states from {len(records) // 10} games; behavior_win={wins:.3%}')
    dataset = tensorize(records, args.win_weight, args.rank_weight, args.score_weight)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    model = load_model(args.initial_checkpoint, device)
    frozen = copy.deepcopy(model).eval()
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=2e-5)

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {'loss': 0.0, 'policy': 0.0, 'kl': 0.0, 'entropy': 0.0, 'count': 0}
        for features, mask, chosen, advantages in loader:
            features = features.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            chosen = chosen.to(device, non_blocking=True)
            advantages = advantages.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features)
            logits = (-prediction * TARGET_SCALE / args.temperature).masked_fill(~mask, -1e9)
            log_probs = torch.log_softmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)
            selected_log_prob = log_probs.gather(1, chosen.unsqueeze(1)).squeeze(1)
            policy_loss = -(selected_log_prob * advantages).mean()
            behavior_cloning = -selected_log_prob.mean()
            entropy = -(probs * log_probs).sum(dim=1).mean()
            with torch.no_grad():
                frozen_logits = (-frozen(features) * TARGET_SCALE / args.temperature).masked_fill(~mask, -1e9)
                frozen_log_probs = torch.log_softmax(frozen_logits, dim=1)
            divergence = (probs * (log_probs - frozen_log_probs)).sum(dim=1).mean()
            loss = policy_loss + args.bc_weight * behavior_cloning - args.entropy_weight * entropy + args.kl_weight * divergence
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = features.shape[0]
            totals['loss'] += loss.item() * count
            totals['policy'] += policy_loss.item() * count
            totals['kl'] += divergence.item() * count
            totals['entropy'] += entropy.item() * count
            totals['count'] += count
        print(
            f"[arena-train] epoch={epoch:02d} loss={totals['loss']/totals['count']:+.5f} "
            f"policy={totals['policy']/totals['count']:+.5f} kl={totals['kl']/totals['count']:.5f} "
            f"entropy={totals['entropy']/totals['count']:.4f}"
        )

    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'state_dict': state,
        'featureVersion': FEATURE_VERSION,
        'featureSize': FEATURE_SIZE,
        'metrics': {'behaviorWinRate': wins, 'arenaStates': len(records)},
        'sources': [str(args.initial_checkpoint), str(args.data)],
        'training': 'real-arena-advantage-kl',
    }, args.checkpoint)
    export_browser_model(model, args.output, {'behaviorWinRate': wins, 'arenaStates': len(records)}, [str(args.initial_checkpoint), str(args.data)])
    print(f'[arena-train] checkpoint={args.checkpoint}')
    print(f'[arena-train] browser_model={args.output}')


if __name__ == '__main__':
    main()
