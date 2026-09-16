"""Shared public-state counterfactual ranking objectives."""

from __future__ import annotations

import torch


def pairwise_ranking_loss(
    logits: torch.Tensor,
    costs: torch.Tensor,
    masks: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Cost-gap-weighted Bradley-Terry loss over every legal action pair."""
    size = logits.shape[1]
    upper = torch.triu(torch.ones((size, size), dtype=torch.bool, device=logits.device), diagonal=1)
    valid = masks[:, :, None] & masks[:, None, :] & upper[None, :, :]
    left_better_logit = logits[:, :, None] - logits[:, None, :]
    left_better_target = torch.sigmoid((costs[:, None, :] - costs[:, :, None]) / temperature)
    gaps = (costs[:, :, None] - costs[:, None, :]).abs()
    weights = (gaps / 4.0).clamp(0.10, 3.0)
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        left_better_logit,
        left_better_target,
        reduction="none",
    )
    return (losses[valid] * weights[valid]).sum() / weights[valid].sum().clamp_min(1e-6)


def optimal_set_loss(logits: torch.Tensor, costs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Negative log probability assigned to any tied minimum-cost action."""
    legal_costs = costs.masked_fill(~masks, 1e9)
    best = legal_costs.min(dim=1, keepdim=True).values
    optimal = masks & (legal_costs - best).abs().lt(1e-6)
    log_probs = torch.log_softmax(logits.masked_fill(~masks, -1e9), dim=1)
    optimal_log_mass = torch.logsumexp(log_probs.masked_fill(~optimal, -1e9), dim=1)
    decisions = masks.sum(1) > 1
    if not decisions.any():
        return logits.sum() * 0.0
    return -optimal_log_mass[decisions].mean()
