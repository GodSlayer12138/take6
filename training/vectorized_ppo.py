"""GPU-vectorized PPO for five-player adaptive 6 nimmt!.

The old outcome fine-tuner spends most of its time constructing Python
objects and only uses a tiny fraction of the RTX 4060.  This trainer keeps the
entire game state on CUDA, plays every seat with a shared policy, and learns a
terminal value baseline.  Public per-seat action history is encoded, but
hidden opponent cards are never exposed to the policy.

Run with the project's ``ntw-ai`` interpreter.  The script refuses Conda base
and CPU execution.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn


PLAYERS = 5
CARDS = 54
HAND_SIZE = 10
ROWS = 4
ROW_CAPACITY = 6
STATE_SIZE = 420
ACTION_SIZE = 17
FORMAT = "ntw-policy-v2"
FEATURE_VERSION = 2


def bull_heads_tensor(cards: torch.Tensor) -> torch.Tensor:
    result = torch.ones_like(cards, dtype=torch.float32)
    result = torch.where(cards.remainder(10) == 5, 2.0, result)
    result = torch.where(cards.remainder(10) == 0, 3.0, result)
    result = torch.where(cards.remainder(11) == 0, 5.0, result)
    result = torch.where(cards == 55, 7.0, result)
    return result


class PolicyValueNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.state_layers = nn.ModuleList([
            nn.Linear(STATE_SIZE, 384),
            nn.Linear(384, 256),
        ])
        self.action_layers = nn.ModuleList([
            nn.Linear(256 + ACTION_SIZE, 160),
            nn.Linear(160, 64),
            nn.Linear(64, 1),
        ])
        self.value_layers = nn.ModuleList([
            nn.Linear(256, 128),
            nn.Linear(128, 1),
        ])

    def forward(self, state: torch.Tensor, action_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        context = state
        for layer in self.state_layers:
            context = torch.nn.functional.silu(layer(context))
        expanded = context.unsqueeze(1).expand(-1, action_features.shape[1], -1)
        actions = torch.cat((expanded, action_features), dim=-1)
        for layer in self.action_layers[:-1]:
            actions = torch.nn.functional.silu(layer(actions))
        logits = self.action_layers[-1](actions).squeeze(-1)
        value = torch.nn.functional.silu(self.value_layers[0](context))
        value = self.value_layers[1](value).squeeze(-1)
        return logits, value


class VectorGame:
    def __init__(self, batch_size: int, device: torch.device, generator: torch.Generator) -> None:
        self.batch_size = batch_size
        self.device = device
        noise = torch.rand((batch_size, CARDS), device=device, generator=generator)
        deck = noise.argsort(dim=1) + 1
        dealt = deck[:, : PLAYERS * HAND_SIZE].reshape(batch_size, HAND_SIZE, PLAYERS).transpose(1, 2)
        self.hands = torch.zeros((batch_size, PLAYERS, CARDS), dtype=torch.bool, device=device)
        self.hands.scatter_(2, (dealt - 1), True)
        initial = deck[:, PLAYERS * HAND_SIZE : PLAYERS * HAND_SIZE + ROWS]
        self.rows = torch.zeros((batch_size, ROWS, ROW_CAPACITY), dtype=torch.long, device=device)
        self.rows[:, :, 0] = initial
        self.lengths = torch.ones((batch_size, ROWS), dtype=torch.long, device=device)
        self.row_costs = bull_heads_tensor(initial)
        self.scores = torch.zeros((batch_size, PLAYERS), dtype=torch.float32, device=device)
        self.seen = torch.zeros((batch_size, CARDS), dtype=torch.bool, device=device)
        self.seen.scatter_(1, initial - 1, True)
        self.played = torch.zeros((batch_size, PLAYERS, CARDS), dtype=torch.bool, device=device)
        self.turn = 0

    def state_features(self) -> torch.Tensor:
        batch = self.batch_size
        player_ids = torch.arange(PLAYERS, device=self.device)
        relative_ids = (player_ids[:, None] + player_ids[None, :]).remainder(PLAYERS)
        relative_played = self.played[:, relative_ids, :]
        relative_scores = self.scores[:, relative_ids]
        hands = self.hands
        seen = self.seen[:, None, :].expand(-1, PLAYERS, -1)
        rows = (self.rows.float() / CARDS).reshape(batch, 1, ROWS * ROW_CAPACITY).expand(-1, PLAYERS, -1)
        lengths = (self.lengths.float() / ROW_CAPACITY).reshape(batch, 1, ROWS).expand(-1, PLAYERS, -1)
        costs = (self.row_costs / 25.0).reshape(batch, 1, ROWS).expand(-1, PLAYERS, -1)
        tail_indices = (self.lengths - 1).unsqueeze(-1)
        tails = self.rows.gather(2, tail_indices).squeeze(-1)
        tails = (tails.float() / CARDS).reshape(batch, 1, ROWS).expand(-1, PLAYERS, -1)
        turn = torch.full((batch, PLAYERS, 1), self.turn / HAND_SIZE, device=self.device)
        features = torch.cat((
            hands.float(),
            seen.float(),
            relative_played.reshape(batch, PLAYERS, PLAYERS * CARDS).float(),
            rows,
            lengths,
            costs,
            tails,
            (relative_scores / 50.0).clamp(0, 1),
            turn,
        ), dim=-1)
        if features.shape[-1] != STATE_SIZE:
            raise RuntimeError(f"state feature mismatch: {features.shape[-1]} != {STATE_SIZE}")
        return features.reshape(batch * PLAYERS, STATE_SIZE)

    def action_features(self) -> torch.Tensor:
        batch = self.batch_size
        count = batch * PLAYERS
        cards = torch.arange(1, CARDS + 1, device=self.device).float().view(1, CARDS)
        hands = self.hands.reshape(count, CARDS)
        seen = self.seen[:, None, :].expand(-1, PLAYERS, -1).reshape(count, CARDS)
        rows = self.rows[:, None, :, :].expand(-1, PLAYERS, -1, -1).reshape(count, ROWS, ROW_CAPACITY)
        lengths = self.lengths[:, None, :].expand(-1, PLAYERS, -1).reshape(count, ROWS)
        costs = self.row_costs[:, None, :].expand(-1, PLAYERS, -1).reshape(count, ROWS)
        tails = rows.gather(2, (lengths - 1).unsqueeze(-1)).squeeze(-1).float()

        card_values = cards.expand(count, -1)
        valid = tails.unsqueeze(1) < card_values.unsqueeze(2)
        gaps = card_values.unsqueeze(2) - tails.unsqueeze(1)
        target = gaps.masked_fill(~valid, 10_000.0).argmin(dim=2)
        too_low = ~valid.any(dim=2)
        cheapest = (costs + lengths.float() * 1e-3).argmin(dim=1, keepdim=True).expand(-1, CARDS)
        target = torch.where(too_low, cheapest, target)
        target_one_hot = torch.nn.functional.one_hot(target, ROWS).float()
        target_tail = tails.gather(1, target)
        target_length = lengths.gather(1, target)
        target_cost = costs.gather(1, target)
        gap = torch.where(too_low, torch.zeros_like(card_values), card_values - target_tail - 1)
        open_slots = torch.where(too_low, torch.zeros_like(card_values), 5 - target_length)
        captures = too_low | (target_length >= 5)
        immediate = torch.where(captures, target_cost, torch.zeros_like(target_cost))

        hand_rank = hands.float().cumsum(dim=1) - hands.float()
        hand_count = hands.sum(dim=1, keepdim=True).clamp_min(2)
        hand_rank = hand_rank / (hand_count - 1)
        unknown = ~(seen | hands)
        all_cards = torch.arange(1, CARDS + 1, device=self.device).view(1, 1, CARDS)
        between = (all_cards > target_tail.unsqueeze(2)) & (all_cards < card_values.unsqueeze(2))
        interval = (between & unknown.unsqueeze(1)).sum(dim=2).float()
        unknown_count = unknown.sum(dim=1, keepdim=True).clamp_min(1)
        interval_fraction = interval / unknown_count
        traffic = interval_fraction * (PLAYERS - 1)
        min_tail = tails.min(dim=1).values
        trapped = ((torch.arange(1, CARDS + 1, device=self.device).view(1, CARDS) < min_tail.unsqueeze(1)) & hands).sum(dim=1)
        trapped = trapped.float().unsqueeze(1).expand(-1, CARDS) / HAND_SIZE

        features = torch.cat((
            (card_values / CARDS).unsqueeze(-1),
            (bull_heads_tensor(card_values.long()) / 7.0).unsqueeze(-1),
            hand_rank.unsqueeze(-1),
            target_one_hot,
            (gap / CARDS).clamp(0, 1).unsqueeze(-1),
            (open_slots.float() / 5.0).clamp(0, 1).unsqueeze(-1),
            (immediate / 25.0).clamp(0, 1).unsqueeze(-1),
            too_low.float().unsqueeze(-1),
            captures.float().unsqueeze(-1),
            interval_fraction.unsqueeze(-1),
            (traffic / (PLAYERS - 1)).unsqueeze(-1),
            trapped.unsqueeze(-1),
            (target_cost / 25.0).clamp(0, 1).unsqueeze(-1),
            (target_length.float() / 5.0).clamp(0, 1).unsqueeze(-1),
        ), dim=-1)
        if features.shape[-1] != ACTION_SIZE:
            raise RuntimeError(f"action feature mismatch: {features.shape[-1]} != {ACTION_SIZE}")
        return features

    def step(self, actions: torch.Tensor) -> None:
        actions = actions.reshape(self.batch_size, PLAYERS) + 1
        order = actions.argsort(dim=1)
        batch_ids = torch.arange(self.batch_size, device=self.device)
        for slot in range(PLAYERS):
            player = order[:, slot]
            card = actions.gather(1, player.unsqueeze(1)).squeeze(1)
            tails = self.rows.gather(2, (self.lengths - 1).unsqueeze(-1)).squeeze(-1)
            valid = tails < card.unsqueeze(1)
            gap = (card.unsqueeze(1) - tails).masked_fill(~valid, 10_000)
            target = gap.argmin(dim=1)
            too_low = ~valid.any(dim=1)
            cheapest = (self.row_costs + self.lengths.float() * 1e-3).argmin(dim=1)
            target = torch.where(too_low, cheapest, target)
            target_length = self.lengths[batch_ids, target]
            capture = too_low | (target_length >= 5)

            if capture.any():
                selected = batch_ids[capture]
                selected_players = player[capture]
                selected_rows = target[capture]
                self.scores[selected, selected_players] += self.row_costs[selected, selected_rows]
                self.rows[selected, selected_rows] = 0
                self.rows[selected, selected_rows, 0] = card[capture]
                self.lengths[selected, selected_rows] = 1
                self.row_costs[selected, selected_rows] = bull_heads_tensor(card[capture])
            append = ~capture
            if append.any():
                selected = batch_ids[append]
                selected_rows = target[append]
                positions = target_length[append]
                self.rows[selected, selected_rows, positions] = card[append]
                self.lengths[selected, selected_rows] += 1
                self.row_costs[selected, selected_rows] += bull_heads_tensor(card[append])

        self.hands.scatter_(2, (actions - 1).unsqueeze(-1), False)
        self.seen.scatter_(1, actions - 1, True)
        self.played.scatter_(2, (actions - 1).unsqueeze(-1), True)
        self.turn += 1

    def rewards(self) -> tuple[torch.Tensor, dict[str, float]]:
        win_share, rank, raw = self.terminal_details()
        centered = raw - raw.mean(dim=1, keepdim=True)
        metrics = {
            "winRate": float(win_share.mean().item()),
            "avgScore": float(self.scores.mean().item()),
            "avgRank": float(rank.mean().item()),
        }
        return centered, metrics

    def terminal_details(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        minimum = self.scores.min(dim=1, keepdim=True).values
        winners = (self.scores == minimum).sum(dim=1, keepdim=True)
        win_share = (self.scores == minimum).float() / winners
        lower = (self.scores.unsqueeze(2) > self.scores.unsqueeze(1)).sum(dim=2).float()
        tied = (self.scores.unsqueeze(2) == self.scores.unsqueeze(1)).sum(dim=2).float()
        rank = 1.0 + lower + (tied - 1.0) / 2.0
        raw = win_share * 6.0 - (rank - 1.0) * 0.45 - self.scores * 0.075
        return win_share, rank, raw


@torch.inference_mode()
def collect_rollout(
    model: PolicyValueNet,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator,
    temperature: float,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    model.eval()
    game = VectorGame(batch_size, device, generator)
    states: list[torch.Tensor] = []
    action_features: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    log_probs: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    for _turn in range(HAND_SIZE):
        state = game.state_features()
        action_feature = game.action_features()
        mask = game.hands.reshape(batch_size * PLAYERS, CARDS)
        logits, value = model(state, action_feature)
        logits = (logits / temperature).masked_fill(~mask, -1e9)
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()
        states.append(state)
        action_features.append(action_feature)
        # ``mask`` is a view of the mutable hand tensor.  Clone it before the
        # environment removes cards, otherwise every PPO state would inherit
        # the empty terminal hand.
        masks.append(mask.clone())
        actions.append(action)
        log_probs.append(distribution.log_prob(action))
        values.append(value)
        entropies.append(distribution.entropy())
        game.step(action)
    reward, metrics = game.rewards()
    reward = reward.reshape(batch_size * PLAYERS)
    count = batch_size * PLAYERS * HAND_SIZE
    rollout = {
        "state": torch.cat(states, dim=0),
        "action_features": torch.cat(action_features, dim=0),
        "mask": torch.cat(masks, dim=0),
        "action": torch.cat(actions, dim=0),
        "old_log_prob": torch.cat(log_probs, dim=0),
        "old_value": torch.cat(values, dim=0),
        "reward": reward.repeat(HAND_SIZE),
        "entropy": torch.cat(entropies, dim=0),
    }
    if rollout["state"].shape[0] != count:
        raise RuntimeError("rollout flattening mismatch")
    return rollout, metrics


def scripted_costs(features: torch.Tensor, style: str, hand_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Vectorized equivalent of the browser greedy/cautious policies."""
    immediate = features[:, :, 9] * 25.0
    gap = features[:, :, 7] * CARDS
    open_slots = features[:, :, 8] * 5.0
    too_low = features[:, :, 10] > 0.5
    captures = features[:, :, 11] > 0.5
    if style == "greedy":
        return immediate * 100.0 + gap + (open_slots <= 1.01).float() * 8.0
    traffic = features[:, :, 13] * (PLAYERS - 1)
    # Exact Poisson tail P(X >= open_slots), matching strategicCost().
    term = torch.exp(-traffic)
    cumulative = term.clone()
    for value in range(1, 5):
        term = term * traffic / value
        cumulative = cumulative + term * (open_slots > value).float()
    overflow = (1.0 - cumulative).clamp(0.0, 1.0)
    row_cost = features[:, :, 15] * 25.0
    trapped = features[:, :, 14] * HAND_SIZE
    heads = features[:, :, 1] * 7.0
    adjacency = torch.zeros_like(gap)
    hand_count = torch.full((features.shape[0], 1), HAND_SIZE, device=features.device)
    if hand_mask is not None:
        adjacent = torch.zeros_like(hand_mask)
        adjacent[:, :-1] |= hand_mask[:, 1:]
        adjacent[:, :-2] |= hand_mask[:, 2:]
        adjacency = adjacent.float()
        hand_count = hand_mask.sum(dim=1, keepdim=True).float()
    endgame_weight = 1.0 + (5.0 - hand_count).clamp_min(0.0) * 0.08
    safe = (
        overflow * (0.8 + row_cost * 0.62)
        + gap.clamp_max(30.0) * 0.014
        + trapped * 0.11
        - adjacency * 0.14
        - (heads - 1.0) * 0.045
    ) * endgame_weight
    forced = immediate * 4.8 - (too_low & (immediate <= 2.01)).float() * 0.45
    return torch.where(captures, forced, safe)


@torch.inference_mode()
def collect_arena_rollout(
    model: PolicyValueNet,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator,
    temperature: float,
    opponent_models: tuple[PolicyValueNet | None, PolicyValueNet | None] = (None, None),
    opponent_temperature: float = 0.0,
    reward_weights: tuple[float, float, float] = (6.0, 0.45, 0.075),
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Train one rotating focal seat against cautious/greedy/cautious/random."""
    model.eval()
    game = VectorGame(batch_size, device, generator)
    batch_ids = torch.arange(batch_size, device=device)
    focal = batch_ids.remainder(PLAYERS)
    focal_flat = batch_ids * PLAYERS + focal
    states: list[torch.Tensor] = []
    action_features: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    log_probs: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    for _turn in range(HAND_SIZE):
        all_states = game.state_features()
        all_action_features = game.action_features()
        all_masks = game.hands.reshape(batch_size * PLAYERS, CARDS)
        state = all_states[focal_flat]
        action_feature = all_action_features[focal_flat]
        mask = all_masks[focal_flat]
        logits, value = model(state, action_feature)
        logits = (logits / temperature).masked_fill(~mask, -1e9)
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()
        joint_actions = torch.zeros((batch_size, PLAYERS), dtype=torch.long, device=device)
        joint_actions[batch_ids, focal] = action
        for offset, style in ((1, "cautious"), (2, "greedy"), (3, "cautious")):
            player = (focal + offset).remainder(PLAYERS)
            flat = batch_ids * PLAYERS + player
            proxy = opponent_models[offset - 1] if offset <= 2 else None
            if proxy is not None:
                proxy_logits, _proxy_value = proxy(all_states[flat], all_action_features[flat])
                proxy_logits = proxy_logits.masked_fill(~all_masks[flat], -1e9)
                if opponent_temperature > 0:
                    proxy_probabilities = torch.softmax(proxy_logits / opponent_temperature, dim=1)
                    joint_actions[batch_ids, player] = torch.multinomial(
                        proxy_probabilities, 1, generator=generator
                    ).squeeze(1)
                else:
                    joint_actions[batch_ids, player] = proxy_logits.argmax(dim=1)
            else:
                costs = scripted_costs(all_action_features[flat], style, all_masks[flat]).masked_fill(~all_masks[flat], 1e9)
                # The actual arena's public ``cautious`` strategy is
                # deterministic.  The optional 11% second-choice exploration
                # is used only inside some search determinizations, not by the
                # real opponent seated at offset three.
                joint_actions[batch_ids, player] = costs.argmin(dim=1)
        random_player = (focal + 4).remainder(PLAYERS)
        random_flat = batch_ids * PLAYERS + random_player
        random_action = torch.multinomial(all_masks[random_flat].float(), 1, generator=generator).squeeze(1)
        joint_actions[batch_ids, random_player] = random_action

        states.append(state)
        action_features.append(action_feature)
        masks.append(mask.clone())
        actions.append(action)
        log_probs.append(distribution.log_prob(action))
        values.append(value)
        entropies.append(distribution.entropy())
        game.step(joint_actions)

    win_share, rank, _raw = game.terminal_details()
    win_weight, rank_weight, score_weight = reward_weights
    focal_win = win_share[batch_ids, focal]
    focal_rank = rank[batch_ids, focal]
    focal_score = game.scores[batch_ids, focal]
    reward = focal_win * win_weight - (focal_rank - 1.0) * rank_weight - focal_score * score_weight
    metrics = {
        "winRate": float(win_share[batch_ids, focal].mean().item()),
        "avgScore": float(game.scores[batch_ids, focal].mean().item()),
        "avgRank": float(rank[batch_ids, focal].mean().item()),
    }
    rollout = {
        "state": torch.cat(states, dim=0),
        "action_features": torch.cat(action_features, dim=0),
        "mask": torch.cat(masks, dim=0),
        "action": torch.cat(actions, dim=0),
        "old_log_prob": torch.cat(log_probs, dim=0),
        "old_value": torch.cat(values, dim=0),
        "reward": reward.repeat(HAND_SIZE),
        "entropy": torch.cat(entropies, dim=0),
    }
    return rollout, metrics


def ppo_update(
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    rollout: dict[str, torch.Tensor],
    epochs: int,
    minibatch_size: int,
    clip: float,
    entropy_weight: float,
    value_weight: float,
    temperature: float,
) -> dict[str, float]:
    model.train()
    reward = rollout["reward"]
    advantage = reward - rollout["old_value"]
    advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(0.1)
    total = reward.shape[0]
    totals = {"loss": 0.0, "policy": 0.0, "value": 0.0, "entropy": 0.0, "steps": 0}
    for _epoch in range(epochs):
        permutation = torch.randperm(total, device=reward.device)
        for start in range(0, total, minibatch_size):
            index = permutation[start : start + minibatch_size]
            logits, value = model(rollout["state"][index], rollout["action_features"][index])
            # Match the behavior distribution used by collect_*_rollout.
            # Omitting this temperature makes PPO's importance ratio compare
            # probabilities from two different policies and collapses entropy.
            logits = (logits / temperature).masked_fill(~rollout["mask"][index], -1e9)
            distribution = torch.distributions.Categorical(logits=logits)
            log_prob = distribution.log_prob(rollout["action"][index])
            ratio = (log_prob - rollout["old_log_prob"][index]).exp()
            unclipped = ratio * advantage[index]
            clipped = ratio.clamp(1 - clip, 1 + clip) * advantage[index]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = torch.nn.functional.smooth_l1_loss(value, reward[index])
            entropy = distribution.entropy().mean()
            loss = policy_loss + value_weight * value_loss - entropy_weight * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals["loss"] += float(loss.item())
            totals["policy"] += float(policy_loss.item())
            totals["value"] += float(value_loss.item())
            totals["entropy"] += float(entropy.item())
            totals["steps"] += 1
    steps = max(1, totals.pop("steps"))
    return {key: value / steps for key, value in totals.items()}


def compact(tensor: torch.Tensor) -> list:
    values = tensor.detach().cpu().tolist()
    if values and isinstance(values[0], list):
        return [[round(float(value), 7) for value in row] for row in values]
    return [round(float(value), 7) for value in values]


def export_model(model: PolicyValueNet, output: Path, update: int, metrics: dict[str, float]) -> None:
    def layers(values: nn.ModuleList) -> list[dict]:
        return [{"weight": compact(layer.weight), "bias": compact(layer.bias)} for layer in values]
    manifest = {
        "format": FORMAT,
        "name": "牧场冠军 Policy v2",
        "featureVersion": FEATURE_VERSION,
        "stateSize": STATE_SIZE,
        "actionSize": ACTION_SIZE,
        "cards": CARDS,
        "activation": "silu",
        "stateLayers": layers(model.state_layers),
        "actionLayers": layers(model.action_layers),
        "valueLayers": layers(model.value_layers),
        "training": {"algorithm": "vectorized-ppo", "update": update, **metrics},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def save_checkpoint(model: PolicyValueNet, optimizer: torch.optim.Optimizer, output: Path, update: int, metrics: dict[str, float]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "format": FORMAT,
        "featureVersion": FEATURE_VERSION,
        "update": update,
        "metrics": metrics,
    }, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--mode", choices=("selfplay", "arena"), default="selfplay")
    parser.add_argument("--games-per-update", type=int, default=2048)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--minibatch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--clip", type=float, default=0.18)
    parser.add_argument("--entropy-weight", type=float, default=0.012)
    parser.add_argument("--value-weight", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=2026082101)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/models/ntw-policy-v2.pt"))
    parser.add_argument("--output", type=Path, default=Path("src/game/models/ntw-policy-v2.json"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--champion-proxy", type=Path)
    parser.add_argument("--external-proxy", type=Path)
    parser.add_argument("--opponent-temperature", type=float, default=0.0)
    parser.add_argument("--win-weight", type=float, default=6.0)
    parser.add_argument("--rank-weight", type=float, default=0.45)
    parser.add_argument("--score-weight", type=float, default=0.075)
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()

    executable = Path(sys.executable).resolve()
    if executable.parent.name.lower() != "ntw-ai":
        raise RuntimeError(f"refusing to train outside ntw-ai: {executable}")
    if not torch.cuda.is_available():
        raise RuntimeError("vectorized PPO requires CUDA")
    device = torch.device("cuda")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    print(f"[vppo] executable={executable}")
    print(f"[vppo] torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}")

    model = PolicyValueNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    start_update = 0
    metrics: dict[str, float] = {}
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        if payload.get("format") != FORMAT:
            raise RuntimeError("resume checkpoint is incompatible")
        model.load_state_dict(payload["state_dict"])
        if payload.get("optimizer") and not args.export_only:
            optimizer.load_state_dict(payload["optimizer"])
        start_update = int(payload.get("update", 0))
        metrics = payload.get("metrics") or {}
        print(f"[vppo] resumed={args.resume} update={start_update}")
    if args.export_only:
        export_model(model, args.output, start_update, metrics)
        print(f"[vppo] exported={args.output}")
        return

    opponent_models: list[PolicyValueNet | None] = [None, None]
    for index, proxy_path in enumerate((args.champion_proxy, args.external_proxy)):
        if proxy_path is None:
            continue
        proxy_payload = torch.load(proxy_path, map_location="cpu", weights_only=False)
        if proxy_payload.get("format") != FORMAT:
            raise RuntimeError(f"opponent proxy is incompatible: {proxy_path}")
        proxy = PolicyValueNet().to(device)
        proxy.load_state_dict(proxy_payload["state_dict"])
        proxy.eval()
        for parameter in proxy.parameters():
            parameter.requires_grad_(False)
        opponent_models[index] = proxy
        print(f"[vppo] opponent_proxy={proxy_path}")

    started = time.perf_counter()
    games_seen = 0
    for update in range(start_update + 1, start_update + args.updates + 1):
        if args.mode == "arena":
            rollout, arena_metrics = collect_arena_rollout(
                model,
                args.games_per_update,
                device,
                generator,
                args.temperature,
                (opponent_models[0], opponent_models[1]),
                args.opponent_temperature,
                (args.win_weight, args.rank_weight, args.score_weight),
            )
        else:
            rollout, arena_metrics = collect_rollout(model, args.games_per_update, device, generator, args.temperature)
        losses = ppo_update(
            model,
            optimizer,
            rollout,
            args.ppo_epochs,
            args.minibatch_size,
            args.clip,
            args.entropy_weight,
            args.value_weight,
            args.temperature,
        )
        games_seen += args.games_per_update
        metrics = {**arena_metrics, **losses, "gamesSeen": games_seen}
        if update == 1 or update % 5 == 0:
            elapsed = time.perf_counter() - started
            allocated = torch.cuda.max_memory_allocated() / 1024**2
            print(
                f"[vppo] update={update:04d} games={games_seen} win={arena_metrics['winRate']:.3%} "
                f"score={arena_metrics['avgScore']:.3f} loss={losses['loss']:+.4f} "
                f"policy={losses['policy']:+.4f} value={losses['value']:.4f} entropy={losses['entropy']:.3f} "
                f"vram={allocated:.0f}MiB time={elapsed:.1f}s"
            )
        if update % args.save_every == 0 or update == start_update + args.updates:
            numbered = args.checkpoint.with_name(f"{args.checkpoint.stem}-u{update:04d}{args.checkpoint.suffix}")
            browser = args.output.with_name(f"{args.output.stem}-u{update:04d}{args.output.suffix}")
            save_checkpoint(model, optimizer, numbered, update, metrics)
            export_model(model, browser, update, metrics)
            print(f"[vppo] saved={numbered}")
    save_checkpoint(model, optimizer, args.checkpoint, start_update + args.updates, metrics)
    export_model(model, args.output, start_update + args.updates, metrics)
    print(f"[vppo] checkpoint={args.checkpoint}")
    print(f"[vppo] browser_model={args.output}")


if __name__ == "__main__":
    main()
