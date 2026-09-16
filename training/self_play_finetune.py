"""Outcome-driven policy fine-tuning for the browser candidate-value network.

The imitation checkpoint is a strong starting point, but its labels optimize a
search heuristic rather than winning a complete game.  This script keeps the
same 270-feature/browser model contract and applies REINFORCE on full ten-turn
games.  A KL penalty to the frozen imitation policy limits destructive drift.

Run this file with the project's ``ntw-ai`` interpreter, never Conda base.
"""

from __future__ import annotations

import argparse
import copy
import math
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn

from train_policy import (
    FEATURE_SIZE,
    FEATURE_VERSION,
    TARGET_SCALE,
    CandidateValueNet,
    bull_heads,
    cheapest_row_index,
    clamp01,
    encode_candidate,
    export_browser_model,
    row_penalty,
    target_row_index,
)


PLAYER_COUNT = 5
HAND_SIZE = 10


def make_game(deck_mode: str, rng: random.Random) -> dict:
    deck_size = 54 if deck_mode == "adaptive" else 104
    deck = list(range(1, deck_size + 1))
    rng.shuffle(deck)
    hands = [[] for _ in range(PLAYER_COUNT)]
    for _ in range(HAND_SIZE):
        for player in range(PLAYER_COUNT):
            hands[player].append(deck.pop())
    for hand in hands:
        hand.sort()
    rows = [[deck.pop()] for _ in range(4)]
    return {
        "deckSize": deck_size,
        "deckMode": deck_mode,
        "hands": hands,
        "rows": rows,
        "seenCards": [row[0] for row in rows],
        "scores": [0] * PLAYER_COUNT,
    }


def resolve_cards(rows: list[list[int]], cards: list[int]) -> tuple[list[list[int]], list[int]]:
    next_rows = [row.copy() for row in rows]
    penalties = [0] * PLAYER_COUNT
    for player, card in sorted(enumerate(cards), key=lambda item: item[1]):
        row_index = target_row_index(next_rows, card)
        too_low = row_index < 0
        if too_low:
            row_index = cheapest_row_index(next_rows)
        row = next_rows[row_index]
        if too_low or len(row) >= 5:
            penalties[player] += row_penalty(row)
            next_rows[row_index] = [card]
        else:
            row.append(card)
    return next_rows, penalties


def perspective_scores(scores: list[int], player: int) -> list[int]:
    return [scores[player], *(score for index, score in enumerate(scores) if index != player)]


def observation(game: dict, player: int) -> dict:
    return {
        "rows": game["rows"],
        "hand": game["hands"][player],
        "seenCards": game["seenCards"],
        "deckSize": game["deckSize"],
        "deckMode": game["deckMode"],
        "playerCount": PLAYER_COUNT,
        "scores": perspective_scores(game["scores"], player),
    }


def placement_cost(card: int, hand: list[int], rows: list[list[int]], record: dict) -> float:
    """Port of the browser's inexpensive cautious opponent policy."""
    row_index = target_row_index(rows, card)
    too_low = row_index < 0
    if too_low:
        row_index = cheapest_row_index(rows)
    row = rows[row_index]
    immediate = row_penalty(row) if too_low or len(row) >= 5 else 0
    if too_low or immediate > 0:
        escape_bonus = 0.45 if too_low and immediate <= 2 else 0.0
        return immediate * 4.8 - escape_bonus

    tail = row[-1]
    known = set(record["seenCards"]) | set(hand)
    unknown = [candidate for candidate in range(1, record["deckSize"] + 1) if candidate not in known]
    interval = sum(tail < candidate < card for candidate in unknown)
    traffic = interval / max(1, len(unknown)) * (PLAYER_COUNT - 1)
    open_slots = 5 - len(row)
    # P(Poisson(traffic) >= open_slots)
    term = math.exp(-traffic)
    cumulative = term
    for value in range(1, max(1, open_slots)):
        term *= traffic / value
        cumulative += term
    overflow = max(0.0, min(1.0, 1.0 - cumulative))
    congestion = overflow * (0.8 + row_penalty(row) * 0.62)
    distance = min(30, card - tail - 1) * 0.014
    remaining = [value for value in hand if value != card]
    min_tail = min(candidate_row[-1] for candidate_row in rows)
    trapped = sum(value < min_tail for value in remaining)
    adjacency = -0.14 if any(value > card and value - card <= 2 for value in remaining) else 0.0
    shed = -(bull_heads(card) - 1) * 0.045
    endgame_weight = 1 + max(0, 5 - len(hand)) * 0.08
    return (congestion + distance + trapped * 0.11 + adjacency + shed) * endgame_weight


def cautious_card(game: dict, player: int) -> int:
    record = observation(game, player)
    hand = game["hands"][player]
    return min(hand, key=lambda card: (placement_cost(card, hand, game["rows"], record), card))


def candidate_tensor(games: list[dict], players: list[int], device: torch.device) -> torch.Tensor:
    hand_size = len(games[0]["hands"][0])
    features: list[list[list[float]]] = []
    for game, player in zip(games, players, strict=True):
        record = observation(game, player)
        features.append([encode_candidate(record, card) for card in record["hand"]])
    tensor = torch.tensor(features, dtype=torch.float32)
    if tensor.shape != (len(games), hand_size, FEATURE_SIZE):
        raise RuntimeError(f"candidate tensor mismatch: {tuple(tensor.shape)}")
    return tensor.to(device, non_blocking=True)


@torch.inference_mode()
def fixed_model_cards(
    games: list[dict], players: list[int], model: nn.Module, device: torch.device
) -> list[int]:
    features = candidate_tensor(games, players, device)
    predictions = model(features)
    indices = predictions.argmin(dim=1).cpu().tolist()
    return [game["hands"][player][index] for game, player, index in zip(games, players, indices, strict=True)]


def remove_card(hand: list[int], card: int) -> None:
    hand.remove(card)


def terminal_rewards(games: list[dict], focal_players: list[int]) -> tuple[torch.Tensor, dict[str, float]]:
    values: list[float] = []
    wins = 0.0
    score_sum = 0.0
    rank_sum = 0.0
    for game, focal in zip(games, focal_players, strict=True):
        scores = game["scores"]
        score = scores[focal]
        minimum = min(scores)
        winner_count = sum(value == minimum for value in scores)
        win_share = 1.0 / winner_count if score == minimum else 0.0
        lower = sum(value < score for value in scores)
        tied = sum(value == score for value in scores)
        rank = 1 + lower + (tied - 1) / 2
        # Win share dominates, with dense rank/score terms reducing variance.
        values.append(win_share * 5.0 - (rank - 1) * 0.42 - score * 0.075)
        wins += win_share
        score_sum += score
        rank_sum += rank
    count = max(1, len(games))
    metrics = {"winRate": wins / count, "avgScore": score_sum / count, "avgRank": rank_sum / count}
    return torch.tensor(values, dtype=torch.float32), metrics


def play_training_batch(
    model: nn.Module,
    frozen: nn.Module,
    device: torch.device,
    deck_mode: str,
    batch_size: int,
    seed: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    rng = random.Random(seed)
    games = [make_game(deck_mode, rng) for _ in range(batch_size)]
    # Rotate the learning seat to make score-perspective and deal position checks explicit.
    focal_players = [index % PLAYER_COUNT for index in range(batch_size)]
    log_probabilities: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    divergences: list[torch.Tensor] = []

    for _turn in range(HAND_SIZE):
        focal_features = candidate_tensor(games, focal_players, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            predictions = model(focal_features)
        logits = (-predictions * TARGET_SCALE / temperature).float()
        distribution = torch.distributions.Categorical(logits=logits)
        chosen_indices = distribution.sample()
        log_probabilities.append(distribution.log_prob(chosen_indices))
        entropies.append(distribution.entropy())
        with torch.no_grad():
            frozen_logits = (-frozen(focal_features) * TARGET_SCALE / temperature).float()
            frozen_distribution = torch.distributions.Categorical(logits=frozen_logits)
        divergences.append(torch.distributions.kl_divergence(distribution, frozen_distribution))

        chosen_cpu = chosen_indices.cpu().tolist()
        cards_by_game: list[list[int | None]] = [[None] * PLAYER_COUNT for _ in games]
        for game_index, (game, player, card_index) in enumerate(zip(games, focal_players, chosen_cpu, strict=True)):
            cards_by_game[game_index][player] = game["hands"][player][card_index]

        # Two frozen neural opponents approximate the arena's two search agents.
        for offset in (1, 2):
            players = [(focal + offset) % PLAYER_COUNT for focal in focal_players]
            cards = fixed_model_cards(games, players, frozen, device)
            for game_index, (player, card) in enumerate(zip(players, cards, strict=True)):
                cards_by_game[game_index][player] = card

        # The remaining seats mirror cautious and random arena baselines.
        cautious_players = [(focal + 3) % PLAYER_COUNT for focal in focal_players]
        random_players = [(focal + 4) % PLAYER_COUNT for focal in focal_players]
        for game_index, (game, cautious_player, random_player) in enumerate(
            zip(games, cautious_players, random_players, strict=True)
        ):
            cards_by_game[game_index][cautious_player] = cautious_card(game, cautious_player)
            random_hand = game["hands"][random_player]
            cards_by_game[game_index][random_player] = random_hand[rng.randrange(len(random_hand))]

        for game, optional_cards in zip(games, cards_by_game, strict=True):
            if any(card is None for card in optional_cards):
                raise RuntimeError("training policy did not fill every seat")
            cards = [int(card) for card in optional_cards]
            for player, card in enumerate(cards):
                remove_card(game["hands"][player], card)
            game["rows"], penalties = resolve_cards(game["rows"], cards)
            for player, penalty in enumerate(penalties):
                game["scores"][player] += penalty
            game["seenCards"].extend(cards)

    rewards_cpu, metrics = terminal_rewards(games, focal_players)
    rewards = rewards_cpu.to(device)
    advantages = (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(0.15)
    log_prob = torch.stack(log_probabilities, dim=1).sum(dim=1)
    entropy = torch.stack(entropies).mean()
    divergence = torch.stack(divergences).mean()
    policy_loss = -(log_prob * advantages.detach()).mean()
    return policy_loss, entropy, divergence, metrics


def play_symmetric_batch(
    model: nn.Module,
    frozen: nn.Module,
    device: torch.device,
    deck_mode: str,
    batch_size: int,
    seed: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Play every seat with the shared trainable policy.

    Centering terminal utility within each game produces a competitive
    zero-sum-like advantage while yielding five trajectories per deal.
    """
    rng = random.Random(seed)
    games = [make_game(deck_mode, rng) for _ in range(batch_size)]
    expanded_games = [game for game in games for _player in range(PLAYER_COUNT)]
    players = [player for _game in games for player in range(PLAYER_COUNT)]
    log_probabilities: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    divergences: list[torch.Tensor] = []

    for _turn in range(HAND_SIZE):
        features = candidate_tensor(expanded_games, players, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            predictions = model(features)
        logits = (-predictions * TARGET_SCALE / temperature).float()
        distribution = torch.distributions.Categorical(logits=logits)
        chosen_indices = distribution.sample()
        log_probabilities.append(distribution.log_prob(chosen_indices).reshape(batch_size, PLAYER_COUNT))
        entropies.append(distribution.entropy())
        with torch.no_grad():
            frozen_logits = (-frozen(features) * TARGET_SCALE / temperature).float()
            frozen_distribution = torch.distributions.Categorical(logits=frozen_logits)
        divergences.append(torch.distributions.kl_divergence(distribution, frozen_distribution))

        chosen_cpu = chosen_indices.cpu().tolist()
        cursor = 0
        for game in games:
            cards = []
            for player in range(PLAYER_COUNT):
                cards.append(game["hands"][player][chosen_cpu[cursor]])
                cursor += 1
            for player, card in enumerate(cards):
                remove_card(game["hands"][player], card)
            game["rows"], penalties = resolve_cards(game["rows"], cards)
            for player, penalty in enumerate(penalties):
                game["scores"][player] += penalty
            game["seenCards"].extend(cards)

    rewards_cpu, flat_metrics = terminal_rewards(expanded_games, players)
    rewards = rewards_cpu.reshape(batch_size, PLAYER_COUNT).to(device)
    advantages = rewards - rewards.mean(dim=1, keepdim=True)
    advantages = advantages / advantages.std(unbiased=False).clamp_min(0.15)
    log_prob = torch.stack(log_probabilities, dim=2).sum(dim=2)
    entropy = torch.stack(entropies).mean()
    divergence = torch.stack(divergences).mean()
    policy_loss = -(log_prob * advantages.detach()).mean()
    metrics = {
        "winRate": flat_metrics["winRate"],
        "avgScore": flat_metrics["avgScore"],
        "avgRank": flat_metrics["avgRank"],
    }
    return policy_loss, entropy, divergence, metrics


def load_checkpoint(path: Path, device: torch.device) -> tuple[CandidateValueNet, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("featureVersion") != FEATURE_VERSION or payload.get("featureSize") != FEATURE_SIZE:
        raise RuntimeError("checkpoint feature contract is incompatible")
    model = CandidateValueNet()
    model.load_state_dict(payload["state_dict"])
    return model.to(device), payload


def save_checkpoint(
    model: CandidateValueNet,
    path: Path,
    source: Path,
    deck_mode: str,
    update: int,
    metrics: dict[str, float],
) -> None:
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": state,
            "featureVersion": FEATURE_VERSION,
            "featureSize": FEATURE_SIZE,
            "metrics": metrics,
            "sources": [str(source)],
            "training": "outcome-reinforce-kl",
            "deckMode": deck_mode,
            "update": update,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deck-mode", choices=("adaptive", "classic"), required=True)
    parser.add_argument("--updates", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=1.35)
    parser.add_argument("--entropy-weight", type=float, default=0.012)
    parser.add_argument("--kl-weight", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=2026082003)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--self-play", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    executable = Path(sys.executable).resolve()
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "none"
    print(f"[rl] executable={executable}")
    print(f"[rl] torch={torch.__version__} device={device} gpu={gpu_name}")
    if executable.parent.name.lower() != "ntw-ai":
        raise RuntimeError(f"refusing to train outside ntw-ai: {executable}")
    if device.type != "cuda":
        raise RuntimeError("outcome fine-tuning requires CUDA")

    model, initial_payload = load_checkpoint(args.initial_checkpoint, device)
    frozen = copy.deepcopy(model).eval()
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=2e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.updates), eta_min=args.learning_rate * 0.12
    )
    started = time.perf_counter()
    latest_metrics: dict[str, float] = {}

    for update in range(1, args.updates + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        batch_function = play_symmetric_batch if args.self_play else play_training_batch
        policy_loss, entropy, divergence, latest_metrics = batch_function(
            model,
            frozen,
            device,
            args.deck_mode,
            args.batch_size,
            args.seed + update * 1_000_003,
            args.temperature,
        )
        loss = policy_loss - args.entropy_weight * entropy + args.kl_weight * divergence
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if update == 1 or update % 5 == 0:
            elapsed = time.perf_counter() - started
            allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(
                f"[rl] update={update:04d} loss={loss.item():+.4f} policy={policy_loss.item():+.4f} "
                f"entropy={entropy.item():.3f} kl={divergence.item():.5f} grad={float(gradient_norm):.3f} "
                f"win={latest_metrics['winRate']:.3%} score={latest_metrics['avgScore']:.3f} "
                f"rank={latest_metrics['avgRank']:.3f} vram={allocated:.0f}MiB time={elapsed:.1f}s"
            )

        if update % args.save_every == 0 or update == args.updates:
            numbered = args.checkpoint.with_name(f"{args.checkpoint.stem}-u{update:04d}{args.checkpoint.suffix}")
            save_checkpoint(model, numbered, args.initial_checkpoint, args.deck_mode, update, latest_metrics)

    save_checkpoint(model, args.checkpoint, args.initial_checkpoint, args.deck_mode, args.updates, latest_metrics)
    export_browser_model(
        model,
        args.output,
        {
            **(initial_payload.get("metrics") or {}),
            "rlWinRate": latest_metrics.get("winRate", 0.0),
            "rlAvgScore": latest_metrics.get("avgScore", 0.0),
            "rlAvgRank": latest_metrics.get("avgRank", 0.0),
            "rlUpdates": args.updates,
        },
        [str(args.initial_checkpoint), f"outcome-self-play:{args.deck_mode}"],
    )
    print(f"[rl] checkpoint={args.checkpoint}")
    print(f"[rl] browser_model={args.output}")


if __name__ == "__main__":
    main()
