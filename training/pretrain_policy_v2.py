"""Distill high-budget search records into the public-history Policy v2."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from counterfactual_loss import optimal_set_loss, pairwise_ranking_loss

from vectorized_ppo import (
    ACTION_SIZE,
    CARDS,
    FEATURE_VERSION,
    FORMAT,
    HAND_SIZE,
    PLAYERS,
    STATE_SIZE,
    PolicyValueNet,
    export_model,
)


def bull_heads(card: int) -> int:
    if card == 55:
        return 7
    if card % 11 == 0:
        return 5
    if card % 10 == 0:
        return 3
    if card % 10 == 5:
        return 2
    return 1


def row_penalty(row: list[int]) -> int:
    return sum(bull_heads(card) for card in row)


def target_row(rows: list[list[int]], card: int) -> tuple[int, bool]:
    valid = [(card - row[-1], index) for index, row in enumerate(rows) if row[-1] < card]
    if valid:
        return min(valid)[1], False
    return min(range(4), key=lambda index: (row_penalty(rows[index]), len(rows[index]))), True


def encode_state(record: dict, played_cards: list[list[int]]) -> list[float]:
    hand_mask = [0.0] * CARDS
    for card in record["hand"]:
        hand_mask[card - 1] = 1.0
    seen_mask = [0.0] * CARDS
    for card in record["seenCards"]:
        seen_mask[card - 1] = 1.0
    played_masks = [[0.0] * CARDS for _ in range(PLAYERS)]
    for player, cards in enumerate(played_cards):
        for card in cards:
            played_masks[player][card - 1] = 1.0
    board: list[float] = []
    for row in record["rows"]:
        board.extend(card / CARDS for card in row)
        board.extend([0.0] * (6 - len(row)))
    scores = list(record.get("scores") or [0] * PLAYERS)
    features = [
        *hand_mask,
        *seen_mask,
        *(value for mask in played_masks for value in mask),
        *board,
        *(len(row) / 6 for row in record["rows"]),
        *(min(1.0, row_penalty(row) / 25) for row in record["rows"]),
        *(row[-1] / CARDS for row in record["rows"]),
        *(min(1.0, max(0.0, score / 50)) for score in scores[:PLAYERS]),
        record["turn"] / HAND_SIZE,
    ]
    if len(features) != STATE_SIZE:
        raise RuntimeError(f"state mismatch: {len(features)}")
    return features


def encode_action(record: dict, card: int) -> list[float]:
    rows = record["rows"]
    hand = record["hand"]
    row_index, too_low = target_row(rows, card)
    row = rows[row_index]
    tail = row[-1]
    captures = too_low or len(row) >= 5
    immediate = row_penalty(row) if captures else 0
    rank = sorted(hand).index(card) / max(1, len(hand) - 1)
    one_hot = [float(index == row_index) for index in range(4)]
    known = set(record["seenCards"]) | set(hand)
    unknown = [candidate for candidate in range(1, CARDS + 1) if candidate not in known]
    interval = sum(not too_low and tail < candidate < card for candidate in unknown)
    interval_fraction = interval / max(1, len(unknown))
    min_tail = min(candidate[-1] for candidate in rows)
    trapped = sum(candidate < min_tail for candidate in hand) / HAND_SIZE
    features = [
        card / CARDS,
        bull_heads(card) / 7,
        rank,
        *one_hot,
        0.0 if too_low else max(0, card - tail - 1) / CARDS,
        0.0 if too_low else max(0, 5 - len(row)) / 5,
        min(1.0, immediate / 25),
        float(too_low),
        float(captures),
        interval_fraction,
        interval_fraction,
        trapped,
        min(1.0, row_penalty(row) / 25),
        min(1.0, len(row) / 5),
    ]
    if len(features) != ACTION_SIZE:
        raise RuntimeError(f"action mismatch: {len(features)}")
    return features


def read_games(
    paths: list[Path],
    limit: int | None,
    game_min: int | None = None,
    game_max: int | None = None,
) -> list[list[dict]]:
    games: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for source_index, path in enumerate(paths):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if "meta" in record or record.get("deckMode") != "adaptive" or record.get("playerCount") != PLAYERS:
                    continue
                game_index = int(record["gameIndex"])
                if game_min is not None and game_index < game_min:
                    continue
                if game_max is not None and game_index >= game_max:
                    continue
                # Counterfactual arena records are already written from the
                # focal player's perspective, so they do not need an absolute
                # seat id. Treat that perspective as seat zero. Per-opponent
                # cards remain unknown; the public seen mask still carries the
                # complete legal information without inventing seat ownership.
                if "player" not in record:
                    record["player"] = 0
                games[(source_index, game_index)].append(record)
                if limit and sum(map(len, games.values())) >= limit:
                    break
    return [sorted(records, key=lambda record: (record["turn"], record["player"])) for records in games.values()]


def prepare(games: list[list[dict]]) -> list[tuple[dict, list[list[int]]]]:
    prepared: list[tuple[dict, list[list[int]]]] = []
    for records in games:
        by_player = {player: sorted((record for record in records if record["player"] == player), key=lambda record: record["turn"]) for player in range(PLAYERS)}
        played_absolute = [[] for _ in range(PLAYERS)]
        action_by_turn = [[None] * HAND_SIZE for _ in range(PLAYERS)]
        for player, player_records in by_player.items():
            for index in range(1, len(player_records)):
                removed = set(player_records[index - 1]["hand"]) - set(player_records[index]["hand"])
                if len(removed) == 1:
                    action_by_turn[player][index - 1] = removed.pop()
        for record in records:
            turn = int(record["turn"])
            player = int(record["player"])
            relative = [(player + offset) % PLAYERS for offset in range(PLAYERS)]
            recorded_histories = record.get("playedCards")
            if recorded_histories:
                histories = [list(cards) for cards in recorded_histories[:PLAYERS]]
                while len(histories) < PLAYERS:
                    histories.append([])
            else:
                histories = [[card for card in action_by_turn[seat][:turn] if card is not None] for seat in relative]
            prepared.append((record, histories))
    return prepared


def tensorize(prepared: list[tuple[dict, list[list[int]]]]) -> TensorDataset:
    count = len(prepared)
    states = torch.zeros((count, STATE_SIZE), dtype=torch.float32)
    actions = torch.zeros((count, HAND_SIZE, ACTION_SIZE), dtype=torch.float32)
    targets = torch.zeros((count, HAND_SIZE), dtype=torch.float32)
    mask = torch.zeros((count, HAND_SIZE), dtype=torch.bool)
    for index, (record, histories) in enumerate(prepared):
        states[index] = torch.tensor(encode_state(record, histories))
        for candidate, (card, utility) in enumerate(record["labels"]):
            actions[index, candidate] = torch.tensor(encode_action(record, int(card)))
            targets[index, candidate] = float(utility)
            mask[index, candidate] = True
    return TensorDataset(states, actions, targets, mask)


@torch.inference_mode()
def evaluate(model: PolicyValueNet, loader: DataLoader, device: torch.device, temperature: float) -> dict[str, float]:
    model.eval()
    correct = 0
    top2 = 0
    top3 = 0
    top4 = 0
    regret = 0.0
    candidate_regret3 = 0.0
    candidate_regret4 = 0.0
    candidate_optimal3 = 0
    candidate_optimal4 = 0
    decisions = 0
    for state, action, target, mask in loader:
        state, action, target, mask = state.to(device), action.to(device), target.to(device), mask.to(device)
        logits, _value = model(state, action)
        logits = logits.masked_fill(~mask, -1e9)
        prediction = logits.argmax(dim=1)
        expert = target.masked_fill(~mask, 1e9).argmin(dim=1)
        decision = mask.sum(dim=1) > 1
        correct += ((prediction == expert) & decision).sum().item()
        top2 += ((logits.topk(2, dim=1).indices == expert.unsqueeze(1)).any(dim=1) & decision).sum().item()
        top3 += ((logits.topk(3, dim=1).indices == expert.unsqueeze(1)).any(dim=1) & decision).sum().item()
        top4 += ((logits.topk(4, dim=1).indices == expert.unsqueeze(1)).any(dim=1) & decision).sum().item()
        selected = target.gather(1, prediction.unsqueeze(1)).squeeze(1)
        best = target.gather(1, expert.unsqueeze(1)).squeeze(1)
        legal_target = target.masked_fill(~mask, 1e9)
        candidate3 = legal_target.gather(1, logits.topk(3, dim=1).indices).min(dim=1).values
        candidate4 = legal_target.gather(1, logits.topk(4, dim=1).indices).min(dim=1).values
        regret += ((selected - best) * decision).sum().item()
        candidate_regret3 += ((candidate3 - best) * decision).sum().item()
        candidate_regret4 += ((candidate4 - best) * decision).sum().item()
        candidate_optimal3 += ((candidate3 - best).abs().lt(1e-6) & decision).sum().item()
        candidate_optimal4 += ((candidate4 - best).abs().lt(1e-6) & decision).sum().item()
        decisions += decision.sum().item()
    return {
        "accuracy": correct / max(1, decisions),
        "top2": top2 / max(1, decisions),
        "top3": top3 / max(1, decisions),
        "top4": top4 / max(1, decisions),
        "regret": regret / max(1, decisions),
        "candidateRegret3": candidate_regret3 / max(1, decisions),
        "candidateRegret4": candidate_regret4 / max(1, decisions),
        "candidateOptimal3": candidate_optimal3 / max(1, decisions),
        "candidateOptimal4": candidate_optimal4 / max(1, decisions),
        "temperature": temperature,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--validation", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--hard-weight", type=float, default=0.0)
    parser.add_argument("--regression-weight", type=float, default=0.0)
    parser.add_argument("--pairwise-weight", type=float, default=0.0)
    parser.add_argument("--pairwise-temperature", type=float, default=1.0)
    parser.add_argument("--anchor-weight", type=float, default=0.0)
    parser.add_argument("--optimal-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026082102)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--train-game-min", type=int)
    parser.add_argument("--train-game-max", type=int)
    parser.add_argument("--validation-game-min", type=int)
    parser.add_argument("--validation-game-max", type=int)
    parser.add_argument("--train-turn-max", type=int, default=9)
    parser.add_argument("--selection", choices=("regret", "top3", "top4", "candidate3", "candidate4", "candidate34"), default="regret")
    parser.add_argument("--initial", type=Path)
    args = parser.parse_args()

    auxiliary_weight = args.hard_weight + args.regression_weight + args.pairwise_weight + args.anchor_weight + args.optimal_weight
    if min(args.hard_weight, args.regression_weight, args.pairwise_weight, args.anchor_weight, args.optimal_weight) < 0 or auxiliary_weight > 1:
        raise ValueError("hard, regression, pairwise, anchor, and optimal weights must be non-negative and sum to at most 1")
    if args.temperature <= 0 or args.pairwise_temperature <= 0:
        raise ValueError("temperatures must be positive")

    executable = Path(sys.executable).resolve()
    if executable.parent.name.lower() != "ntw-ai":
        raise RuntimeError(f"refusing to train outside ntw-ai: {executable}")
    if not torch.cuda.is_available():
        raise RuntimeError("Policy v2 pretraining requires CUDA")
    device = torch.device("cuda")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(f"[v2-pretrain] executable={executable}")
    print(f"[v2-pretrain] torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}")
    train = [
        item
        for item in prepare(read_games(args.train, args.limit, args.train_game_min, args.train_game_max))
        if int(item[0]["turn"]) <= args.train_turn_max
    ]
    validation = prepare(read_games(args.validation, args.limit, args.validation_game_min, args.validation_game_max))
    print(f"[v2-pretrain] tensorizing {len(train)} train / {len(validation)} validation states")
    train_loader = DataLoader(tensorize(train), batch_size=args.batch_size, shuffle=True, pin_memory=True)
    validation_loader = DataLoader(tensorize(validation), batch_size=args.batch_size * 2, shuffle=False, pin_memory=True)
    model = PolicyValueNet().to(device)
    anchor_model = None
    if args.initial:
        payload = torch.load(args.initial, map_location="cpu", weights_only=False)
        if payload.get("format") != FORMAT or payload.get("featureVersion") != FEATURE_VERSION:
            raise RuntimeError(f"incompatible initial checkpoint: {args.initial}")
        model.load_state_dict(payload["state_dict"])
        print(f"[v2-pretrain] initial={args.initial}")
        if args.anchor_weight > 0:
            anchor_model = copy.deepcopy(model).eval()
            for parameter in anchor_model.parameters():
                parameter.requires_grad_(False)
    elif args.anchor_weight > 0:
        raise ValueError("--anchor-weight requires --initial")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    best_regret = math.inf
    best_selection = -math.inf
    best_state = None
    best_metrics: dict[str, float] = {}
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        examples = 0
        for state, action, target, mask in train_loader:
            state, action, target, mask = state.to(device), action.to(device), target.to(device), mask.to(device)
            raw_logits, _value = model(state, action)
            logits = (raw_logits / args.temperature).masked_fill(~mask, -1e9)
            target_probs = torch.softmax((-target / args.temperature).masked_fill(~mask, -1e9), dim=1)
            soft_loss = -(target_probs * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
            expert = target.masked_fill(~mask, 1e9).argmin(dim=1)
            hard_loss = torch.nn.functional.cross_entropy(logits, expert)
            masked_raw_logits = raw_logits.masked_fill(~mask, 0)
            counts = mask.sum(1, keepdim=True).clamp_min(1)
            target_mean = target.masked_fill(~mask, 0).sum(1, keepdim=True) / counts
            logit_mean = masked_raw_logits.sum(1, keepdim=True) / counts
            target_values = -(target - target_mean) / 5.0
            predicted_values = raw_logits - logit_mean
            regression_loss = torch.nn.functional.smooth_l1_loss(predicted_values[mask], target_values[mask])
            pairwise_loss = pairwise_ranking_loss(raw_logits, target, mask, args.pairwise_temperature)
            best_set_loss = optimal_set_loss(raw_logits, target, mask)
            if anchor_model is not None:
                with torch.no_grad():
                    anchor_logits, _ = anchor_model(state, action)
                    anchor_probs = torch.softmax(anchor_logits.masked_fill(~mask, -1e9) / args.temperature, dim=1)
                anchor_loss = -(anchor_probs * torch.log_softmax(logits, dim=1).masked_fill(~mask, 0)).sum(1).mean()
            else:
                anchor_loss = raw_logits.new_zeros(())
            loss = (
                soft_loss * (1.0 - auxiliary_weight)
                + hard_loss * args.hard_weight
                + regression_loss * args.regression_weight
                + pairwise_loss * args.pairwise_weight
                + anchor_loss * args.anchor_weight
                + best_set_loss * args.optimal_weight
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * state.shape[0]
            examples += state.shape[0]
        metrics = evaluate(model, validation_loader, device, args.temperature)
        if args.selection == "regret":
            selection_value = -metrics["regret"]
        elif args.selection == "candidate3":
            selection_value = -metrics["candidateRegret3"]
        elif args.selection == "candidate4":
            selection_value = -metrics["candidateRegret4"]
        elif args.selection == "candidate34":
            selection_value = -(metrics["candidateRegret3"] + metrics["candidateRegret4"])
        else:
            selection_value = metrics[args.selection]
        if selection_value > best_selection:
            best_selection = selection_value
            best_regret = metrics["regret"]
            best_metrics = metrics
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"[v2-pretrain] epoch={epoch:03d} loss={total_loss/examples:.4f} "
                f"accuracy={metrics['accuracy']:.3%} top2={metrics['top2']:.3%} "
                f"top3={metrics['top3']:.3%} top4={metrics['top4']:.3%} regret={metrics['regret']:.4f} "
                f"c3={metrics['candidateRegret3']:.4f} c4={metrics['candidateRegret4']:.4f}"
            )
    if best_state is None:
        raise RuntimeError("pretraining produced no checkpoint")
    model.load_state_dict(best_state)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "optimizer": None, "format": FORMAT, "featureVersion": FEATURE_VERSION, "update": 0, "metrics": best_metrics}, args.checkpoint)
    export_model(model, args.output, 0, best_metrics)
    print(f"[v2-pretrain] checkpoint={args.checkpoint}")
    print(f"[v2-pretrain] browser_model={args.output}")


if __name__ == "__main__":
    main()
