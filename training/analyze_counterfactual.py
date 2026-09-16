"""Report per-turn quality and noise diagnostics for counterfactual labels."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def read_games(paths: list[Path]) -> dict[tuple[int, int], list[dict]]:
    games: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for source, path in enumerate(paths):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if "meta" not in record:
                    games[(source, int(record["gameIndex"]))].append(record)
    return games


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", type=Path, nargs="+")
    args = parser.parse_args()
    aggregates = defaultdict(lambda: {"states": 0, "choices": 0, "regret": 0.0, "gap": 0.0, "ties": 0})
    for records in read_games(args.paths).values():
        records.sort(key=lambda item: int(item["turn"]))
        for index, record in enumerate(records):
            labels = {int(card): float(cost) for card, cost in record["labels"]}
            ordered = sorted(labels.values())
            turn = int(record["turn"])
            entry = aggregates[turn]
            entry["states"] += 1
            entry["gap"] += ordered[1] - ordered[0] if len(ordered) > 1 else 0.0
            entry["ties"] += int(len(ordered) > 1 and abs(ordered[1] - ordered[0]) < 1e-9)
            if index + 1 < len(records):
                removed = set(record["hand"]) - set(records[index + 1]["hand"])
                if len(removed) == 1:
                    chosen = int(removed.pop())
                    entry["regret"] += labels[chosen] - ordered[0]
                    entry["choices"] += 1
    total_choices = total_regret = 0.0
    for turn in sorted(aggregates):
        entry = aggregates[turn]
        choices = max(1, entry["choices"])
        states = max(1, entry["states"])
        total_choices += entry["choices"]
        total_regret += entry["regret"]
        print(
            f"turn={turn} states={entry['states']} behaviorRegret={entry['regret']/choices:.4f} "
            f"bestGap={entry['gap']/states:.4f} tieRate={entry['ties']/states:.2%}"
        )
    print(f"all behaviorRegret={total_regret/max(1, total_choices):.4f} choices={int(total_choices)}")


if __name__ == "__main__":
    main()
