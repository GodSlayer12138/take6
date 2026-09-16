"""Parallelize independent ETH evaluation deal groups without changing games.

Models are frozen. Each worker sees exactly the same per-game seeds and seat
rotation as reproduce_eth_dirv.evaluate. Four rotations form one bootstrap unit.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

from reproduce_eth_dirv import (
    FORMAT, REPORT_URL, ROOT, bootstrap_interval, dump_json, load_agent,
    np, play_game, provenance, torch,
)

WORKER_AGENTS = None
WORKER_SEED = None
NAMES = ["alpha", "dirv", "random1", "random2"]


def initialize_worker(alpha_path, dirv_path, alpha_simulations, dirv_simulations, seed):
    global WORKER_AGENTS, WORKER_SEED
    torch.set_num_threads(1)
    alpha, _ = load_agent(alpha_path, alpha_simulations, "cpu")
    dirv, _ = load_agent(dirv_path, dirv_simulations, "cpu")
    if alpha.kind != "alpha" or dirv.kind != "dirv":
        raise ValueError("Reversed algorithm checkpoints")
    WORKER_AGENTS = [alpha, dirv, None, None]
    WORKER_SEED = seed


def evaluate_block(block):
    agents = WORKER_AGENTS
    for agent in agents[:2]:
        agent.decision_seconds.clear()
    shares, penalties, stricts, legacies, records = [], [], [], [], []
    for rotation in range(4):
        order = [(i + rotation) % 4 for i in range(4)]
        deal_seed = WORKER_SEED + block * 104729
        scores, _ = play_game([agents[i] for i in order], deal_seed, WORKER_SEED + 2_000_000_000 + block * 65537 + rotation * 8191)
        wins = (scores == max(scores)).astype(float)
        share = wins / wins.sum()
        strict = wins if wins.sum() == 1 else np.zeros(4)
        legacy = np.eye(4)[int(np.argmax(scores))]
        inverse = np.argsort(order)
        shares.append(share[inverse])
        penalties.append(-scores[inverse])
        stricts.append(strict[inverse])
        legacies.append(legacy[inverse])
        records.append({"block": block, "deal_seed": deal_seed, "rotation": rotation,
                        "seats": [NAMES[i] for i in order], "negative_scores": scores.tolist(),
                        "win_share": share.tolist()})
    return {"records": records, "win_share": np.mean(shares, axis=0).tolist(),
            "bullheads": np.mean(penalties, axis=0).tolist(),
            "strict": np.mean(stricts, axis=0).tolist(),
            "legacy": np.mean(legacies, axis=0).tolist(),
            "seconds": [agent.decision_seconds.copy() for agent in agents[:2]]}


def evaluate_parallel(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "games.jsonl").exists():
        raise FileExistsError("Use a new evaluation output directory")
    checkpoints = {name: torch.load(path, map_location="cpu", weights_only=True) for name, path in [("alpha", args.alpha), ("dirv", args.dirv)]}
    for name, checkpoint in checkpoints.items():
        if checkpoint["format"] != FORMAT or checkpoint["kind"] != name:
            raise ValueError("Wrong checkpoint format or kind")
    hashes = provenance()
    hashes[str(Path(__file__).relative_to(ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    protocol = vars(args).copy()
    protocol.update({"format": FORMAT, "paper": REPORT_URL, "source_sha256": hashes,
                     "checkpoint_sha256": {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in [("alpha", args.alpha), ("dirv", args.dirv)]},
                     "training_games": {name: value["games"] for name, value in checkpoints.items()},
                     "independent_unit": "deal seed; four seat rotations per bootstrap group",
                     "timing_note": "Decision wall times collected during concurrent evaluation; report concurrency."})
    dump_json(out / "protocol.json", protocol)
    results, timings = [], [[], []]
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"), initializer=initialize_worker,
                             initargs=(args.alpha, args.dirv, args.alpha_simulations, args.dirv_simulations, args.seed)) as executor:
        with (out / "games.jsonl").open("w", encoding="utf-8") as log:
            for result in executor.map(evaluate_block, range(args.deals), chunksize=1):
                results.append(result)
                for i in range(2):
                    timings[i].extend(result["seconds"][i])
                for record in result["records"]:
                    log.write(json.dumps(record) + "\n")
                log.flush()
                if len(results) % 5 == 0 or len(results) == args.deals:
                    print(json.dumps({"evaluated_games": len(results) * 4, "seconds": round(time.perf_counter() - started, 2)}), flush=True)
    shares = np.array([result["win_share"] for result in results])
    deltas = shares[:, 1] - shares[:, 0]
    ci = bootstrap_interval(deltas, args.seed + 999)
    summary = {"format": FORMAT, "games": args.deals * 4, "independent_deals": args.deals,
               "training_games": protocol["training_games"], "workers": args.workers,
               "seconds": time.perf_counter() - started,
               "win_share": dict(zip(NAMES, shares.mean(axis=0).tolist())),
               "strict_win_rate": dict(zip(NAMES, np.mean([r["strict"] for r in results], axis=0).tolist())),
               "upstream_first_seat_tiebreak_win_rate": dict(zip(NAMES, np.mean([r["legacy"] for r in results], axis=0).tolist())),
               "mean_bullheads": dict(zip(NAMES, np.mean([r["bullheads"] for r in results], axis=0).tolist())),
               "dirv_minus_alpha_pp": float(deltas.mean() * 100), "ci95_pp": [x * 100 for x in ci] if ci else None,
               "decision_seconds": {name: {"mean": float(np.mean(values)), "p95": float(np.quantile(values, .95))} for name, values in zip(NAMES[:2], timings)},
               "interpretation": "Independent method reimplementation. Compare timing budgets separately; no original ETH weights."}
    dump_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    if Path(sys.prefix).name.lower() != "ntw-ai":
        raise RuntimeError("Use the project ntw-ai Python environment")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", required=True)
    parser.add_argument("--dirv", required=True)
    parser.add_argument("--deals", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--alpha-simulations", type=int, default=50)
    parser.add_argument("--dirv-simulations", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.deals, args.alpha_simulations, args.dirv_simulations, args.workers) <= 0:
        parser.error("deals, simulations, and workers must be positive")
    torch.set_num_threads(1)
    evaluate_parallel(args)


if __name__ == "__main__":
    main()
