"""Audit finished ETH experiment summaries against raw records and frozen files."""

import hashlib
import json
from pathlib import Path

from reproduce_eth_dirv import ROOT, np

BASE = ROOT / "artifacts/eth-reproduction"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit():
    plan = read(BASE / "experiment-plan.json")
    checks = {}
    for kind in ("alpha", "dirv"):
        directory = BASE / f"paper-{kind}"
        summary = read(directory / "summary.json")
        records = [json.loads(line) for line in (directory / "training.jsonl").read_text(encoding="utf-8").splitlines()]
        expected = plan["training"][kind]["games"]
        assert summary["games"] == len(records) == expected
        assert [record["game"] for record in records] == list(range(1, expected + 1))
        for record in records:
            assert np.isfinite(record["scores"]).all()
            assert all(np.isfinite(list(loss.values())).all() for loss in record["losses"])
        checks[f"{kind}_training"] = f"{expected} unique completed games, finite scores and losses"
    experiments = [("primaryEvaluation", "paper-evaluation", "paper-dirv"),
                   ("secondaryEvaluation", "replication-evaluation", "paper-dirv"),
                   ("timeMatchedEvaluation", "time-matched-evaluation", "paper-dirv-time-matched")]
    for key, directory_name, checkpoint_name in experiments:
        directory = BASE / directory_name
        summary = read(directory / "summary.json")
        protocol = read(directory / "protocol.json")
        spec = plan[key]
        assert summary["games"] == spec["games"]
        assert protocol["seed"] == spec["seed"]
        assert protocol["alpha_simulations"] == spec["alphaSimulations"]
        assert protocol["dirv_simulations"] == spec["dirvSimulations"]
        for file, expected_hash in protocol["source_sha256"].items():
            assert hashlib.sha256((ROOT / file).read_bytes()).hexdigest() == expected_hash, file
        for name, folder in [("alpha", "paper-alpha"), ("dirv", checkpoint_name)]:
            actual = hashlib.sha256((BASE / folder / "checkpoint.pt").read_bytes()).hexdigest()
            assert actual == protocol["checkpoint_sha256"][name]
        records = [json.loads(line) for line in (directory / "games.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(records) == spec["games"]
        seen = set()
        wins = dict.fromkeys(summary["win_share"], 0.0)
        penalties = dict.fromkeys(wins, 0.0)
        for record in records:
            identity = (record["block"], record["rotation"])
            assert identity not in seen
            seen.add(identity)
            assert record["deal_seed"] == spec["seed"] + record["block"] * 104729
            assert sorted(record["seats"]) == sorted(wins)
            names = ["alpha", "dirv", "random1", "random2"]
            assert record["seats"] == [names[(i + record["rotation"]) % 4] for i in range(4)]
            scores = record["negative_scores"]
            top = max(scores)
            count = scores.count(top)
            for index, (name, score) in enumerate(zip(record["seats"], scores)):
                share = 1 / count if score == top else 0
                assert abs(record["win_share"][index] - share) < 1e-12
                wins[name] += share
                penalties[name] -= score
        assert seen == {(block, rotation) for block in range(spec["deals"]) for rotation in range(4)}
        for name in wins:
            assert abs(wins[name] / len(records) - summary["win_share"][name]) < 1e-10
            assert abs(penalties[name] / len(records) - summary["mean_bullheads"][name]) < 1e-5
        difference = 100 * (wins["dirv"] - wins["alpha"]) / len(records)
        assert abs(difference - summary["dirv_minus_alpha_pp"]) < 1e-10
        checks[directory_name] = f"{len(records)} records verified; summaries, seeds, rotations, source and checkpoint hashes match"
    matched = read(BASE / "paper-dirv-time-matched/summary.json")
    assert .95 <= matched["training_time_ratio"] <= 1.05
    checks["time_matching"] = {"training_time_ratio": matched["training_time_ratio"], "within_five_percent": True}
    result = {"status": "passed", "checks": checks}
    (BASE / "audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    audit()
