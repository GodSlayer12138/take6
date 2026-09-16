"""Finish the already-running, predeclared ETH experiment and write its report.

This runner does not tune models or choose checkpoints using evaluation wins.
It records a time-matched checkpoint and launches only the predeclared tests.
"""

import io
import json
from pathlib import Path
import subprocess
import sys
import time

from reproduce_eth_dirv import ROOT, dump_json, torch

BASE = ROOT / "artifacts/eth-reproduction"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def latest_progress(name):
    path = BASE / name / "training.jsonl"
    if not path.exists():
        return None
    # Training logs are small; ignore an incomplete last line during a write.
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in reversed(lines[-2:]):
        try:
            value = json.loads(line)
            return {"game": value["game"], "seconds": round(value["seconds"], 1)}
        except json.JSONDecodeError:
            continue
    return None


def capture_time_matched():
    destination = BASE / "paper-dirv-time-matched"
    if (destination / "summary.json").exists():
        return
    alpha_path = BASE / "paper-alpha/summary.json"
    if not alpha_path.exists():
        return
    alpha = read_json(alpha_path)
    # Read an entire atomically-published checkpoint once before inspecting it.
    data = (BASE / "paper-dirv/checkpoint.pt").read_bytes()
    checkpoint = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    ratio = checkpoint["seconds"] / alpha["seconds"]
    if not .95 <= ratio <= 1.05:
        raise RuntimeError(f"Missed time-matched checkpoint window: ratio={ratio:.3f}")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "checkpoint.pt").write_bytes(data)
    dump_json(destination / "summary.json", {
        "kind": "dirv", "games": checkpoint["games"], "seconds": checkpoint["seconds"],
        "alpha_seconds": alpha["seconds"], "training_time_ratio": ratio,
        "selection": "Current saved checkpoint when Alpha finished; selected by elapsed time only, before any formal evaluation",
    })
    print(json.dumps({"time_matched_checkpoint_games": checkpoint["games"], "time_ratio": ratio}), flush=True)


def report():
    alpha = read_json(BASE / "paper-alpha/summary.json")
    dirv = read_json(BASE / "paper-dirv/summary.json")
    matched = read_json(BASE / "paper-dirv-time-matched/summary.json")
    names = [("论文局数：主测试", "paper-evaluation"), ("论文局数：独立复核", "replication-evaluation"), ("相同训练耗时对照", "time-matched-evaluation")]
    results = [(label, read_json(BASE / folder / "summary.json")) for label, folder in names]
    text = ["# ETH 方法复现实验结果", "", "本报告由完整训练与预先声明的评测结果自动生成。模型均为本仓库重新实现和训练，并非 ETH 作者权重。", "",
            "| 实验 | 局数 | Alpha 夺冠份额 | DirV 夺冠份额 | 差值（百分点） | 差值 95% CI |", "|---|---:|---:|---:|---:|---|",
    ]
    for label, result in results:
        lo, hi = result["ci95_pp"]
        text.append(f"| {label} | {result['games']} | {result['win_share']['alpha']:.2%} | {result['win_share']['dirv']:.2%} | {result['dirv_minus_alpha_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    text.extend(["", "并列第一按人数平分胜利份额。每副牌轮换四个座位，置信区间按独立发牌组重采样。", "",
                 "## 训练与时间预算", "",
                 f"- Alpha：{alpha['games']:,} 局，{alpha['seconds'] / 60:.1f} 分钟。",
                 f"- DirV：{dirv['games']:,} 局，{dirv['seconds'] / 60:.1f} 分钟；训练耗时为 Alpha 的 {dirv['seconds'] / alpha['seconds']:.2f} 倍。",
                 f"- 时间对照 DirV：{matched['games']:,} 局，{matched['seconds'] / 60:.1f} 分钟；训练耗时比 {matched['training_time_ratio']:.3f}。", "",
                 "| 实验 | Alpha 平均每步（ms） | DirV 平均每步（ms） | DirV / Alpha | Alpha 平均牛头 | DirV 平均牛头 |", "|---|---:|---:|---:|---:|---:|"])
    for label, result in results:
        a, d = [result["decision_seconds"][key]["mean"] for key in ["alpha", "dirv"]]
        text.append(f"| {label} | {a * 1000:.2f} | {d * 1000:.2f} | {d / a:.3f} | {result['mean_bullheads']['alpha']:.3f} | {result['mean_bullheads']['dirv']:.3f} |")
    text.extend(["", "决策耗时测于 4 个并行评测进程，属于当前运行环境的墙钟观测。每步模拟上限为 Alpha 50、DirV 200。", "", "## 如何解读", ""])
    for label, result in results:
        lo, hi = result["ci95_pp"]
        verdict = "本次结果支持 DirV 的夺冠份额更高。" if lo > 0 else "本次结果支持 Alpha 的夺冠份额更高。" if hi < 0 else "差值区间跨 0，未确认哪方夺冠份额更高。"
        text.append(f"- {label}：{verdict}")
    text.extend(["", "按论文训练局数的比较与相同训练时间的比较应分别解读。前者的 DirV 实际使用了更多训练时间；后者的区间跨 0，因此本轮尚未充分验证相同训练耗时下的优势。",
                 "这些结果不能证明我们精确恢复了论文的 40% 对 29%，也不能证明胜过所有开源 AI。这里只训练了一组独立初始化；更多训练种子的稳定性尚未验证。",
                 "网络结构等未公开细节的假设、与论文的差异及复现命令见 [复现说明](ETH_REPRODUCTION.md)。", "",
                 "[原始研究报告](https://pub.tik.ee.ethz.ch/students/2021-HS/GA-2021-02.pdf)", ""])
    audit_path = BASE / "audit.json"
    if audit_path.exists() and read_json(audit_path)["status"] == "passed":
        text.extend(["## 结果审计", "", "审计通过：12,000 局训练日志及三组共 1,800 局正式评测记录完整，逐局重算胜率与平均牛头一致，种子、轮换和冻结权重/源文件哈希匹配。", "",
                     "审计记录：`artifacts/eth-reproduction/audit.json`；验证记录：`artifacts/eth-reproduction/validation.json`。", ""])
    (ROOT / "docs/ETH_REPRODUCTION_RESULTS.md").write_text("\n".join(text), encoding="utf-8")
    dump_json(BASE / "complete.json", {"status": "complete", "training": {"alpha": alpha, "dirv": dirv, "time_matched": matched},
                                      "evaluations": dict(results), "report": str(ROOT / "docs/ETH_REPRODUCTION_RESULTS.md")})
    print("All predeclared training and evaluation completed. Report: docs/ETH_REPRODUCTION_RESULTS.md", flush=True)


def main():
    started = time.monotonic()
    last_status = 0.0
    while True:
        capture_time_matched()
        if all((BASE / name / "summary.json").exists() for name in ["paper-alpha", "paper-dirv"]):
            break
        if time.monotonic() - last_status > 50:
            print(json.dumps({name: latest_progress(name) for name in ["paper-alpha", "paper-dirv"]}), flush=True)
            last_status = time.monotonic()
        if time.monotonic() - started > 7200:
            raise TimeoutError("Training has not completed within two hours; inspect training sessions before continuing")
        time.sleep(5)
    plan = read_json(BASE / "experiment-plan.json")
    experiments = [("primaryEvaluation", "paper-evaluation", "paper-dirv"),
                   ("secondaryEvaluation", "replication-evaluation", "paper-dirv"),
                   ("timeMatchedEvaluation", "time-matched-evaluation", "paper-dirv-time-matched")]
    # Two simultaneous four-worker evaluations; then the time-matched comparison.
    processes = []
    for key, output, checkpoint_name in experiments:
        spec = plan[key]
        if (BASE / output / "summary.json").exists():
            continue
        if len(processes) == 2:
            for process, log in processes:
                if process.wait() != 0:
                    raise RuntimeError("Evaluation failed; inspect its runner log")
                log.close()
            processes = []
        command = [sys.executable, str(ROOT / "training/evaluate_eth_parallel.py"),
                   "--alpha", str(BASE / "paper-alpha/checkpoint.pt"), "--dirv", str(BASE / checkpoint_name / "checkpoint.pt"),
                   "--deals", str(spec["deals"]), "--seed", str(spec["seed"]),
                   "--alpha-simulations", str(spec["alphaSimulations"]), "--dirv-simulations", str(spec["dirvSimulations"]),
                   "--workers", "4", "--output", str(BASE / output)]
        log = (BASE / f"{output}-runner.log").open("w", encoding="utf-8")
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        processes.append((process, log))
        print(json.dumps({"evaluation_started": output, "pid": process.pid, "games": spec["games"]}), flush=True)
    for process, log in processes:
        if process.wait() != 0:
            raise RuntimeError("Evaluation failed; inspect its runner log")
        log.close()
    report()


if __name__ == "__main__":
    main()
