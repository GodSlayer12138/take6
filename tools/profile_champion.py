"""Bounded memory/timing diagnostic of the frozen web champion, without edits."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'server'))
from champion_api import Champion
import torch


def main():
    champion = Champion()
    fixtures = ROOT / 'artifacts/web-integration/archived-observations.json'
    cases = json.loads(fixtures.read_text(encoding='utf-8'))['cases']
    fields = ('hand', 'rows', 'seenCards', 'playerCount', 'deckSize', 'scores', 'seed')
    rows = []
    for players in (3, 4):
        for hand_size in (10, 7, 4, 2):
            case = next(c for c in cases if c['observation']['playerCount'] == players and len(c['observation']['hand']) == hand_size)
            observation = {k: case['observation'][k] for k in fields}
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            cold = champion.choose(observation)
            torch.cuda.synchronize()
            timings = []
            for repeat in range(3):
                started = time.perf_counter()
                result = champion.choose(observation)
                torch.cuda.synchronize()
                timings.append((time.perf_counter() - started) * 1000)
                assert result['card'] == case['archived_card']
            row = dict(players=players, hand_size=hand_size, worlds=2048,
                       milliseconds=timings, median_ms=statistics.median(timings),
                       peak_tensor_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                       after_tensor_bytes=torch.cuda.memory_allocated())
            rows.append(row)
            print(json.dumps(row), flush=True)

    profile_case = next(c for c in cases if c['observation']['playerCount'] == 4 and len(c['observation']['hand']) == 10)
    observation = {k: profile_case['observation'][k] for k in fields}
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.profiler.ProfilerActivity.CUDA in torch.profiler.supported_activities():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=activities) as profiler:
        result = champion.choose(observation)
        torch.cuda.synchronize()
    assert result['card'] == profile_case['archived_card']
    events = profiler.key_averages()
    def item(event):
        return dict(name=event.key, calls=event.count, self_cpu_us=event.self_cpu_time_total,
                    self_device_us=getattr(event, 'self_device_time_total', 0))
    operation_rows = sorted([item(event) for event in events], key=lambda r: -r['self_device_us'])
    cpu_rows = sorted(operation_rows, key=lambda r: -r['self_cpu_us'])
    report = dict(observed_at=datetime.now().astimezone().isoformat(timespec='seconds'),
                  identity=champion.identity, torch=torch.__version__,
                  device_total_bytes=torch.cuda.get_device_properties(0).total_memory,
                  fixture_sha256=hashlib.sha256(fixtures.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='Eight archived states, three timed warm repetitions each. Separate diagnostic process; desktop and web service remain running. No strength or hardware-comparison claim.',
                  measurements=rows, profiler=dict(activities=[str(a) for a in activities],
                  top_device=operation_rows[:20], top_cpu=cpu_rows[:20],
                  interpretation='Operator and CUDA kernel views overlap: do not sum their device times. Synchronization CPU time includes waiting for GPU work. Profiled timings include instrumentation overhead; use measurements for latency.'))
    output = ROOT / 'artifacts/web-integration/hardware-profile.json'
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(output=str(output), top_device=operation_rows[:6], top_cpu=cpu_rows[:6])), flush=True)


if __name__ == '__main__':
    main()
