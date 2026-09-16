"""Run a small frozen development batch without an extra Python worker process."""
import argparse
import json
from pathlib import Path
import time
import progressive_campaign as campaign


def main(args):
    path=Path(args.manifest).resolve();manifest=campaign.load(path)
    assert manifest['purpose'] in ('development','transport-validation')
    campaign.verify(manifest,path.parent)
    name='training/run_progressive_inline.py'
    assert campaign.sha(Path(__file__))==manifest['source_sha256'][name]
    execution=dict(kind='inline-development',source_sha256=campaign.sha(Path(__file__)),
        policy='Frozen archived policies, same schedule and replay audit; one local process to limit commit memory')
    record=path.parent/'execution.json'
    if record.exists():assert campaign.load(record)==execution
    else:campaign.dump(record,execution)
    trace=path.parent/'blocks.jsonl';completed=set()
    if trace.exists():
        for line in trace.read_text(encoding='utf-8').splitlines():
            block=json.loads(line);completed.add((block['players'],block['block']))
    campaign.initialize(str(path));start=time.perf_counter()
    try:
        with trace.open('a',encoding='utf-8') as file:
            for task in campaign.schedule(manifest):
                if task[:2] in completed:continue
                block=campaign.task_work(task);file.write(json.dumps(block)+'\n');file.flush()
                completed.add(task[:2])
                print(json.dumps(dict(run=manifest['name'],blocks=len(completed),total=2*manifest['deals'],
                    players=task[0],seconds=time.perf_counter()-start)),flush=True)
    finally:campaign.arena.BRIDGE.close()
    assert campaign.sha(Path(__file__))==manifest['source_sha256'][name]
    campaign.summarize(path)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);main(parser.parse_args())
