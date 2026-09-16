"""Fresh paired heldout evaluation for three genuinely different strategy branches."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
import atexit
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys
import time
import run_local_tournament as arena
import evaluate_small_iterations as paired
from explore_small_strategies import ROOT,OUT,dump,sha

VERSIONS=['baseline','v5','v6','v7']
OPPONENTS=paired.OPPONENTS

def configure():
    paired.OUT=OUT;paired.VERSIONS=VERSIONS;paired.report=report

class ExplorationBridge(arena.Bridge):
    def __init__(self,manifest):
        self.process=subprocess.Popen([arena.NODE,str(ROOT/'scripts/small-exploration-worker.mjs'),str(manifest)],cwd=ROOT,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8',bufsize=1,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        atexit.register(self.close)
        assert self.call({'type':'ping'})['ready']

def initialize(path):
    configure();arena.Bridge=ExplorationBridge;arena.initialize(path)

def prepare():
    if not (OUT/'training-complete.json').exists():raise RuntimeError('All branches must finish before heldout evaluation')
    path=OUT/'evaluation-manifest.json'
    if path.exists():return path
    prior=json.loads((ROOT/'artifacts/local-tournament/manifest.json').read_text(encoding='utf-8'))
    mapping={e['id']:e for e in prior['entrants']}
    entries=[mapping[e] for e in OPPONENTS]
    base=dict(mapping['ntw-adaptive5-v1']);base.update(id='baseline',name='原 adaptive5-v1');entries.append(base)
    for v in VERSIONS[1:]:
        model=OUT/v/'model.json'
        entries.append(dict(id=v,name=v,kind='model',format='ntw-small-player-v1',path=str(model),sha256=sha(model)))
    source_paths=set(ROOT/p for p in prior['source_sha256'])
    source_paths.update(ROOT/p for p in ['training/explore_small_strategies.py','training/evaluate_small_exploration.py',
        'training/evaluate_small_iterations.py','training/report_local_tournament.py','training/small_player_env.py','training/train_policy.py',
        'training/test_small_exploration.py','training/audit_small_exploration.py','scripts/small-exploration-worker.mjs','scripts/small-strategy-runtime.mjs'])
    manifest=dict(entrants=entries,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(source_paths)},
        candidates={v:json.loads((OUT/v/'summary.json').read_text(encoding='utf-8')) for v in VERSIONS[1:]},
        plan_sha256=sha(OUT/'experiment-plan.json'),
        scope='2-4 players, 104 cards. Four-player primary; two/three-player secondary.',
        protocol='Same opponent pool as previous paired test, completely new seeds; each deal substitutes baseline and three frozen candidates, with every seat rotation.',
        inference='Actual JavaScript exported policy. V5 performs 25 network evaluations per candidate, V6 one network plus 27-term correction, V7 one specialist network.',
        selection='Only development games used for per-player-count selection; no heldout selection of checkpoints or training parameters.')
    dump(path,manifest)
    snapshot=OUT/'source-snapshot'
    for p in source_paths:
        # Models are hashed in the manifest; archive code separately without duplicating all old weights.
        if p.suffix in ('.py','.js','.mjs'):
            target=snapshot/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
    return path

def report(summaries,manifest,audit):
    candidates=manifest['candidates'];evo=candidates['v6']['training_games'];cf=candidates['v7']
    lines=['# 自研策略多路线探索：V5—V7','',
        '本轮三个分支都从原 adaptive5-v1 出发，各人数设置单独按开发集选择，完成后统一冻结。旧正式测试没有作为本轮测试使用。','',
        '| 版本 | 改变的内容 | 训练／推理预算 |','|---|---|---|',
        '| V5 | 对四行做全部 24 种特征排列，平均预测并与原预测混合 | 不更新网络；每张候选牌 25 次网络计算 |',
        f'| V6 | 用比赛胜率做交叉熵进化优化，分别训练二／三／四人的 27 个修正参数 | {evo:,} 次策略评估对局；推理为一次网络加线性修正 |',
        f"| V7 | 对每个候选出牌做隐藏牌抽样和整局反事实模拟，再训练三个专项网络 | {cf['root_games']:,} 局轨迹产生 {sum(p['public_states'] for p in cf['players']):,} 个公共局面；{cf['completion_rollouts']:,} 次假设续局 |",'',
        'V6 每代的各组参数面对同一批牌和对手，统计中的策略评估对局不是互相独立的发牌样本。V7 的假设续局从中间局面开始，不能把它们称为同数量的新发牌完整训练局。','',
        'V7 丢弃真实对手手牌，只从自己手牌及公开已见牌之外均匀抽取对手手牌；用共同隐藏牌样本比较所有合法候选。该信念模型尚未按对手历史出牌进行条件化。','',
        f"正式测试 {audit['replayed_games']:,} 局，所有动作重放核验通过。对手池含三种历史神经策略、DirV、Alpha、MCS、冠军搜索和随机策略；四人是主要结果，二／三人分别作为补充。",'',
        '胜率指夺冠份额，并列第一平分。置信区间按独立发牌组进行 20,000 次配对 bootstrap；校正区间针对同一人数下三个新候选与基线的比较。V5 使用更多推理计算量，不能把本表解释为等计算预算排名。']
    for mode in ('classic4','classic3','classic2'):
        s=summaries[mode]
        lines+=['',f"## {s['players']} 人 / 104 张",'',f"每版 {s['independent_deals']*s['players']:,} 局，{s['independent_deals']} 组独立牌局。",'',
            '| 模型 | 胜率 | 相对原模型（百分点） | 差值 95% CI | 三比较校正区间 | 平均每次出牌 ms |',
            '|---|---:|---:|---|---|---:|']
        for r in s['ranking']:
            lo,hi=r['delta_ci95_pp'];al,ah=r['delta_familywise95_pp']
            lines.append(f"| {r['version']} | {r['win_rate']*100:.2f}% | {r['delta_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | [{al:+.2f}, {ah:+.2f}] | {r['mean_decision_ms']:.3f} |")
        supported=[r['version'] for r in s['ranking'] if r['version']!='baseline' and r['delta_familywise95_pp'][0]>0]
        lines+=['','经三比较校正后领先原模型：'+(', '.join(supported) if supported else '本轮尚未确认')+'。']
    lines+=['','## 保存与重现','',
        '各分支参数、开发集结果、模型、训练源代码快照、逐局动作与审计保存在 `artifacts/small-player-exploration/`。',
        'V5 是推理集成；V6 提供可解释修正参数；V7 同时提供三个 PyTorch 检查点及按人数选择的浏览器 JSON。',
        '完整方法、局限和运行命令见 [实验协议](SMALL_PLAYER_EXPLORATION_PROTOCOL.md)。']
    path=ROOT/'docs/SMALL_PLAYER_EXPLORATION.md';path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    dump(OUT/'complete.json',dict(evaluation_games=audit['replayed_games'],results=summaries,report=str(path)))
    print(json.dumps({'status':'complete','evaluation_games':audit['replayed_games'],'results':summaries}),flush=True)

def main():
    if Path(sys.prefix).name.lower()!='ntw-ai':raise RuntimeError('Use ntw-ai')
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    configure();path=prepare();manifest=json.loads(path.read_text(encoding='utf-8'))
    plan=json.loads((OUT/'experiment-plan.json').read_text(encoding='utf-8'))
    assert sha(OUT/'experiment-plan.json')==manifest['plan_sha256']
    arena.verify_frozen(manifest)
    completed=set();trace=OUT/'heldout-blocks.jsonl'
    if trace.exists():
        for line in trace.read_text().splitlines():
            block=json.loads(line);completed.add((block['mode'],block['block']))
    remaining=[task for task in paired.tasks(plan) if (task[0],task[1]) not in completed]
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(str(path),)) as pool:
        with trace.open('a',encoding='utf-8') as log:
            for i,block in enumerate(pool.map(paired.run_task,remaining,chunksize=1),1):
                log.write(json.dumps(block)+'\n');log.flush()
                if i%8==0:print(json.dumps({'mode':block['mode'],'completed_blocks':len(completed)+i,'total_blocks':1024,'seconds':round(time.perf_counter()-start,1)}),flush=True)
    paired.summarize(manifest,plan)

if __name__=='__main__':main()
