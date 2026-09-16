"""Paired, frozen-policy evaluation of baseline and three trained small-player revisions."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
import numpy as np
import run_local_tournament as arena
from report_local_tournament import replay

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/small-player-iterations'
VERSIONS=['baseline','v2','v3','v4']
OPPONENTS=['ntw-adaptive-rl-v3b','ntw-adaptive-arena-v8-distill','ntw-champion','dirv-10000','alpha-2000','mcs','champion-search','random']

def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def prepare():
    if not (OUT/'training-complete.json').exists():raise RuntimeError('Finish all three versions before evaluating heldout games')
    target=OUT/'evaluation-manifest.json'
    if target.exists():return target
    prior=json.loads((ROOT/'artifacts/local-tournament/manifest.json').read_text(encoding='utf-8'))
    mapping={e['id']:e for e in prior['entrants']}
    entries=[mapping[eid] for eid in OPPONENTS]
    base=dict(mapping['ntw-adaptive5-v1']);base.update(id='baseline',name='原 adaptive5-v1');entries.append(base)
    for version in VERSIONS[1:]:
        path=OUT/version/'model.json'
        entries.append(dict(id=version,name=f'自研 {version.upper()}',kind='model',path=str(path),sha256=sha(path),format='ntw-neural-v1'))
    sources=[Path(__file__),ROOT/'training/run_local_tournament.py',ROOT/'training/report_local_tournament.py',ROOT/'scripts/local-tournament-worker.mjs',ROOT/'training/reproduce_eth_dirv.py']
    sources+=list((ROOT/'src/game').glob('*.js'))+list((ROOT/'src/game/models').glob('*.json'))+list((ROOT/'artifacts/models').glob('*.json'))
    manifest=dict(entrants=entries,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        baseline_checkpoint_sha256=sha(ROOT/'artifacts/models/ntw-adaptive5-v1.pt'),
        candidates={v:json.loads((OUT/v/'summary.json').read_text(encoding='utf-8')) for v in VERSIONS[1:]},
        protocol='Each independent deal samples a frozen opponent group. Substitute baseline,V2,V3,V4 with identical deal seeds, opponent seats and action streams; rotate all seats.',
        scope='2,3,4 players; 104 cards only. Four-player primary; 2/3-player secondary.',
        selection='All candidate checkpoints frozen before first heldout game. No promotion based on development win rate alone.')
    dump(target,manifest);return target

def tasks(plan):
    for mode in ['classic4','classic3','classic2']:
        players=int(mode[-1]);seed=plan['heldout_seeds'][mode]
        rng=np.random.default_rng(seed)
        for block in range(plan['heldout_deals'][mode]):
            opponents=rng.choice(OPPONENTS,players-1,replace=False).tolist()
            yield mode,block,opponents,seed+100000000+block*104729

def run_task(task):
    mode,block,opponents,seed=task;results=[]
    for version in VERSIONS:
        records=arena.run_block((mode,block,0,[version]+opponents,seed))
        for r in records:r['candidate']=version
        results.extend(records)
    return dict(mode=mode,block=block,opponents=opponents,deal_seed=seed,records=results)

def summarize(manifest,plan):
    mode_blocks={mode:{} for mode in plan['heldout_deals']};games=0
    expected={(t[0],t[1]):t for t in tasks(plan)}
    with (OUT/'heldout-blocks.jsonl').open(encoding='utf-8') as f:
        for line in f:
            b=json.loads(line);mode=b['mode'];players=int(mode[-1]);block=b['block']
            assert block not in mode_blocks[mode]
            task=expected[(mode,block)];assert b['opponents']==task[2] and b['deal_seed']==task[3]
            assert len(b['records'])==players*4
            shares=np.zeros(4);scores=np.zeros(4);times=np.zeros(4)
            for vi,version in enumerate(VERSIONS):
                group=[version]+b['opponents']
                for rotation in range(players):
                    r=b['records'][vi*players+rotation]
                    assert r['candidate']==version and r['rotation']==rotation and r['seats']==group[rotation:]+group[:rotation]
                    assert r['deal_seed']==task[3] and r['block']==block and r['mode']==mode
                    replay(r,players,104);games+=1
                    seat=(-rotation)%players
                    shares[vi]+=r['shares'][seat]/players;scores[vi]+=r['bullheads'][seat]/players;times[vi]+=r['decision_seconds'][seat]/players/10*1000
            mode_blocks[mode][block]=(shares,scores,times)
    summaries={}
    for mode,blocks in mode_blocks.items():
        assert len(blocks)==plan['heldout_deals'][mode]
        n=len(blocks);players=int(mode[-1]);x=np.array([blocks[i][0] for i in range(n)])
        penalties=np.array([blocks[i][1] for i in range(n)]);timings=np.array([blocks[i][2] for i in range(n)])
        rng=np.random.default_rng(plan['heldout_seeds'][mode]+999)
        boot=np.empty((20000,4))
        for k in range(0,20000,100):boot[k:k+100]=x[rng.integers(0,n,(100,n))].mean(axis=1)
        ranking=[]
        for vi,version in enumerate(VERSIONS):
            delta=boot[:,vi]-boot[:,0];ci=np.quantile(delta,[.025,.975])*100
            adjusted=np.quantile(delta,[.05/6,1-.05/6])*100
            ranking.append(dict(version=version,games=n*players,win_rate=float(x[:,vi].mean()),ci95=(np.quantile(boot[:,vi],[.025,.975])).tolist(),
                delta_pp=float((x[:,vi]-x[:,0]).mean()*100),delta_ci95_pp=ci.tolist(),delta_familywise95_pp=adjusted.tolist(),
                mean_bullheads=float(penalties[:,vi].mean()),mean_decision_ms=float(timings[:,vi].mean())))
        ranking.sort(key=lambda r:-r['win_rate'])
        for rank,row in enumerate(ranking,1):row['rank']=rank
        summaries[mode]=dict(players=players,independent_deals=n,total_games=n*players*4,ranking=ranking,
            inference='Paired deal-block bootstrap 20,000; familywise intervals use Bonferroni across three candidate-baseline comparisons within this player count.')
        dump(OUT/f'{mode}-results.json',summaries[mode])
    for e in manifest['entrants']:
        if 'sha256' in e:assert sha(e['path'])==e['sha256'],e['id']
    for path,value in manifest['source_sha256'].items():assert sha(ROOT/path)==value,path
    audit=dict(status='passed',replayed_games=games,complete_paired_substitutions=True,every_move_legal=True,
        scores_and_win_shares_recalculated=True,frozen_sources_and_weights=True,trace_sha256=sha(OUT/'heldout-blocks.jsonl'))
    dump(OUT/'evaluation-audit.json',audit)
    report(summaries,manifest,audit)

def report(summaries,manifest,audit):
    lines=['# 自研策略三轮训练结果','',
        '基线为原 adaptive5-v1。新版本只在 2—4 人、104 张规则下训练；全部沿用 270→256→128→64→1 的出牌网络，最终比赛均直接使用网络出牌。','',
        '| 版本 | 训练方式 | 新增训练局数 | 选中更新 |','|---|---|---:|---:|']
    methods={'v2':'四人优势加权微调（AWR）','v3':'四人整局收益 PPO','v4':'2/3/4 人历史对手池 PPO'}
    for version,s in manifest['candidates'].items():lines.append(f"| {version.upper()} | {methods[version]} | {s['total_training_games']:,} | {s['selected_update']} |")
    total=sum(s['total_training_games'] for s in manifest['candidates'].values())
    lines+=['',f'共完成 {total:,} 局新训练。各版本从前一版本开发集选中的检查点继续训练；选中检查点可能早于该版本最终训练步，之后的检查点也已保留。','',
        f"独立评测共 {audit['replayed_games']:,} 局。每副牌使用相同冻结对手，分别替换为原模型及 V2/V3/V4，并轮换所有座位。模型只读取自己的手牌和公共信息。",'',
        '对手池包含历史自研网络、DirV 10,000 局版、Alpha、MCS、冠军搜索和随机策略。胜率为全桌夺冠份额，并列第一平分。该对手池不同于之前的 29 模型联赛，胜率不能直接与旧榜单百分比相减。']
    for mode in ['classic4','classic3','classic2']:
        s=summaries[mode];lines+=['',f"## {s['players']} 人 / 104 张",'',f"每个版本 {s['independent_deals']*s['players']:,} 局；{s['independent_deals']} 组独立牌局。",'',
            '| 版本 | 胜率 | 相对原模型（百分点） | 差值 95% CI | 平均牛头↓ |','|---|---:|---:|---|---:|']
        for r in s['ranking']:
            lo,hi=r['delta_ci95_pp'];lines.append(f"| {r['version']} | {r['win_rate']*100:.2f}% | {r['delta_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {r['mean_bullheads']:.2f} |")
        supported=[r['version'] for r in s['ranking'] if r['version']!='baseline' and r['delta_familywise95_pp'][0]>0]
        lines+=['',('经三次候选比较校正后，确认领先原模型的版本：'+', '.join(supported)+'。') if supported else '考虑三个新版本的多重比较后，本轮尚未确认某个版本显著领先原模型。']
    lines+=['','## 保存与验证','',
        '权重、完整训练参数、随机种子、开发集选择记录、逐局评测动作及审计均位于 `artifacts/small-player-iterations/`。V2、V3、V4 分别提供 `model.pt` 和 `model.json`。','',
        '批量环境已核对 2/3/4 人规则与 270 维特征一致性；验证隐藏手牌不会进入本方特征、训练可重复且参数确实更新。所有正式评测动作独立重放，权重及源文件哈希核验通过。','',
        '四人结果是主要评估；二人、三人作为补充，分别报告。训练过程中只按开发集选择检查点，没有使用最终比赛选训练步。此次未更换游戏默认策略。']
    path=ROOT/'docs/SMALL_PLAYER_ITERATIONS.md';path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    dump(OUT/'complete.json',dict(training_games=total,evaluation_games=audit['replayed_games'],results=summaries,report=str(path)))
    print(json.dumps({'status':'complete','training_games':total,'evaluation_games':audit['replayed_games'],'results':summaries},ensure_ascii=True),flush=True)

def main():
    if Path(sys.prefix).name.lower()!='ntw-ai':raise RuntimeError('Use ntw-ai')
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=8);parser.add_argument('--wait-training',action='store_true');args=parser.parse_args()
    if args.wait_training:
        deadline=time.monotonic()+7200
        while not (OUT/'training-complete.json').exists():
            if time.monotonic()>deadline:raise TimeoutError('Training incomplete')
            time.sleep(5)
    path=prepare();manifest=json.loads(path.read_text(encoding='utf-8'));plan=json.loads((OUT/'experiment-plan.json').read_text(encoding='utf-8'))
    completed=set()
    if (OUT/'heldout-blocks.jsonl').exists():
        with (OUT/'heldout-blocks.jsonl').open(encoding='utf-8') as f:
            for line in f:
                b=json.loads(line);completed.add((b['mode'],b['block']))
    remaining=[t for t in tasks(plan) if (t[0],t[1]) not in completed];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),initializer=arena.initialize,initargs=(str(path),)) as executor:
        with (OUT/'heldout-blocks.jsonl').open('a',encoding='utf-8') as log:
            for i,block in enumerate(executor.map(run_task,remaining,chunksize=1),1):
                log.write(json.dumps(block)+'\n');log.flush()
                if i%8==0:print(json.dumps({'mode':block['mode'],'completed_blocks':i+len(completed),'total_blocks':len(remaining)+len(completed),'seconds':round(time.perf_counter()-start,1)}),flush=True)
    summarize(manifest,plan)

if __name__=='__main__':main()
