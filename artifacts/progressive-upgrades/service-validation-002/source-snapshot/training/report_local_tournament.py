"""Independent trace audit and joint bootstrap of completed local tournaments."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/local-tournament'

def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def bullheads(card):
    if card==55:return 7
    if card%11==0:return 5
    if card%10==0:return 3
    if card%5==0:return 2
    return 1

def replay(record,players,deck):
    cards=(np.random.default_rng(record['deal_seed']).permutation(deck)+1).tolist()
    hands=[set(cards[s*10:s*10+10]) for s in range(players)]
    rows=[[cards[-1-i]] for i in range(4)];scores=[0]*players
    assert len(record['actions'])==10
    for raw in record['actions']:
        actions=[c+1 for c in raw]
        assert len(set(actions))==players
        for seat,card in enumerate(actions):
            assert card in hands[seat];hands[seat].remove(card)
        for seat,card in sorted(enumerate(actions),key=lambda pair:pair[1]):
            eligible=[(row[-1],i) for i,row in enumerate(rows) if row[-1]<card]
            forced=not eligible
            index=max(eligible)[1] if eligible else min(range(4),key=lambda i:(sum(map(bullheads,rows[i])),len(rows[i]),i))
            if forced or len(rows[index])==5:
                scores[seat]+=sum(map(bullheads,rows[index]));rows[index]=[card]
            else:rows[index].append(card)
    assert all(not h for h in hands)
    assert scores==record['bullheads']
    minimum=min(scores);n=scores.count(minimum)
    assert record['shares']==[1/n if s==minimum else 0 for s in scores]

def analyze(mode,manifest):
    folder=BASE/mode;summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    protocol=summary['protocol'];p=5 if mode=='adaptive5' else 4;deck=54 if p==5 else 104
    assert protocol['manifest_sha256']==hashlib.sha256((BASE/'manifest.json').read_bytes()).hexdigest()
    entries=[e for e in manifest['entrants'] if mode in e['modes']];ids=[e['id'] for e in entries];index={eid:i for i,eid in enumerate(ids)}
    epochs=protocol['epochs'];shares=np.zeros((epochs,len(ids)));counts=np.zeros_like(shares,dtype=int)
    assert epochs==manifest['planned_epochs'][mode]
    blocks=set();seat_counts=np.zeros((len(ids),p),dtype=int);pairs=np.zeros((len(ids),len(ids)),dtype=int)
    penalty_sums=np.zeros(len(ids));rank_sums=np.zeros(len(ids));time_sums=np.zeros(len(ids))
    rng=np.random.default_rng(protocol['seed']);expected_orders=[rng.permutation(ids).tolist() for _ in range(epochs)]
    games=0
    with (folder/'blocks.jsonl').open(encoding='utf-8') as source:
        for line in source:
            records=json.loads(line);block=records[0]['block'];epoch=block//len(ids);offset=block%len(ids)
            assert block not in blocks and len(records)==p
            blocks.add(block)
            group=[expected_orders[epoch][(offset+j)%len(ids)] for j in range(p)]
            for rotation,r in enumerate(records):
                assert r['block']==block and r['epoch']==epoch and r['rotation']==rotation and r['mode']==mode
                assert r['deal_seed']==protocol['seed']+100000000+block*104729
                assert r['seats']==group[rotation:]+group[:rotation]
                assert np.isfinite(r['decision_seconds']).all() and min(r['decision_seconds'])>=0
                replay(r,p,deck);games+=1
                ix=[index[eid] for eid in r['seats']]
                shares[epoch,ix]+=r['shares'];counts[epoch,ix]+=1
                penalty_sums[ix]+=r['bullheads'];time_sums[ix]+=r['decision_seconds']
                rank_sums[ix]+=[1+sum(other<score for other in r['bullheads'])+(r['bullheads'].count(score)-1)/2 for score in r['bullheads']]
                for seat,i in enumerate(ix):
                    seat_counts[i,seat]+=1
                    for j in ix:
                        if i!=j:pairs[i,j]+=1
    assert len(blocks)==epochs*len(ids) and np.all(counts==p*p)
    assert np.all(seat_counts==epochs*p)
    shares/=counts
    point=shares.mean(axis=0)
    # One bootstrap unit is an entire independently randomized scheduling epoch.
    # This preserves shared opponents, all seat rotations and between-agent covariance.
    rng=np.random.default_rng(protocol['seed']+777)
    weights=rng.multinomial(epochs,np.full(epochs,1/epochs),size=20000)/epochs
    boot=weights@shares
    ci=np.quantile(boot,[.025,.975],axis=0)
    order=np.argsort(-point,kind='stable');winner=int(order[0]);runner=int(order[1])
    top_probability=np.bincount(np.argmax(boot,axis=1),minlength=len(ids))/len(boot)
    rows=[];original={e['id']:e for e in summary['ranking']}
    for rank,i in enumerate(order,1):
        row=dict(original[ids[i]])
        assert abs(row['win_rate']-point[i])<1e-10
        assert abs(row['mean_bullheads']-penalty_sums[i]/row['games'])<1e-10
        assert abs(row['mean_rank']-rank_sums[i]/row['games'])<1e-10
        assert abs(row['mean_decision_ms']-time_sums[i]/row['games']/10*1000)<1e-8
        row.update(rank=rank,ci95=ci[:,i].tolist(),bootstrap_first_fraction=float(top_probability[i]))
        rows.append(row)
    delta=boot[:,winner]-boot[:,runner]
    # Selection-aware simultaneous intervals for all pairwise differences.
    # max(error)-min(error) bounds the absolute error of every pair at once.
    error=boot-point
    simultaneous_radius=float(np.quantile(error.max(axis=1)-error.min(axis=1),.95))
    selected_top_comparisons=[]
    for i in order[1:]:
        diff=float(point[winner]-point[i])
        selected_top_comparisons.append({'other':ids[i],'delta_pp':diff*100,
            'familywise95_ci_pp':[(diff-simultaneous_radius)*100,(diff+simultaneous_radius)*100]})
    named=['final-agent','default-classic','champion-search','mcs','alpha-2000','dirv-10000','dirv-6200','random']
    key_comparisons=[]
    for a,b in [('dirv-10000','alpha-2000'),('dirv-10000','dirv-6200'),('final-agent','dirv-10000'),('default-classic','dirv-10000')]:
        if a in index and b in index:
            ai,bi=index[a],index[b];interval=np.quantile((boot[:,ai]-boot[:,bi])*100,[.025,.975])
            key_comparisons.append({'a':a,'b':b,'delta_pp':float((point[ai]-point[bi])*100),'ci95_pp':interval.tolist()})
    result={**summary,'ranking':rows,'ci_method':'20,000 joint percentile bootstrap resamples of independently randomized epochs; preserves shared opponents and seat rotations.',
            'independent_epochs':epochs,'winner_vs_runner_up':{'winner':ids[winner],'runner_up':ids[runner],
                'delta_pp':float((point[winner]-point[runner])*100),'ci95_pp':(np.quantile(delta,[.025,.975])*100).tolist(),
                'selection_note':'Point winner selected from this evaluation; use simultaneous intervals for a winner claim.'},
            'all_pairwise_familywise95_radius_pp':simultaneous_radius*100,'selected_top_comparisons':selected_top_comparisons,
            'key_comparisons':key_comparisons,'audit':{'status':'passed','replayed_games':games,'unique_blocks':len(blocks),
                'trace_sha256':hashlib.sha256((folder/'blocks.jsonl').read_bytes()).hexdigest(),
                'protocol_sha256':hashlib.sha256((folder/'protocol.json').read_bytes()).hexdigest(),
                'schedule_and_deal_seeds_match':True,'legal_complete_traces':True,'scores_and_shares_match':True,'equal_games_and_seats':True,
                'opponent_pair_games_min':int(pairs[~np.eye(len(ids),dtype=bool)].min()),'opponent_pair_games_max':int(pairs.max()),
                'unplayed_opponent_pairs':int(np.sum((pairs==0)&~np.eye(len(ids),dtype=bool))//2)}}
    dump(folder/'audited-summary.json',result)
    title='五人 / 54 张' if p==5 else '四人 / 104 张'
    lines=[f'# 本地模型随机赛：{title}','',f"{len(ids)} 个版本，共 {games:,} 局；每个版本 {summary['games_per_entrant']:,} 局。",'',
           '胜率是拿到全桌第一名的份额，并列第一平分。所有对手从本榜参赛池随机分组；每个版本出场次数和各座位次数完全相同。', '',
           f'95% 区间按 {epochs} 个独立随机分组轮次联合重采样 20,000 次，同时保留同组对手和发牌轮换的相关性。名次按点估计排列，接近的名次不等于已证明强弱。', '',
           '各版本使用冻结的现有推理方式：普通权重直接选牌，完整策略保留搜索。项目搜索配置为 18 样本；MCS 每步至多 100 次整局模拟，平均分配给候选牌；Alpha 最多 50 次，DirV 最多 200 次。不是等算力比较。', '',
           'ETH 在五人 54 张模式属于未重新训练的迁移测试；四人 104 张保留训练场景。所有模型统一使用项目吃牌规则（同牛头时取较短行），与 ETH 原实验的同分取第一行略有不同。不给模型提供对手身份或真实手牌。', '',
           '| 排名 | 模型／策略 | 推理方式 | 胜率 | 95% 区间 | 平均牛头↓ | 每步 ms |', '|---:|---|---|---:|---|---:|---:|']
    for r in rows:
        entry=entries[index[r['id']]]
        method='纯网络' if entry['kind']=='model' else '随机' if r['id']=='random' else '搜索'
        lo,hi=r['ci95'];lines.append(f"| {r['rank']} | {r['name']} | {method} | {r['win_rate']*100:.2f}% | {lo*100:.2f}–{hi*100:.2f}% | {r['mean_bullheads']:.2f} | {r['mean_decision_ms']:.2f} |")
    lines+=['','## 主要版本的位置','','| 版本 | 排名 | 胜率 |','|---|---:|---:|']
    for eid in named:
        if eid in index:
            r=next(r for r in rows if r['id']==eid);lines.append(f"| {r['name']} | {r['rank']} | {r['win_rate']*100:.2f}% |")
    lines+=['','## 差距与不确定性','',f"榜首与第二名的点估计差距为 {(point[winner]-point[runner])*100:.2f} 个百分点。",
            f'考虑从全部版本中挑选赢家之后，所有两两差距同时覆盖的 95% 误差半宽约为 {simultaneous_radius*100:.2f} 个百分点。小于这一量级的榜首差距不足以断言唯一最强。','',
            '大量近亲训练分支各占一个参赛名额，因此本榜反映当前版本池的表现；改变对手池、规则或搜索预算可能改变排名。部分完整精度 PyTorch 导出与早期四舍五入的浏览器导出只存在微小参数差异，不应过度解读其名次差别。只有现有冻结权重的比赛抽样误差，未包含重新训练的随机性。','',
            f'审计通过：重放全部 {games:,} 局逐回合出牌，合法性、牛头、胜利份额、随机种子、分组、轮换与出场均衡全部一致。']
    if mode=='classic4':
        lines+=['','`ntw-adaptive5-v1` 是原精度 PyTorch 导出，`ntw-adaptive5` 是浏览器导出；逐参数最大差约 5×10⁻⁸，属于几乎相同的网络。两者的点估计名次差不能当成模型升级收益。']
    (folder/'ranking.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'mode':mode,'audit':result['audit'],'top10':rows[:10],'key_comparisons':key_comparisons},ensure_ascii=False),flush=True)
    return result

def main():
    manifest=json.loads((BASE/'manifest.json').read_text(encoding='utf-8'))
    for path,expected in manifest['source_sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path
    for entry in manifest['entrants']:
        if 'sha256' in entry:assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest()==entry['sha256'],entry['id']
    results={mode:analyze(mode,manifest) for mode in ('adaptive5','classic4')}
    dump(BASE/'protocol-clarifications.json',{
        'mcs_budget':'Frozen manifest mistakenly labels mcs_samples_per_card as 18. Actual frozen source uses mcMax=100 total and floor(100/hand_size) per card; observation.samples is ignored. No runtime behavior, model, seed, or game was changed.',
        'final_confidence_intervals':'Audited reports use the larger independent scheduling epoch as joint bootstrap unit, retaining shared-opponent covariance as well as seat rotations. Raw summary.json retains initial deal-block intervals; audited-summary.json and ranking.md are authoritative.'})
    dump(BASE/'audit.json',{'status':'passed','source_and_weight_hashes_match':True,
        'games':sum(v['total_games'] for v in results.values()),'modes':{k:v['audit'] for k,v in results.items()},
        'analysis_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    lines=['# 本地模型随机比赛结果','',f"完成 {sum(v['total_games'] for v in results.values()):,} 局正式比赛，全部逐回合重放审计通过。",'',
        '- [五人 54 张完整排行榜](../artifacts/local-tournament/adaptive5/ranking.md)',
        '- [四人 104 张完整排行榜](../artifacts/local-tournament/classic4/ranking.md)',
        '- [冻结参赛名单、规则与排除记录](../artifacts/local-tournament/manifest.json)',
        '- [逐局重放审计](../artifacts/local-tournament/audit.json)','']
    for mode,result in results.items():
        lines += ['## '+('五人 54 张' if mode=='adaptive5' else '四人 104 张'),'',
            f"{result['entrants']} 个版本，每个版本 {result['games_per_entrant']:,} 局。",'',
            '| 排名 | 模型／策略 | 胜率 | 95% 区间 |','|---:|---|---:|---|']
        for row in result['ranking'][:10]:
            lo,hi=row['ci95'];lines.append(f"| {row['rank']} | {row['name']} | {row['win_rate']*100:.2f}% | {lo*100:.2f}–{hi*100:.2f}% |")
        lines+=['']
    lines+=['## 解读限制','',
        '胜率按全桌夺冠份额计算，并列第一平分。随机分组、等出场、全座位轮换；95% 区间按完整随机分组轮次联合重采样。', '',
        '这是一份现有本地版本、当前搜索预算下的排行榜。ETH 的 54 张表现属于跨规则迁移；项目最终 54 张策略不能在 104 张场景直接运行，该场景参赛的是游戏已有的 104 张默认路由。', '',
        '普通策略权重直接选牌，成品搜索策略保留搜索，因此不能把本榜解释为相同计算时间下的网络质量排名。', '',
        '近亲训练分支较多，池中每个版本等权；未按算法家族等权。接近的点估计名次没有证明强弱，完整榜单提供多重比较下的误差范围。']
    (ROOT/'docs/LOCAL_MODEL_TOURNAMENT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':main()
