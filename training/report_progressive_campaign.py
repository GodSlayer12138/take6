"""Report completed evaluations and progress, without inspecting pending test outcomes."""
from datetime import datetime
import json
from pathlib import Path
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/progressive-upgrades'


def load(path):return json.loads(path.read_text(encoding='utf-8'))


def pending_development():
    rows=[]
    for path in sorted(OUT.glob('development-*/manifest.json')):
        if (path.parent/'results.json').exists():continue
        manifest=load(path);counts=Counter();trace=path.parent/'blocks.jsonl'
        if trace.exists():
            for line in trace.read_text(encoding='utf-8').splitlines():
                try:counts[json.loads(line)['players']]+=1
                except json.JSONDecodeError:pass
        rows.append(f'| {path.parent.name} | {counts[3]} / {manifest["deals"]} | {counts[4]} / {manifest["deals"]} |')
    if not rows:return []
    return ['## 尚未完成的开发实验（不计升级）','',
            '| 实验 | 三人已完成配对组 | 四人已完成配对组 |','|---|---:|---:|',
            *rows,'','仅报告完整牌局组数；完整开发实验结束后才按冻结规则选候选。','']


def main():
    state=load(OUT/'state.json');n=len(state['accepted_upgrades'])
    lines=['# 三人、四人连续升级：当前进度','',f'更新时间：{datetime.now().astimezone().isoformat(timespec="seconds")}','',
        f'**已验收通过 {n} / 3 次连续升级。** 当前接受基线：`{state["active_baseline"]}`。',
        '', '开发胜率、对手预测准确率、训练收益以及版本号均不计入升级次数。验收规则见 [实验说明](PROGRESSIVE_UPGRADES.md)。', '',
        '## 独立验收', '']
    if not state['certificate_attempts']:lines+=['尚无完成的正式验收。','']
    for attempt in state['certificate_attempts']:
        lines += [f'### 第 {attempt["attempt"]} 次尝试：`{attempt["candidate"]}` — '+('通过' if attempt['passed'] else '未通过'),'',
            '| 人数 | 前版胜率 | 候选胜率 | 提升（百分点） | 95% 差值区间 | 多次尝试校正后的单侧下界 |',
            '|---|---:|---:|---:|---|---:|']
        for p in (3,4):
            rows=attempt['results'][str(p)];old=next(r for r in rows if r['version']=='reference');new=next(r for r in rows if r['version']==attempt['candidate'])
            ci=new['delta_ci95_pp']
            lines += [f'| {p} | {old["win_rate"]*100:.2f}% | {new["win_rate"]*100:.2f}% | {new["delta_pp"]:+.2f} | [{ci[0]:+.2f}, {ci[1]:+.2f}] | {new["adjusted_one_sided_lower_pp"]:+.2f} |']
        lines+=['',f'原始结果：[结果 JSON](../artifacts/progressive-upgrades/{attempt["run"]}/results.json)。每次共 28,672 局，全部回放审计。','']
    lines += pending_development()
    if state.get('pending_certificate'):
        pending=state['pending_certificate'];folder=OUT/pending['run'];counts=Counter()
        trace=folder/'blocks.jsonl'
        if trace.exists():
            for line in trace.read_text(encoding='utf-8').splitlines():
                try:counts[json.loads(line)['players']]+=1
                except json.JSONDecodeError:pass # Concurrent append may expose a partial final line.
        lines += [f'正在运行 `{pending["run"]}`：`{pending["candidate"]}`。', '',
            f'仅报告进度：三人 {counts[3]} / 2,048 组，四人 {counts[4]} / 2,048 组。**不计算未结束验收的胜率或显著性。**','']
    lines += ['## 开发实验（不计升级）','', '| 实验 | 候选 | 三人相对参照提升 | 四人相对参照提升 |', '|---|---|---:|---:|']
    for path in sorted(OUT.glob('development-*/results.json')):
        result=load(path)
        for row in result['results']['3']:
            name=row['version']
            if name=='reference':continue
            other=next(r for r in result['results']['4'] if r['version']==name)
            lines += [f'| {path.parent.name} | {name} | {row["delta_pp"]:+.2f} 点 | {other["delta_pp"]:+.2f} 点 |']
    lines += ['', '不同开发实验的参照、牌局不同，不能用不同实验的原始胜率直接相减。开发候选的区间未经候选选择校正，最终结论只看独立验收。', '',
        '## 已完成训练', '', '| 训练 | 训练状态数或对局数 | 开发验证状态数 |', '|---|---:|---:|']
    for folder in ('opponent-proxies-v1','opponent-proxies-v2','opponent-proxies-v3-devshift','continuation-v1','continuation-v2','continuation-v3-fourp','ppo-proxy-v1',
            'qstudent-v1','ppo-specialist-3-v1','ppo-specialist-4-v1'):
        path=OUT/folder/'complete.json'
        if path.exists():
            result=load(path);lines += [f'| {folder} | {result.get("states",result.get("training_games","—"))} | {result.get("validation_states","—")} |']
    proxy3=OUT/'opponent-proxies-v3-devshift'
    if (proxy3/'protocol.json').exists():
        lines += ['', '## 开发对局中的对手预测训练', '',
            '第三版对手预测从第二版各对手权重继续训练，使用已完成的 development-004 至 007 中抽取的 4,096 局和旧数据。按原始牌局分组划分训练与验证，旧／新来源各占一半训练采样和验证选择权重。训练使用两条 CPU 线程；其权重尚未加入本轮已冻结的 development-008。']
        if (proxy3/'complete.json').exists():
            trained=load(proxy3/'complete.json')
            lines += ['', '| 对手预测模型 | 选中轮次 | 旧验证 NLL（初始→选中） | 新验证 NLL（初始→选中） |', '|---|---:|---:|---:|']
            for name, value in trained['models'].items():
                a,b=value['initial'],value['best']
                lines += [f'| {name} | {value["best_epoch"]} | {a["old"]["nll"]:.4f} → {b["old"]["nll"]:.4f} | {a["new"]["nll"]:.4f} → {b["new"]["nll"]:.4f} |']
            lines += ['', 'NLL 是对出牌概率的预测误差，越低越好。开发验证误差下降不等于比赛胜率提升，不能计为正式升级。']
        else:
            count=len(list(proxy3.glob('*-epoch-??.json')))
            lines += ['', f'训练尚未结束，已保存 {count}/48 个对手训练轮次。该训练不计为已接受升级。']
    continuation3=OUT/'continuation-v3-fourp/complete.json'
    if continuation3.exists():
        trained=load(continuation3);a,b=trained['initial'],trained['best']
        lines += ['', '## 四人强搜索续局蒸馏', '',
            '从四人 PPO 专项网络继续训练，模仿已完成的 development-004、006、007 中共 2,304 局、576 个原始牌局的强搜索行动。训练与验证按整副牌划分，固定 20 轮，并把初始权重纳入选择。', '',
            f'选中第 {trained["best_epoch"]} 轮：验证 NLL 从 {a["nll"]:.4f} 降至 {b["nll"]:.4f}，出牌预测准确率从 {a["accuracy"]*100:.2f}% 变为 {b["accuracy"]*100:.2f}%。', '',
            '该权重保存在 continuation-v3-fourp，尚未加入已冻结的 development-008。模仿指标须通过后续新牌局比赛验证，不能计为正式升级。']
    diagnostic=OUT/'four-player-proxy-diagnostic-001/results.json'
    if diagnostic.exists():
        checked=load(diagnostic)
        lines += ['', '## 四人续局网络的 CPU 开发对照', '',
            '三种网络分别在两套学习得到的预测对手下各打 4,096 局，共 24,576 局。自方座位固定，初始牌局、对手类型和动作随机流配对；这些对手并非正式验收使用的真实搜索程序。', '',
            '| 预测对手池 | 续局网络 | 胜率 | 相对原四人 PPO | 普通配对 95% 区间 |', '|---|---|---:|---:|---:|']
        for pool, rows in checked['results'].items():
            for row in rows:
                lower,upper=row['development_delta_ci95_pp']
                lines += [f'| {pool} | {row["model"]} | {row["win_rate"]*100:.2f}% | {row["delta_pp"]:+.2f} 点 | [{lower:+.2f}, {upper:+.2f}] 点 |']
        lines += ['', '新四人网络在两套预测对手下的点估计均为正，但普通区间都包含零。该结果只作为后续开发线索，不计为正式提升；已冻结的 development-008 保持原配置。']
    for comparison in sorted(OUT.glob('continuation-comparison-*/results.json')):
        protocol=load(comparison.parent/'protocol.json');games=protocol['games_per_model_per_player_count']
        lines+=['',f'## 续局网络的模拟对手开发对照：{comparison.parent.name}','',
            f'以下对手是学习得到的近似模型，不能代替真实本地对手池验收。各网络在每个人数下评测 {games:,} 局。','',
            '| 网络 | 三人胜率 | 四人胜率 |','|---|---:|---:|']
        results=load(comparison)['results']
        for a in results['3']:
            b=next(r for r in results['4'] if r['model']==a['model'])
            lines+=[f'| {a["model"]} | {a["win_rate"]*100:.2f}% | {b["win_rate"]*100:.2f}% |']
    lines += ['', '所有正式结论限定于三人、四人、104 张牌及冻结的八模型对手池。各策略推理预算不同，搜索增益不代表等算力增益。','']
    dest=ROOT/'docs/PROGRESSIVE_UPGRADES_RESULTS.md';dest.write_text('\n'.join(lines),encoding='utf-8')
    print(dest)


if __name__=='__main__':main()
