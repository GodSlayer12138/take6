"""Update only the final take6 result table; preserve concise model documentation."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/take6-evaluation'
TITLES = {
    'four104_project': '四人 / 104 张，本项目规则',
    'four104_take6_rules': '四人 / 104 张，take6 选行规则',
    'two24_native': '二人 / 24 张，take6 原生规则',
    'two104_transfer': '二人 / 104 张，take6 迁移',
}
NAMES = {
    'distill2048-specialist2048': '当前冠军',
    'web-adaptive2': '网页兼容 AI',
    'v6': 'V6',
}
START = '<!-- take6-results:start -->'
END = '<!-- take6-results:end -->'


def main():
    result = json.loads((OUT/'results.json').read_text(encoding='utf-8'))
    audit = json.loads((OUT/'decision-audit.json').read_text(encoding='utf-8'))
    assert result['audited_games'] == audit['games'] == 1280 and audit['decisions'] == 12800
    assert audit['status'] == 'passed' and audit['complete']
    assert result['trace_sha256'] == audit['trace_snapshot_sha256']
    assert result['protocol_sha256'] == audit['protocol_sha256']
    path = ROOT/'docs/EVALUATION.md'
    text = path.read_text(encoding='utf-8')
    if text.count(START) != 1 or text.count(END) != 1 or text.index(END) < text.index(START):
        raise ValueError('Final evaluation document must contain one ordered result section')
    lines = [START,
        '固定计划共 **1,280 局**。四人主比较同桌为当前冠军、take6、DirV 本地复现和 MCS 适配版。', '',
        '| 场景 | 本地模型 | 本地夺冠份额 | take6 夺冠份额 | 本地减 take6（百分点） | 差值 95% 区间 |',
        '|---|---|---:|---:|---:|---|']
    summaries = result['summaries']
    for mode, title in TITLES.items():
        s = summaries[mode]
        ranking = {r['id']:r for r in s['ranking']}
        a,b = ranking[s['ours']], ranking[s['take6']]
        lo,hi = s['delta_ci95_pp']
        lines.append(f"| {title} | {NAMES[s['ours']]} | {a['first_share']*100:.2f}% | {b['first_share']*100:.2f}% | {s['delta_first_share_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    primary = summaries['four104_project']
    lo,hi = primary['delta_ci95_pp']
    if lo > 0:
        conclusion = '四人主比较的差值区间高于零，支持本地冠军在该固定对手池和规则下领先 take6'
    elif hi < 0:
        conclusion = '四人主比较的差值区间低于零，支持 take6 在该固定对手池和规则下领先本地冠军'
    else:
        conclusion = '四人主比较的差值区间跨零，本轮不能确认本地冠军与 take6 的差异'
    others = {r['id']:r for r in primary['ranking']}
    lines += ['', conclusion + f"；同桌 DirV、MCS 的夺冠份额分别为 {others['dirv-10000']['first_share']*100:.2f}%、{others['mcs']['first_share']*100:.2f}%。", END]
    start,end = text.index(START),text.index(END)+len(END)
    updated = text[:start]+'\n'.join(lines)+text[end:]
    if updated != text:
        path.write_text(updated,encoding='utf-8')
    print(json.dumps(dict(status='reported',document=str(path.relative_to(ROOT)),games=1280)))


if __name__ == '__main__':
    main()
