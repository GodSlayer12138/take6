# 最终模型技术说明

更新：2026-09-15。本文仅描述当前网页使用的模型。最终成绩与对外比较见 [评测结果](EVALUATION.md)。

## 1. 模型与适用范围

| 场景 | 当前模型 | 运行方式 |
|---|---|---|
| 三人、四人 / 104 张 | `distill2048-specialist2048` | 本机 Python / CUDA 搜索 |
| 二人 / 104 张 | V6 神经评分与线性修正 | 浏览器 Web Worker |
| 其他人数或 `10n+4` 牌池 | `neural_hybrid` 兼容策略 | 浏览器 Web Worker |

网页默认四人、104 张；对局、AI 建议和竞技场通过 [ai-router.js](../src/web/ai-router.js) 使用同一路由。推荐范围限于已有评测，不代表所有人数、规则或公开模型中的最强策略。

每局十回合、每人十张牌、四行公共牌。同步暗出后按牌值递增结算，最终牛头最少者胜；第六张牌或低于所有行尾时收行。评测自动选择牛头最少、再最短、再最前的行；网页人类玩家可以自主选行。

## 2. 三四人冠军：网络与搜索

### 网络结构与训练方法

候选评分网络为 `270 → 256 → 128 → 64 → 1`，隐藏层使用 ReLU，单网 **110,593 个参数**。270 维输入包含公共状态与自己手牌 252 维，以及当前候选牌、目标行和风险特征 18 维。输出是评分代价，不是校准后的胜率。

最终策略组合了神经评分、对手行为预测和完整续局搜索。三人续局网络通过搜索轨迹蒸馏训练；四人专项续局网络通过 PPO 自博弈训练。对手预测网络学习固定对手策略在公共局面下的出牌行为，供模拟使用；真实对手身份不会作为输入。

### 决策流程

1. 排除自己的手牌和已公开牌，无放回抽样对手暗牌，构造 **2,048 个假设世界**。
2. 所有合法候选共享这些世界和随机流，模拟打出该牌后的剩余回合。
3. 模拟对手从固定八类型池抽取，包括神经策略、搜索策略的学习近似和随机策略。
4. 三人使用蒸馏续局网络，四人使用专项 PPO 续局网络；根据最终夺冠份额、牛头代价及弱神经先验选择候选。

`2,048` 是每张候选共享的世界数。开局十张候选对应 20,480 条候选续局；只剩一张牌时直接出牌。

| 参数 | 最终值 |
|---|---|
| 引擎 / 设备 | Torch / CUDA |
| 对手模拟模式 | `actual-proxy` |
| 每候选世界数 | 2,048 |
| 神经先验权重 | 0.003 |
| 牛头辅助代价权重 | 0.002 |
| 四人先验 | 使用四人续局网络 |
| 历史似然重加权 / 两张牌计划 / CUDA Graphs | 未启用 |

忽略候选之间相同的常数，选择使下式最小的牌：

```text
utility(a) = -mean_worlds[win_share - 0.002 × future_bullheads]
             + 0.003 × normalized_prior_cost(a)
```

### 参数量与权重位置

十个 PT 依赖的网络参数合计 **1,105,930**，纯 FP32 参数约 **4.42 MB**；这是这组依赖的总量，不表示每一步都调用全部网络。模型文件还可能包含训练状态，磁盘大小不等于推理参数大小。

| 文件或目录 | 用途 |
|---|---|
| [distill2048-specialist2048.json](../artifacts/progressive-upgrades/development-007/distill2048-specialist2048.json) | 最终搜索配置 |
| [归档 progressive_planner.py](../artifacts/progressive-upgrades/development-007/source-snapshot/training/progressive_planner.py) | 正式运行时 |
| [归档 progressive_torch_env.py](../artifacts/progressive-upgrades/development-007/source-snapshot/training/progressive_torch_env.py) | 批量模拟环境 |
| [continuation-v2/model.pt](../artifacts/progressive-upgrades/continuation-v2/model.pt) | 三人续局网络 |
| [ppo-specialist-4-v1/model.pt](../artifacts/progressive-upgrades/ppo-specialist-4-v1/model.pt) | 四人续局网络及先验 |
| [opponent-proxies-v2/](../artifacts/progressive-upgrades/opponent-proxies-v2/) | DirV、Alpha、MCS、冠军搜索的四个行为预测网络 |
| [完整依赖清单](../models/registry.json) | 其余四个 PT 依赖、路径和 SHA-256 |

服务还读取冻结清单中的三个神经对手 JSON。配置、归档运行时和全部依赖需要成组保留；仅复制一个 PT 或 JSON 不能完整部署冠军。

网页启动校验选中策略的配置、运行时、显式权重依赖、共享源代码和三个神经对手，不再要求整批历史候选导出存在。完整研究审计仍使用原清单；执行前需按 [恢复说明](RESEARCH.md) 取回历史资料。

## 3. 二人 V6

V6 以同样的 270 维网络评分为基础，加上 **27 个二人专项修正参数**，按修正后的代价升序选牌：

```text
score(a) = base_network(features(a)) + dot(theta_2p, basis(a))
basis(a) = [x, x × phase, x × score_deficit]  # x 为 9 个候选特征
```

修正项通过以比赛胜率为目标的交叉熵进化优化得到；推理不更新权重，也不进行完整续局搜索。二人实际使用 **110,620 个参数**（基础网络 110,593 + 修正 27）。导出文件还保存其他人数的修正项，当前网页只在二人 / 104 张路由中使用 V6。

权重：[v6/model.json](../artifacts/small-player-exploration/v6/model.json)；执行代码：[small-strategy-runtime.mjs](../scripts/small-strategy-runtime.mjs)。

## 4. 网页接口与信息边界

`src/web/` 的异步 Worker 按人数与牌池选模型。三四人请求 [champion_api.py](../server/champion_api.py)，二人 V6 和兼容策略在浏览器计算。后端串行调度 GPU 请求，启动时验证配置、模型与冻结代码摘要。

| 接口 / 字段 | 含义 |
|---|---|
| `GET /api/ai/health` | 模型身份、设备、世界数、文件摘要和已完成请求数 |
| `POST /api/ai/choose` | 返回合法牌、候选排序、耗时和模型身份 |
| `hand`, `rows`, `seenCards` | 自己的手牌、公开四行及全部公开已见牌 |
| `playerCount`, `deckSize` | 人数和牌池 |
| `scores` | 自己在首位、按座位循环排列的累计牛头数 |
| `seed` | 非负安全整数随机种子 |

当前网页只发送上述七个字段，不发送对手暗牌、对手模型身份或尚未翻开的动作。后端验证牌号、重复牌、手牌与公共牌互斥、人数及回合一致性；当前冠军不使用历史重加权。

服务仅绑定 `127.0.0.1`。前端再次检查模型身份、合法牌和 2,048 世界预算。CUDA 或请求失败时显示错误并允许重试，不自动把随机牌或低预算结果当成冠军建议。

## 5. 运行与资源需求

实测环境：Node 24.13.0、Python 3.11.15、NumPy 2.4.6、PyTorch 2.11.0+cu128、RTX 4060 Laptop 8 GB。本机默认 Python 为 `D:/Programs/miniconda3/envs/ntw-ai/python.exe`，可通过 `NTW_PYTHON` 修改。

```powershell
npm ci
npm run dev    # 开发网页 http://127.0.0.1:5173，自动启动后端

# 生产运行时使用以下两条，与开发模式择一启动
npm run build
npm start      # http://127.0.0.1:8765
```

`NTW_API_PORT` 可修改后端端口。冠军需要可用 CUDA；启动器会管理配套进程，Ctrl+C 停止本次启动的服务。历史清单包含固定路径，跨机器迁移还需要处理路径兼容。

固定 2,048 世界、归档局面预热后的实测：

| 局面 | 决策中位耗时 | 张量显存峰值 | PyTorch 预留峰值 |
|---|---:|---:|---:|
| 三人，10 张手牌 | 233 ms | 507 MiB | 792 MiB |
| 四人，10 张手牌 | 317 ms | 541 MiB | 920 MiB |
| 四人，4 张手牌 | 63 ms | 118 MiB | 188 MiB |

数据来自单独诊断进程，不包含桌面、驱动及其他服务的显存，不是独占机器基准。当前单请求预算下，8 GB 显存容量不是主要限制；网络计算、特征编码、规则模拟与 GPU 调度均有开销。更好的设备可能加速计算，相同模型与预算不会自动提高策略强度。原始数据：[hardware-profile.json](../artifacts/web-integration/hardware-profile.json)。

## 6. 验证

网页路由和 API 已通过测试、开发及生产运行检查；70 个归档决策经直连接口与网页代理分别验证，出牌一致。证据：[validation.json](../artifacts/web-integration/validation.json)。

```powershell
npm test
& 'D:/Programs/miniconda3/envs/ntw-ai/python.exe' -m unittest discover -s server -p 'test_*.py'
& 'D:/Programs/miniconda3/envs/ntw-ai/python.exe' tools/organize_models.py --check
```
