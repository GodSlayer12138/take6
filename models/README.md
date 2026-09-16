# 最终模型文件索引

本目录只保存索引；权重和冻结依赖保持原路径。

| 场景 | 策略入口 | 必需运行环境 |
|---|---|---|
| 三人、四人 / 104 张 | [distill2048-specialist2048.json](../artifacts/progressive-upgrades/development-007/distill2048-specialist2048.json) | Python / CUDA 及配置对应的归档运行时 |
| 二人 / 104 张 | [v6/model.json](../artifacts/small-player-exploration/v6/model.json) | [small-strategy-runtime.mjs](../scripts/small-strategy-runtime.mjs) / 浏览器 Worker |
| 其他网页规则 | [ai-core.js](../src/game/ai-core.js) 的 `neural_hybrid` 路由 | 浏览器 Worker 及其 JSON 依赖 |

三四人冠军包含十个 PT 依赖、搜索配置和归档模拟代码，必须成组保留；V6 的 JSON 可供二人策略直接推理。具体结构、参数与路径见 [技术说明](../docs/TECHNICAL_REPORT.md)。

[registry.json](registry.json) 记录 10 个逻辑条目、70 个不重复依赖文件及 SHA-256，包含主力所需组件和核验对照；[inventory.json](inventory.json) 保存文件盘点。原始权重、训练数据和评测证据不因文档精简而删除。

take6 原始对照权重单独保留在 [external/take6/trained-anns/](../external/take6/trained-anns/)，未接入网页默认模型；最终对比见 [评测结果](../docs/EVALUATION.md)。

在仓库根目录核验依赖：

```powershell
& 'D:/Programs/miniconda3/envs/ntw-ai/python.exe' tools/organize_models.py --check
```
