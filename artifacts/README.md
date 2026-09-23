# 实验与模型产物导航

本目录只跟踪当前模型依赖、必要的冻结运行时和最终评测摘要，保留其原始路径与字节。当前推荐入口见 [模型索引](../models/README.md)，完整依赖见 [registry.json](../models/registry.json)。

| 目录 | 角色 |
|---|---|
| `progressive-upgrades/` | 当前三四人冠军、必要的冻结代码、接受状态和保留的结果摘要 |
| `small-player-exploration/` | 二人 V6、基线与对手索引、最终结果摘要 |
| `small-player-iterations/` | 早期迭代模型；其中 V4 仍是冠军依赖 |
| `models/` | 当前浏览器 AI、冠军及保留基线所需的权重与配置 |
| `web-integration/` | 网页接口验证和资源测量摘要，不是新增胜率证据 |

历史逐局记录、旧候选和重复快照已移出当前版本。需要复核完整实验（包括失败的正式验收）时，按 [恢复说明](../docs/RESEARCH.md) 从原 Git 提交取回，文件恢复到原路径并校验 SHA-256。已有本地未提交数据未删除；新产物和恢复的历史默认被 Git 忽略。
