# 训练与历史资料

当前版本保留网页游戏、现有 AI、全部训练与评测代码、模型索引列出的 70 个依赖文件以及最终评测摘要。旧候选模型、逐局对战记录、训练日志、重复快照和旧文档不再放在默认工作区；清理前已提交的文件仍可从 Git 历史按原字节恢复。

## 继续训练

训练脚本仍位于 `training/`，数据生成脚本仍位于 `scripts/`。例如，生成训练数据并训练一个新的候选模型：

```powershell
npm run data:train -- --output artifacts/training/new-train.jsonl
npm run train:gpu -- --train artifacts/training/new-train.jsonl --output artifacts/models/new-candidate.json --checkpoint artifacts/models/new-candidate.pt
```

新模型应使用新路径，避免覆盖冻结模型。训练结果不会自动替换网页使用的冠军；接入新版本需要评测、更新策略配置和依赖索引。

## 恢复旧实验

```powershell
# 查看历史文件分组和大小，不恢复文件
python tools/restore_research.py

# 恢复一个目录或文件；支持多个路径
python tools/restore_research.py artifacts/progressive-upgrades/development-008

# 完整重放旧实验、续跑原实验或审计原清单前，恢复全部历史依赖
python tools/restore_research.py --all
```

恢复清单为 `models/history.json`，包含原提交、路径、字节数和 SHA-256。恢复工具逐个校验内容，拒绝覆盖已修改文件，不会切换分支或修改 Git 索引。旧清单会引用跨目录依赖，单独恢复一个目录不保证足以重放整个实验；完整研究审计优先使用 `--all`。

浅克隆若没有原提交，工具会提示 `git fetch origin <原提交>`。保留原仓库远端或含该提交的 Git 副本，才能持续恢复。未提交的本地训练数据没有 Git 备份，本次清理未删除它们；下载的 take6 文件仍用 `tools/fetch_take6.py` 获取。

`server/verify_web_champion.py` 依赖 `development-008` 的原始逐局记录，运行前需恢复该目录。网页日常推理、JavaScript 测试、服务接口测试及 `tools/organize_models.py --check` 无需历史记录。

## 后续文件管理

`artifacts/` 新文件默认被忽略，防止实验产物重新堆进仓库；已跟踪的运行依赖和结果摘要仍正常跟踪。正式发布新模型时，先更新并核验 `models/registry.json`，再用 `git add -f <需要发布的文件>` 明确加入版本控制。恢复的历史资料无需再次提交。

完整本地盘点需要时运行 `python tools/organize_models.py --inventory`，生成的 `models/inventory.json` 不再提交。

本次精简缩小当前版本和后续构建上下文，不改写提交历史，因此现有 `.git` 体积不会同步减小。新环境可使用浅克隆减少历史下载；完整克隆仍会下载旧模型。已有本地训练数据与 `node_modules/`、`dist/` 不计入精简后的已跟踪文件体积。
