# 牛头王 · AI 牌桌

本地《6 nimmt! / 牛头王》游戏与 AI 推理项目。网页支持 2—10 人及固定 104 张 / `10n+4` 牌池，当前专项模型适用于二至四人、104 张。

模型展示名称为 **牛头王·推演版、牛头王·双人版、牛头王·通用版**，下表使用简称；内部策略编号见 [名称对照](docs/TECHNICAL_REPORT.md#名称与内部编号)。

| 场景 | 网页模型 | 运行方式 |
|---|---|---|
| 三人、四人 / 104 张 | 推演版 | 本机 Python / CUDA |
| 二人 / 104 张 | 双人版 | 浏览器 Worker |
| 其他规则 | 通用版 | 浏览器 Worker |

默认四人、104 张；对局、AI 建议和竞技场使用同一模型路由。AI 只读取自己的手牌及公开信息。

## 启动

需要 Node.js，以及安装 NumPy、PyTorch 且可使用 CUDA 的 Python 环境。

```powershell
git submodule update --init --recursive  # 获取固定提交的外部参考实现
npm ci
npm run dev       # http://127.0.0.1:5173，自动启动 Python 后端
```

生产运行时执行 `npm run build` 后再执行 `npm start`，网页为 `http://127.0.0.1:8765`；与开发模式择一启动。默认 Python 为 `D:/Programs/miniconda3/envs/ntw-ai/python.exe`，可通过 `NTW_PYTHON` 修改，后端端口可通过 `NTW_API_PORT` 修改。

纯静态发布使用 `npm run build:static`，将 `dist/` 部署到静态托管平台；仓库的 Vercel 配置已使用该命令。此版本无需 Python 或 GPU，二人 104 张使用双人版，其余规则使用通用版。推演版的评测成绩不适用于纯静态版三四人 AI。

## 文档

- [最终模型技术说明](docs/TECHNICAL_REPORT.md)：网络、搜索、参数量、权重位置、接口和资源需求。
- [最终评测结果](docs/EVALUATION.md)：推演版、双人版及 take6 对比，包含结论的适用范围。
- [模型索引](models/README.md)：必要文件和完整依赖清单。
- [训练与历史资料](docs/RESEARCH.md)：继续训练、按需恢复旧实验及仓库体积说明。

## 检查

```powershell
npm test
& 'D:/Programs/miniconda3/envs/ntw-ai/python.exe' -m unittest discover -s server -p 'test_*.py'
& 'D:/Programs/miniconda3/envs/ntw-ai/python.exe' tools/organize_models.py --check
```

仓库保留游戏、训练代码、模型索引中的依赖和最终评测摘要。旧候选模型、逐局记录、重复源代码快照及历史文档已从当前版本移除，可用 `python tools/restore_research.py --all` 从清理前的 Git 提交恢复；也可指定目录按需恢复。大型本地训练数据未删除，新实验产物默认不提交。

复测 take6 时先运行 `python tools/fetch_take6.py` 获取固定版本的上游文件。冻结清单仍含原机器路径，跨机器运行需按技术说明处理路径兼容。
