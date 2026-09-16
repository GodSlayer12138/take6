# 牛头王 · AI 牌桌

本地《6 nimmt! / 牛头王》游戏与 AI 推理项目。网页支持 2—10 人及固定 104 张 / `10n+4` 牌池，当前专项模型适用于二至四人、104 张。

| 场景 | 网页模型 | 运行方式 |
|---|---|---|
| 三人、四人 / 104 张 | `distill2048-specialist2048`，每候选 2,048 世界 | 本机 Python / CUDA |
| 二人 / 104 张 | V6 神经评分与修正策略 | 浏览器 Worker |
| 其他规则 | `neural_hybrid` 兼容策略 | 浏览器 Worker |

默认四人、104 张；对局、AI 建议和竞技场使用同一模型路由。AI 只读取自己的手牌及公开信息。

## 启动

需要 Node.js，以及安装 NumPy、PyTorch 且可使用 CUDA 的 Python 环境。

```powershell
git submodule update --init --recursive  # 获取固定提交的外部参考实现
npm ci
npm run dev       # http://127.0.0.1:5173，自动启动 Python 后端
```

生产运行时执行 `npm run build` 后再执行 `npm start`，网页为 `http://127.0.0.1:8765`；与开发模式择一启动。默认 Python 为 `D:/Programs/miniconda3/envs/ntw-ai/python.exe`，可通过 `NTW_PYTHON` 修改，后端端口可通过 `NTW_API_PORT` 修改。

## 文档

- [最终模型技术说明](docs/TECHNICAL_REPORT.md)：网络、搜索、参数量、权重位置、接口和资源需求。
- [最终评测结果](docs/EVALUATION.md)：三四人冠军、二人 V6 及 take6 对比，包含结论的适用范围。
- [模型索引](models/README.md)：必要文件和完整依赖清单。

## 检查

```powershell
npm test
& 'D:/Programs/miniconda3/envs/ntw-ai/python.exe' -m unittest discover -s server -p 'test_*.py'
& 'D:/Programs/miniconda3/envs/ntw-ai/python.exe' tools/organize_models.py --check
```

权重、冻结运行时及原始评测证据保留在原路径。历史文档已压缩保存于 [documentation-archive](artifacts/documentation-archive/)，不再作为当前阅读入口。

Git 仓库包含模型依赖与评测记录；大型训练数据、编号中间检查点、缓存和日志仅保留在原机器。复测 take6 时先运行 `python tools/fetch_take6.py` 获取固定版本的上游文件。冻结清单仍含原机器路径，跨机器运行需按技术说明处理路径兼容。
