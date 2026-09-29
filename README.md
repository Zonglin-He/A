# Visual Grounding / DESTA-3D

**手动看代码、改超参数，从 [DESTA-3D 工作区](docs/desta3d/README.md) 开始。**

| 入口 | 内容 |
|---|---|
| [快速上手](docs/desta3d/README.md) | 可执行命令、配置复制、输出记录 |
| [参数表](docs/desta3d/HYPERPARAMETERS.md) | 默认值、实际作用、修改位置 |
| [计算图与代码地图](docs/desta3d/CODE_MAP.md) | PTD→共享 THW→双 reader→native tube；mixer/cache 路径 |
| [可编辑配置](configs/desta3d/) | adapter、Local、GlobalMean、R16、pixel views |
| [可读模型代码](desta3d/models.py) | forward、projection、direction loss 与构造 API |
| [手动训练循环](desta3d/train_cached.py) | 显式运行、固定终态、整数 Adam state、日志/seal |
| [研究状态与结果导航](REVIEW_START_HERE.md) | 当前结论、历史路线、公开结果 |

```bash
.venv-ptd-audit/bin/python -B -m desta3d inspect --config configs/desta3d/direction_local.json
.venv-ptd-audit/bin/python -B -m desta3d summary --config configs/desta3d/adapter_tta.json
```

以上只看配置/CPU结构，不加载 PTD 或启动实验。运行环境与私有数据、模型权重不包含在公开仓库中。

本次是代码整理和手动调参工具，不是新方法成绩。DESTA-3D privileged-correction/OPD 的最后 Dev16 gate 已结束，原 NO-GO 与所有正负结果保留；现有生产方法登记不因本工作区改变。

历史入口原文：[旧 README](README_HISTORY_20260929.md)、[旧审阅记录](REVIEW_START_HERE_HISTORY_20260929.md)。本地完整研究档案为 `docs/RESEARCH_HISTORY.md`，实际生产登记为 `methods/CURRENT_METHOD.json`；已有 DeCoTA 精简路径见 [方法入口](methods/decota_final_simplified_v1/README.md)。
