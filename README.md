> 2026-09-28 更新：全量源训练已按用户要求停止，新外部证据路线请从 [REVIEW_START_HERE.md](REVIEW_START_HERE.md) 和 [结构图](docs/desta3d_v3/EXTERNAL_PRIVILEGED_OPD.md) 开始。代码/CPU接口已实现，teacher权重下载中，OPD效用未测。

# A — STVG adaptation research code

**2026-09-28 最新审阅包已更新。** 从 [REVIEW_START_HERE.md](REVIEW_START_HERE.md) 开始；包含当前 DESTA-3D v2 的代码、结构、单因素对照、失败更正和未解问题。

历史实验和独立审计保留。当前正在准备首次外部teacher及四视图PTD资格验证；五项审阅修复和16项CPU控制已完成，官方权重下载中，尚无teacher GPU资格结果。用户授权Luna max每30分钟检查并由主代理处理异常。尚未建立可靠目标TTA净收益，也未晋升生产方法。

| 阅读内容 | 入口 |
|---|---|
| 推荐阅读顺序与具体审查问题 | [REVIEW_START_HERE](REVIEW_START_HERE.md) |
| 计算图、训练和实际更新/冻结路径 | [结构说明](docs/DESTA3D_V2_ARCHITECTURE.md) |
| 完整正负汇总与解释范围 | [2026-09-28结果](results/desta3d_v2/2026-09-28/README.md) |
| 优化器恢复事故及其他版本更正 | [CORRECTIONS](docs/DESTA3D_V2_CORRECTIONS.md) |
| 可复制给其他GPT的提示词 | [EXTERNAL_REVIEW_PROMPT](docs/EXTERNAL_REVIEW_PROMPT.md) |
| CPU检查和公开范围 | [REVIEW_VALIDATION](docs/REVIEW_VALIDATION.md) |
| 外部依赖与固定版本 | [DEPENDENCIES](docs/DEPENDENCIES.md) |
| 源码到公开副本的hash映射 | [PUBLICATION_UPDATE_20260928](docs/PUBLICATION_UPDATE_20260928.json) |
| 既有DeCoTA精简模块 | [decota_final_simplified_v1](methods/decota_final_simplified_v1/) |

此仓库不包含视频、数据集、标注、权重、私有manifest/授权附件、逐样本预测或raw梯度/logits。原科研证据在本地保留；完整GPU runner依赖这些原件及严格pins，仅克隆不能重跑全实验，不要禁用其完整性检查。

## CPU模块使用

```bash
python -m pip install -r requirements-core.txt
python examples/desta3d_v2_cpu.py
python -m pytest -q tests/test_desta3d_v2.py tests/test_desta3d_v2_training.py tests/test_desta3d_v2_tta_objective.py
```

CPU示例使用合成特征，不是视频定位demo或任务精度结果。更完整的本次CPU测试见验证记录。

历史材料继续保留：[2026-09-27结果](results/desta3d_v2/2026-09-27/README.md)、[旧README快照](docs/README_SNAPSHOT_20260927.md)。旧“源训练running、target TTA未开始”和连续optimizer说明已过时；以上最新入口与更正优先。递归依赖里有历史和失败runner，它们不是当前方法或自动续跑指令。
