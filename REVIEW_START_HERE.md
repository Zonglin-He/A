# 代码与研究审阅入口

## 先读代码

手动调参入口是 [`docs/desta3d/README.md`](docs/desta3d/README.md)。核心 adapter 保留在 [`vg_tta/desta3d_v2.py`](vg_tta/desta3d_v2.py)，新可编辑工作区在 [`desta3d/`](desta3d/)，参数集中于 [`configs/desta3d/`](configs/desta3d/)。

建议阅读顺序：

1. [计算图与文件地图](docs/desta3d/CODE_MAP.md)：分清 adapter、Joint mixer、DirectionMixer 和 pixel evidence。
2. [参数表](docs/desta3d/HYPERPARAMETERS.md)：尤其输入 feature_dim 与 hidden_dim、radius 与 loss 的关系。
3. [`models.py`](desta3d/models.py) 的 `DirectionMixer.forward` 和 `direction_loss`。
4. [`train_cached.py`](desta3d/train_cached.py) 的 `_fit`：等 query 梯度累积、Adam counters、末态保存。
5. [`test_desta3d_workbench.py`](tests/test_desta3d_workbench.py)：CPU 等价与输入/输出合同。

新入口只在明确执行 `fit-cache` 时训练小 predictor，没有自动 PTD native、external teacher、OPD、target 或后续队列。Local/Global/R16 preset 保留原实验起点供人工审阅，不代表恢复已停止的调参链。

## 当前科学状态（2026-09-29）

最后一次 external evidence→same-PTD Dev16 qualification 已完成并独立审计。B1 的 vIoU 为23.041295%，TS privileged policy 为19.079401%，Δv为−3.961893pp，CI跨0；按用户预锁 gate 停止该 privileged-correction/OPD 主线投入。此结论不撤回 Full/R16 oracle 的正证据，也不泛化成所有 latent correction 不可能。

详细指标、正负尾、两个工程修复与不确定性见 [最终报告](https://github.com/Zonglin-He/A/blob/6b0cce3e4defd982ab63407107b9cba159429ab9/results/desta3d_v3/2026-09-29/EXTERNAL_POLICY_GATE.md)。代码整理没有新增 GPU 研究测量。

## 历史与复现

旧审阅页完整快照在 [REVIEW_START_HERE_HISTORY_20260929.md](REVIEW_START_HERE_HISTORY_20260929.md)。旧 `scripts/` 和 `protocols/` 仍是各自已锁实验的原件；不要把其中早期“running / next”文本当成当前队列状态。

公开仓库只含代码、协议和匿名结果；媒体、标签、cache、权重、预测 raw 留在本地。本地最高优先级状态入口为 `docs/RESEARCH_HISTORY.md`，生产方法由 `methods/CURRENT_METHOD.json` 决定。

## New authorized DESTA experiment (running; results pending)

Two independent specialists → native pseudo-targets → R16 latent adaptation. [Readable method and hyperparameters](docs/desta3d/DUAL_EXPERT_NATIVE.md), [27 → 6 → 1 protocol](protocols/desta_dual_expert_native_v1.md). Previous negative results remain; this is not yet an efficacy claim.
