# Visual Grounding / DESTA-3D

## 2026-10-05：冻结 DeCoTA 正式论文实验已启动，旧全 query corruption 队列已停

[实际执行与接续](docs/DECOTA_PAPER_EXPERIMENTS_EXECUTION.md)、[正式协议](protocols/decota_paper_experiments_v1.md)、[登记与四真实输入 smoke](results/decota_paper_experiments/2026-10-05/REGISTRATION.json)、[基线移植与 Fisher 条件](protocols/decota_paper_baselines_20261005_v1.md)。Table 1 先做自然 clean cross-domain，两个方向、全部 13,785 queries、三固定完整顺序；Ours 共 41,355 到达，已在第一方向实际运行。Source / DINO-Refine / target-trained 参考及 TENT/EATA/SAR 逐方向 runner 已实现并独立 pin，有限接续只等 Ours 退出后串行运行；32 CPU 合同不是基线 GPU 资格或指标。EATA HC 源 2,000 个锁定 train 片段目前本地 178 个，官方连接失败已保存，完整 Fisher 待媒体；不把 ETA 冒充 EATA。

Table 2 的 HC 抽样单位待具体二选一确认：严格每父来源一条为 237 HC／总 969 queries（每在线方法 15,504 到达），覆盖每官方片段一条为 3,482 HC／总 4,214 queries（67,424 到达）；clean＋五类×2.5/5/10% physical burst。早期 all-clip metadata 不代表已批准单位；该表及匹配 one-query 阶段零预测，单位明确后才锁正式名单，Table 1 全 query 继续。后续组件／监督／匹配 temporal／预算／alpha／成本／封存后 GT pipeline 均仍待实际执行。HC2 validation 不改称 test，统计按父来源聚类。方法、生产登记保持原保护边界，当前没有新正式指标。旧 330,840 到达队列在 880 partial 时按新 scope 停止保存，不评分、不恢复。

[磁盘清理](docs/STORAGE_CLEANUP_20261005.md)已完成：311个闲置权重／临时缓存文件，净释放30.36GiB；当前依赖、数据与封存科学记录保留。旧 Sa2VA/PTD 精确重放须恢复退休权重。

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


## New authorized DESTA experiment (running; results pending)

Two independent specialists → native pseudo-targets → R16 latent adaptation. [Readable method and hyperparameters](docs/desta3d/DUAL_EXPERT_NATIVE.md), [27 → 6 → 1 protocol](protocols/desta_dual_expert_native_v1.md). Previous negative results remain; this is not yet an efficacy claim.

## DeCoTA C1–Scale06 online：最新锁定研究入口

[正式 online 版本与代码导航](docs/DECOTA_C1_ONLINE_RELEASE.md)；[空间锁定配置](methods/C1_FINAL_RESEARCH_CONFIG.json)；[原 NLL＋hinge 时间状态](methods/C1_TEMPORAL_RESEARCH_STATUS.json)。

C1 online 每个 query 重置残差和 Adam，只以 1/16 写回空间 LN；原时间适应仍逐 query 丢弃。更早的 `decota_final_simplified_v1` 是 episodic 路径，不能替代 online 状态继承。两份历史 CURRENT 登记保持不变。

[同域 corruption online 评估](docs/TA_DECOTA_C1_SAME_DOMAIN_REVIEW.md)与[新增 Spatial-DeCoTA Direct/critic P0](docs/TA_DECOTA_CRITIC_P0_REVIEW.md)分别列出实际配置、正负结果及审计。P0 是无时间适应、无 LN 继承的当前空间纠错对照，不自动晋升正式配置。
