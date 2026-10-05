# Visual Grounding / DESTA-3D

## 2026-10-05：固定 DeCoTA 全量同域／跨域 corruption 正在运行

[实际启动与边界](docs/TA_DECOTA_FIXED_FULL_LAUNCH.md)、[协议](protocols/decota_fixed_full_corruption_v1.md)、[固定配置](methods/DECOTA_FIXED_FULL_RESEARCH_CONFIG.json)、[公开登记](results/decota_fixed_full_corruption/2026-10-05/REGISTRATION.json)。Native-WHEN + Uniform4 + 单 DINO admitted Frame-Top1 + joint1792 Adam .03 + LN delta/16；全 VidSTG test 10303query/732源及 HC2 val 3482query/237源，四种 checkpoint→target、clean+五5%、双序，共330840在线到达。八个真实 smoke 和独立 CPU 算术通过；**没有本轮完整指标，GT 仍等待全封存**。完整 pipeline 诊断与结果公开是必须的收尾。原暂停队列未恢复；生产登记未改。

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
