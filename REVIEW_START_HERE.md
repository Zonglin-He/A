# DESTA-3D v2 外部代码审阅入口

更新：2026-09-28。当前实验阶段及其独立审计均已完成，GPU 空闲；按用户要求结束定时跟进并交给外部审阅。后续 native-endpoint 源正控只有协议草案，未实现、未登记、未运行。此仓库是代码与汇总证据快照，不是已验证成功的部署方案。

## 先了解当前判断

目标是冻结官方 ParallelTubeDecoding 4B，用 query-conditioned 双支视觉残差改善时空视频定位，并研究无标签测试时适应。**目前没有建立可靠的目标域净收益。** 小开发集上存在 noise 改善、clean 损害及其权衡；后续监督正控说明 CE 的局部下降和 native tube 改善不能直接等同。

请先看 [结构与作用路径](docs/DESTA3D_V2_ARCHITECTURE.md)、[最新实验结果及解释范围](results/desta3d_v2/2026-09-28/README.md)、[必须保留的更正](docs/DESTA3D_V2_CORRECTIONS.md)，再查实际代码。机器可读汇总由已审计原报告提取，保存来源文件 SHA-256；本次没有重跑实验。

## 推荐代码阅读顺序

| 问题 | 实际入口 |
|---|---|
| 双支模块、初始化、FiLM/LN、输出残差 | [desta3d_v2.py](vg_tta/desta3d_v2.py) |
| PTD merger 注入、真实 caption features、原两次 decode | [desta3d_v2_ptd.py](vg_tta/desta3d_v2_ptd.py) |
| 共用 event reference/time，spatial 独立 KV、官方分块 | [desta3d_v2_shared_reference_cached.py](vg_tta/desta3d_v2_shared_reference_cached.py) |
| 源标签支持、A/B 目标和优化组 | [source](vg_tta/desta3d_v2_source.py)、[training](vg_tta/desta3d_v2_training.py) |
| 固定三臂辅助回传对照与正确恢复 | [aux_backflow_run](scripts/desta3d_v2_aux_backflow_run.py)、[recovery_v2](scripts/desta3d_v2_aux_backflow_recovery_v2.py)、[optimizer_checkpoint](vg_tta/optimizer_checkpoint.py) |
| 实际无标签更新目标及白名单 | [tta_objective](vg_tta/desta3d_v2_tta_objective.py)、[tta_pilot](vg_tta/desta3d_v2_tta_pilot.py)、[target8 recovery](scripts/desta3d_v2_tta8_recovery_v2.py) |
| 真实可微 time/coordinate output anchor | [output_anchor](vg_tta/desta3d_v2_output_anchor.py)、[output_anchor_tta](vg_tta/desta3d_v2_output_anchor_tta.py)、[temporal_anchor_tta](vg_tta/desta3d_v2_temporal_anchor_tta.py) |
| 无标签与监督正控、实际 Adam 位移 | [source_task_control](vg_tta/desta3d_v2_source_task_control.py)、[runner v2](scripts/desta3d_v2_source_task_control_v2.py) |
| 只替换 task CE 最终投影的 FP32 helper | [fp32_task_head](vg_tta/desta3d_v2_fp32_task_head.py)、[source16 control](scripts/desta3d_v2_source_fp32_head_control.py) |
| cast / VJP / 完整词表投影诊断 | [cast probe](scripts/desta3d_v2_source_cast_probe.py)、[input VJP](vg_tta/desta3d_v2_input_vjp.py)、[head probe](vg_tta/desta3d_v2_head_probe.py) |
| 评分与二次独立复算 | [有效 source FP32 scorer v2](scripts/score_desta3d_v2_source_fp32_head_control_v2.py)、[crosscheck](scripts/crosscheck_desta3d_v2_source_fp32_control_v2.py) |
| 16 源训练前缀与 native 支持审计 | [CPU audit v2](scripts/audit_desta3d_v2_source_prefix_support16_v2.py) |

文件路径以实际代码为准。`scripts/` 同时保留递归导入的历史工具和失败版本；其存在不代表当前启用。旧 `source_fit/source_continue` 不是现在应续跑的主线。旧 whole-prefix shared-reference 实现未通过 zero-gate 几何等价验收；实际通过的是 `shared_reference_cached`。

## 希望其他 GPT 具体审什么

1. **任务与可更新接口是否匹配。** 66816 FiLM/LN 上的 pre-gate latent/ref/event/moment 目标，与最终 PTD temporal/spatial native 解码之间，哪些联系是代码保证、哪些只是希望？禁止把 loss 降低直接当定位改善。
2. **监督正控是否有效。** 原 GT teacher-forced 全 token CE 与 native 按预测 anchors 构造的时间/框支持不同。NTP/MTP 是位置分组，不能直接称时间/坐标损失。请核原 `joint_loss` 的分母、mask、分块、重算与冻结范围。
3. **数值与离散读出。** 小残差经过 BF16 转换，少量离散变化能明显改变 endpoint argmax；最后输出舍入在单源确有贡献，但只换最终 head 为 FP32 的源16正控更差。请不要据此建议无区别精度、LR、gate、步数网格。
4. **恢复与可复现性。** 检查整数 optimizer key、live Parameter 绑定、actual step、RNG、完整 accumulation window。旧测试只比序列化曾漏掉 silent momentum reset；新测试在真实模块上比恢复后的两次更新。
5. **提出一个有区分力的下一项。** 写清旧证据、新机制、固定条件、正控、过程读出、native 效用、失败分支和资源。当前 native-endpoint 源正控草案可讨论，但不视为已通过。优先指出实际代码行和支持/反证，避免泛泛换 backbone/teacher/PCGrad。

审查时请区分：代码事实、已测结果、解释假设、未执行提案、生产方法。所有历史正负和工程失败都应保留。当前没有证明双支路线不可能，也没有证据足以宣称稳定全正。

## 数据、运行与证据边界

- 源训练是 Vid-only；HC 是 cross-dataset transfer。198 源 val 已用于开发，目标8父源的24条件例也已多次离线 GT 开发曝光。无标签更新不读取 GT，固定终点不按 loss/GT 选态；不能把该目标8称独立未见测试。
- 模型权重、视频、标注、逐样本预测、原始梯度/logits、私有 manifest/授权附件不上传。本地原件保留。只有本仓库无法独立复算所有历史数值，汇总来源 hash 提供对应关系而不是替代 raw evidence。
- 运行环境与第三方版本见 [DEPENDENCIES](docs/DEPENDENCIES.md)。CPU synthetic 检查可运行；完整 GPU runner 依赖本地数据和锁定证据，不能直接克隆后运行，也不要绕过它们的 hash 检查。
- 公开副本只做路径和私有授权引用脱敏，原实验代码和 pins 不改；映射见 [PUBLICATION_UPDATE_20260928.json](docs/PUBLICATION_UPDATE_20260928.json)。历史协议可能引用未打包的本地 artifacts。
- 测试说明见 [REVIEW_VALIDATION](docs/REVIEW_VALIDATION.md)。当前累计 GPU 实测 38820.55116519004 秒，含失败、加载、重放及不重叠 wrapper 时间；本次发布和清理未增加 GPU 实验。

可直接使用 [外部审查提示词](docs/EXTERNAL_REVIEW_PROMPT.md)。
