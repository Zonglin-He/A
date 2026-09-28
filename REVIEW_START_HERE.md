# 最新：三层源可控性诊断已完成并独立核验

时间病例的自由残差与冻结输出列空间控制均把tIoU从29.17%提高至68.06%；空间先暴露坐标类CE与原生全词表argmax的支持差别，保留格式失败后，仅修loss分母得到sIoU59.65→74.88%，列空间控制为73.83%。原模型/门控全部冻结，固定30步、无best选择。

关键限制：列空间控制等效latent改变量为原latent的2698/5310倍；它证明两个源病例存在可达方向，不证明reader容易学会，更不是OPD/target收益。旧mask与监督控制不是纯位置单因素。下一检验同norm的before/after reader和正确/错误mask；尚未运行。

见[完整对照表](results/desta3d_v3/2026-09-28/ACTUATION_CHAIN.md)、[机器聚合及校验](results/desta3d_v3/2026-09-28/ACTUATION_CONTROL.json)、[解释与实现](docs/desta3d_v3/ORACLE_FAILURE_DIAGNOSIS.md)。所有失败、旧oracle负数和原始数据本地保留；公开不含权重、标签、caption、预测raw。GPU累计41976.21386800704秒，cap=null。官方外部权重下载已完成，后续pixel baseline尚待独立smoke/qualification。

## 以下为历史记录，当前状态以本节为准

# 最新：先诊断原生可控性，再推进 latent privilege

2026-09-28 已完成两个源训练诊断样本的自由 merger 残差各30步正控，全PTD/B1参数冻结。时间样本 tIoU29.17→68.06%、vIoU18.67→50.70%；空间坐标受限CE下降但原生格式失败，完整tube按原合同计零。原失败和精确Adam恢复记录保留；不选best、不删负数。这证明一个时间样本可控，尚不能声称空间不可控或3D路线失败。

发现的具体目标差别：原native decoder全词表argmax，而1001-class坐标CE忽略非坐标竞争者。已隔离登记同一空间样本、同30steps/LR，仅改完整词表分母的控制；真实结果待封存二核。原span门未通过，尚未跑span/early-conditioning/OPD。

请先看[完整诊断与竞争解释](docs/desta3d_v3/ORACLE_FAILURE_DIAGNOSIS.md)、[聚合数值及核验](results/desta3d_v3/2026-09-28/ACTUATION_CONTROL.json)、[支持恢复协议](protocols/desta3d_v3_free_actuation_v2.md)、[完整词表单因素协议](protocols/desta3d_v3_free_actuation_full_vocab_v1.md)。7项新增CPU控制通过，和实际GPU效用严格分开。原六臂oracle及全部历史正负继续保留。

官方LLaVA-ST与SigLIP权重已下载并独立hash核验，后续pixel baseline仍待official-loader smoke与资格运行，当前不抢占源诊断。全量source fit仍取消、CURRENT不变；单Luna max半小时只读监测已更新到当前handoff。无target/64/生产晋升。

## 以下为此前完整审阅入口（按时间保留）

# DESTA：特权分支 latent 主线与实际 oracle 结果

更新：2026-09-28。主线候选回到 **共享THW → event/referent readers → 分支特权latent → same-PTD条件策略 → 有条件native-state OPD**。3D是THW而非XYZ，外部模型只提供证据。当前已经实际完成source-only GT oracle四主臂及两个匹配错误证据负控，**尚未建立正确证据优势，不能开始OPD或宣称方法成功**。

先看[主线与文献边界](docs/desta3d_v3/LATENT_PRIVILEGED_OPD.md)、[六臂完整聚合表](results/desta3d_v3/2026-09-28/LATENT_ORACLE.md)、[机器可读结果](results/desta3d_v3/2026-09-28/LATENT_ORACLE.json)、[锁定协议](protocols/desta3d_v3_latent_oracle_v1.md)。同16个Vid源训练父源/同PTD4B+B1/0optimizer；Base不是裸Frozen。sourceGT用于oracle mask，不是无标签TTA。96预测加1全一控制，288独立几何和105根汇总核验通过，6CPU检查通过。

T-oracle主要Δt=-1.785714pp；S-oracle Δs=+.218113pp，但错误位置负控Δs=+.365087pp。正确−错误空间Δs=-.146974pp，CI跨0；13个可区分时间负控的正确−错误Δt=+.053605pp，CI跨0。局部正负与负尾均保留，不据小涨包装正确性方向。注入和策略logits实际发生变化，全一及分支隔离控制通过。

[实际调制](vg_tta/desta3d_v3_latent_oracle.py)、[runner](scripts/desta3d_v3_latent_oracle.py)、[独立scorer](scripts/score_desta3d_v3_latent_oracle.py)、[root masks/injection/KL](scripts/audit_desta3d_v3_latent_oracle_raw.py)、[root汇总](scripts/crosscheck_desta3d_v3_oracle_summary.py)。空间比较固定Base reference/interval/anchors；T/TS可能改变条件，不能把最终sIoU当固定条件空间收益。

外部LLaVA-ST及SigLIP下载曾按用户停止，后经用户明确确认恢复，目的是补 **pixel Stage B baseline**，不是恢复全源训练。官方loader/decode与teacher资格GPU仍待完整权重/hash/注册。full-source fit仍取消于140query occurrences/35steps，checkpoint仅历史证据。下一按条件分别验teacher advantage、native参数可控性、无privilege最终OPD效用，三者不能替代。用户单Luna max每30min监测，异常由根处理；不自动target/64/生产晋升/网格。

公开仅代码、协议、结构和聚合结果；无权重、视频、源标签、caption、逐样本预测或raw。累计GPU实测40122.33975609803秒cap=null，全部失败/加载/重放保持计费。

## 以下保留之前外部资格修复的历史审阅入口

# DESTA-3D 双分支与外部证据 OPD 审阅入口

更新：2026-09-28，针对 main439e6f2 的外部审阅落实五项修复。full-source fit 已取消于140 query occurrences /35个完整Adam步；checkpoint只留来源证据，不用于方法比较/teacher/OPD。当前主线为 **LLaVA-ST证据 → 同PTD时空特权视图 → 资格通过后另登记native正控 → 条件性OPD**。尚无新teacher GPU结果。用户随后要求单Luna max每30分钟监测下载/已启动阶段，异常由主代理处理；CURRENT和旧队列不变。

先看[当前结构与状态](docs/desta3d_v3/EXTERNAL_PRIVILEGED_OPD.md)、[修正版资格协议v2](protocols/desta3d_v3_external_privileged_opd_v2.md)和[机器状态](results/desta3d_v3/2026-09-28/ROUTE_SWITCH_STATUS.json)。v1错误时间合同已保留并明确替代，未用于任何真实teacher GPU评分。

|本次修复|实现及验证状态|
|---|---|
|100槽物理时间|[views](vg_tta/external_privileged_views.py)：uniform physical clip time选最近原观测；u=.5在[10,11,90]映射50，不依赖重复像素|
|many-to-one/局部坏框|同模块：逐帧median；保留raw、duplicates、pairwise IoU、dispersion；局部fallback，coverage/max gap诊断；不新增gap阈值|
|官方loader/decode smoke|[smoke](scripts/desta3d_v3_external_loader_smoke.py)：两固定源输入、官方temp.01 vs greedy、显式max_frame100、真实official loader对照；权重下载中，GPU未验收|
|完整Stage B|[runner](scripts/desta3d_v3_privileged_ptd_qualification.py)：原图/T/S/TS四视图同PTD4B+B1，完整64新预测封存后[独立scorer](scripts/score_desta3d_v3_privileged_qualification.py)/[root复算](scripts/crosscheck_desta3d_v3_privileged_summary.py)；代码及合成流程就位，实际结果待跑|
|native控制scope|[新隔离helper](vg_tta/desta3d_v3_native_scopes_v2.py)：L2加入branch QueryPool、event temporal最后pointwise；L3共享projection/conv且norm_stem冻结；真实native控制尚未跑|

17项CPU检查通过，含真实hidden128可更新范围、逐帧无效证据、独立几何、合成16父源封存/评分/root汇总、坏seal在读标签前拒绝。不是GPUteacher资格。Q0固定16源训练父源；官方ST-Align stage3列有VidOR来源，不能声称teacher-unseen。未建立provenance-clean Q1。当前不实现OPD optimizer。

260.159GB不再需要的权重/视频包/缓存已按用户授权清理，科学raw和正负报告保留；历史精确重放若依赖已删资源须重建。

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
5. **提出一个有区分力的下一项。** 写清旧证据、新机制、固定条件、正控、过程读出、native 效用、失败分支和资源。当前外部teacher资格与native分支正控可审查，但不视为已通过。优先指出实际代码行和支持/反证，避免泛泛换 backbone/teacher/PCGrad。

审查时请区分：代码事实、已测结果、解释假设、未执行提案、生产方法。所有历史正负和工程失败都应保留。当前没有证明双支路线不可能，也没有证据足以宣称稳定全正。

## 数据、运行与证据边界

- 源训练是 Vid-only；HC 是 cross-dataset transfer。198 源 val 已用于开发，目标8父源的24条件例也已多次离线 GT 开发曝光。无标签更新不读取 GT，固定终点不按 loss/GT 选态；不能把该目标8称独立未见测试。
- 模型权重、视频、标注、逐样本预测、原始梯度/logits、私有 manifest/授权附件不上传。本地原件保留。只有本仓库无法独立复算所有历史数值，汇总来源 hash 提供对应关系而不是替代 raw evidence。
- 运行环境与第三方版本见 [DEPENDENCIES](docs/DEPENDENCIES.md)。CPU synthetic 检查可运行；完整 GPU runner 依赖本地数据和锁定证据，不能直接克隆后运行，也不要绕过它们的 hash 检查。
- 公开副本只做路径和私有授权引用脱敏，原实验代码和 pins 不改；映射见 [PUBLICATION_UPDATE_20260928.json](docs/PUBLICATION_UPDATE_20260928.json)。历史协议可能引用未打包的本地 artifacts。
- 测试说明见 [REVIEW_VALIDATION](docs/REVIEW_VALIDATION.md)。当前累计 GPU 实测 39977.51307785203 秒，含失败、加载、重放及不重叠 wrapper 时间；最新累计包含后续已取消的全源fit；新外部teacher代码发布、清理与下载没有新增GPU推理。

可直接使用 [外部审查提示词](docs/EXTERNAL_REVIEW_PROMPT.md)。
