# 最新：Joint mixer两seed确认完成，当前配置未建立teacher优势

两个seed各完成618query/95源父源/155actualAdam。新锁31源父源全部447query，B1＋两seed共1341自由native预测封存、独立评分和根复核完成；**不是仍在训练，也不是只看loss。**

| 父源宏 | tIoU % | sIoU % | vIoU % | Δv vs B1，pp |
|---|---:|---:|---:|---:|
| B1 | 46.637512 | 48.627481 | 32.407600 | — |
| Seed1 | 46.887168 | 47.764785 | 32.167599 | −0.240001 |
| Seed2 | 46.741185 | 48.246816 | 32.295832 | −0.111768 |

Δv配对95%CI分别[−1.121475,+.713042]和[−.915397,+.750465]，均跨0。两seed空间均值下降，原B1 v>.5保持127/136和126/136；当前配置未通过预注册资格。正例、完整31父源匿名数据、query/父源不同层级负尾都保留，不以单个失败否定全部joint路线，也不自动进入expert/OPD/target。

看[完整结果与决定](results/desta3d_v3/2026-09-28/JOINT_LEARNABILITY_CONFIRMATION.md)、[全部31父源匿名数值](results/desta3d_v3/2026-09-28/JOINT_LEARNABILITY_CONFIRMATION.json)、[两seed训练合同与CPU审计修复](results/desta3d_v3/2026-09-28/JOINT_LEARNABILITY_BOTH_TRAINING.md)。4023 scalar/tensor误差4.441e−16、263根汇总误差3.375e−14；所有GPU阶段结束，注册监测已删除。GT仍明确作为source privilege，确认现在开发曝光，不能称无标签TTA或预训练未见。公开只有代码/协议/匿名聚合，raw/标签/媒体/权重保持本地。

## 以下为历史阶段记录

# 最新：开始主方法 GT-evidence Joint Correction learnability

已按用户新指令结束architecture diagnosis优先级；finite component intervention转为后续消融，不再挡住方法构造。新增joint mixer已经实现、4CPU检查和真实PTD接口验收通过，正式训练已启动；**尚无learnability或held-out native收益结果**。

冻结PTD4B、B1、审计256维union；只训练103424参数THW mixer。输入冻结公共stem/双query特征与时间/空间异质GT evidence，输出同一union残差进入两个native pass。零输出初始化精确复现B1；两个可微replay与原native logits逐值相等，空间CE保持完整152775类。原BF16主体/head/native均未换精度。GT privilege明确披露，不能当无标签TTA。

训练618query/95Vid父源，两个seed各固定一完整epoch；native endpoint meanCE+full-vocab coordinate meanCE，AdamW1e-3/accum4/clip1，固定末态不val选步。原31val已有多轮曝光，另从已有source验证metadata按固定hash规则锁31父源全部447query，排除登记的旧source/PANEL16/target64父源和media哈希；source标签曾预处理，**不宣称全历史或PTD预训练未见**。最终Base与两seed共1341自由native预测全封存后评分；GT提前仅作privilege，不冒称score前未读标签。

看[当前协议](protocols/desta3d_v3_joint_learnability_v1.md)、[mixer](vg_tta/desta3d_v3_joint_mixer.py)、[训练](scripts/desta3d_v3_joint_learnability.py)、[确认推理](scripts/desta3d_v3_joint_learnability_eval.py)、[评分](scripts/score_desta3d_v3_joint_learnability.py)。串行GPU/可审计完整窗口恢复/8GiB底线；单Luna每30min读进程状态，异常根修复。只有完成的自由native优势能推进后续资格，expert/OPD/target尚未启动。旧source full-fit保持取消，所有正负与optimizer勘误保留。

## 以下为历史记录

# 最新：Joint component attribution 完成；主线候选改为 Decomposed Evidence, Joint Correction

撤回“时间/空间纠正必须分开”作为当前主方法假设。旧六臂结果和全部正负不变；当前只完成已封存PANEL16梯度/residual的CPU归因，没有新model/GPU/optimizer或预测。

两个128维span交集维数0、并集256维，但并不正交；直接把两投影相加会有53.13%–70.31%重建误差。本轮用T-first/S-first两套正交分解，并独立核原span非正交direct-sum。Joint超出T自身span的部分对T一阶收益16/16为正，平均占Joint局部T下降量10.59%；超出S自身span部分对S也16/16为正，占34.10%。原span归属下S成分对T、T成分对S也全部局部正向，完整数值及分母见[报告](results/desta3d_v3/2026-09-28/JOINT_COMPONENT_ATTRIBUTION.md)和[匿名逐例结果](results/desta3d_v3/2026-09-28/JOINT_COMPONENT_ATTRIBUTION.json)。

**不能把span归属与梯度来源混同。** 按gT/gS来源拆Joint，两方向跨任务贡献均为9正7负；Joint本就包含每个任务的监督梯度。这里只支持局部互补容量，不证明另一个任务监督普遍有益，也没有测单分量native因果效应。旧Joint对Base的CI仍跨0，负尾全部保留。

[新候选与边界](docs/desta3d_v3/JOINT_CORRECTION_CANDIDATE.md)、[锁定CPU协议](protocols/desta3d_v3_joint_component_attribution_v1.md)、[执行入口](scripts/audit_desta3d_v3_joint_components.py)、[独立原始复算](scripts/crosscheck_desta3d_v3_joint_components.py)。4CPU合成控制、48旧seal原件、1570双实现标量核验通过；最大绝对误差6.0042e-8符合预锁混合容差。没有更换幅度/steps/投影去重试旧native，不实现新mixer/TVG/SVG/OPD。下一因果缺口仅记录为未验证，不自动开GPU。旧监测保持已删除。

## 以下为历史完成记录

# 最新：PANEL16 Decomposition Oracle 完成；本配置下 Joint 优于 Decomposed

**核心结论：T/S有不同纠错几何和部分交叉伤害，但当前一次有限纠正未支持分开更新更好。Decomposed相对两个Joint的vIoU差均为负，描述性95%CI均不跨0。不能把低cos当成解耦必要性。**

本轮按同16源、同B1起点计算共同F上的native T/S梯度，并完成一次固定解析纠正的六臂96预测；0optimizer，无新MLP、外部teacher或OPD。原始梯度cos均值-0.004501、中位0.045818，负值7/16，|cos|≤.1为11/16。低相似度本身不当作冲突或解耦必要性。

Decomposed对Base Δv +1.754549pp，对Joint -4.401181pp，对实际双pass能量匹配Joint -5.941072pp；完整CI、t/s/v、固定支持CE、逐例负尾与原好保持见[全部结果](results/desta3d_v3/2026-09-28/DECOMPOSITION_ORACLE.md)和[匿名数值](results/desta3d_v3/2026-09-28/DECOMPOSITION_ORACLE.json)。预注册实用性门：False。本轮不自动推进下一实验。

梯度在进入B1之前的共同THW F处定义，含identity和冻结reader导数，query固定；旧post-adapter两例不冒称本接口已验。各128维分支列空间与256维Joint并集、单位分支梯度平衡、继承源幅度、一次终点，均先锁定。用户残差预算与实际两pass能量两种公平性口径均保留。源GT明确用于oracle，非无标签/目标效果。见[方法合同](docs/desta3d_v3/DECOMPOSITION_ORACLE.md)、[协议](protocols/desta3d_v3_decomposition_oracle_v1.md)、[runner](scripts/desta3d_v3_decomposition_oracle.py)。

## 以下保留历史结果

# 最新：方向审计完成，时间与空间结论不同

按新建议直接读取两个旧成功病例的完整 merger 梯度和固定30步 span 残差，另读 late/early、large、context 的全部正确/错误方向。本轮纯CPU，无新模型执行或预测。

- 时间：成功span与负梯度cos+.061574；large/context正确mask为−.002177/−.002080，确为该起点的局部CE上升方向。
- 空间：成功span为+.085043；large/context正确mask仍为+.006761/+.007645，方向弱但并未反向，旧native sIoU分别改善+4.839735/+4.446244pp。
- 两支mask与成功span末态cos仅.003左右/.015左右。但末态路径不同，低相似度不能单独证明错误方向或共同根因。完整负控与early正例保留。

因此保留scalar-mask停止决定，但不把两支失败一概解释成direction sign错误。附件提出的强符号诊断只在时间例成立；本轮未直接登记prototype或OPD。3D/dual-reader继续作为候选，reader可学性和蒸馏收益未建立。

见[完整20方向数值与裁决](results/desta3d_v3/2026-09-28/DIRECTION_ALIGNMENT.md)、[匿名机器聚合](results/desta3d_v3/2026-09-28/DIRECTION_ALIGNMENT.json)、[审计协议](protocols/desta3d_v3_direction_alignment_v1.md)、[有效CPU入口v2](scripts/audit_desta3d_v3_direction_alignment_v2.py)。4CPU控制、54原seal文件、87pins、176双实现数值核验通过；首次metadata合同失败和原脚本保留。无新target/训练/GPU，累计42493.15652965409s不变；公开不含视频、标签、caption、权重或raw。

## 以下是之前的完成记录

# 最新：context-support 对照完成，停止继续调 scalar mask

按固定PANEL16/B1/late/8.7687%时间与17.0316%空间merger范数，仅增加时间前后各一个已有观测、空间一圈latent-cell邻域。正确时间context对Base的tIoU **−3.2991pp**，正确空间context的sIoU **−1.5757pp**；correct−wrong分别+1.6383pp、+.5718pp，CI均跨0。未满足正确证据同时胜Base和wrong的预定条件，因此停止追加scalar-mask变体，保留3D/dual-reader主线，directional residual仅下一待设计机制，尚无reader训练或OPD。

完整五臂t/s/v、匿名16例、所有CI/原好保持/负尾、旧large对照与核验见[报告](results/desta3d_v3/2026-09-28/CONTEXT_MASK_CONTROL.md)、[机器结果](results/desta3d_v3/2026-09-28/CONTEXT_MASK_CONTROL.json)。正确空间仍9正6负1零，不能称全无收益。78/80格式合法：两空间臂同一源末框null替代box_end仍保留；空间event端点逐值等Base，表面t下降来自原整条格式失败合同。

[协议](protocols/desta3d_v3_context_mask_v1.md)、[context构造](vg_tta/desta3d_v3_context_mask.py)、[runner](scripts/desta3d_v3_context_mask.py)、[独立scorer](scripts/score_desta3d_v3_context_mask.py)、[根病例/旧对照](scripts/audit_desta3d_v3_context_cases.py)。12CPU控制、64mask独立逐值核验、240scalar/tensor几何、4080margin、90parent汇总、256完整BF16端点hash通过；与实际GPU效用分开。80native/0optimizer，GPU153.088997s，累计42493.15652965409s cap=null，已退出。

所有16训练源已开发曝光、尺度来自两个旧源控制；不是无标签TTA/target或泛化验证。扩张后边界裁切导致mask权重不等，未重新选择wrong；残差范数独立配平。科研原件/失败/负数保留，公开不含视频、标签、caption、权重或raw。旧full-source训练取消、CURRENT保持，外部pixel基线暂缓，无自动64/生产晋升。

## 以下为历史记录，当前状态以本节为准

# 最新：大幅度 mask 单因素对照完成；尚不进入 OPD

同原16源/B1/冻结PTD，仅将旧late correct/wrong mask诱导的merger差分别独立归一化到stockF的8.7687%(时间)/17.0316%(空间)，五臂80native、0optimizer。三例中性时间方向原样保留；不扫幅度、不改位置、不扩context、不训练reader或外部teacher。

正确时间mask相对Base的tIoU **−2.5320pp**，正确空间mask的sIoU **−1.7126pp**。正确−错误：时间13例+2.9092pp[0,+8.4008]，空间16例+.4161pp[−.4832,+1.5204]；时间优势集中2例且一例是wrong损害更大，不能当成teacher净收益。空间7正8负1零，混合结果不等于全部mask无效。完整正负/逐例/CI见[主表](results/desta3d_v3/2026-09-28/LARGE_MASK_CONTROL.md)、[匿名机器结果](results/desta3d_v3/2026-09-28/LARGE_MASK_CONTROL.json)。

78/80格式合法：两空间臂同一源末框输出null而非box_end，保留原raw并按原grammar失败计零。空间event端点和logits全等Base；主表t下降是整条格式失效，不是时间分支被改动。原好阈值与连续负尾分别报告。

[协议](protocols/desta3d_v3_large_mask_v1.md)、[执行入口](scripts/desta3d_v3_large_mask_recovery_v2.py)、[独立scorer](scripts/score_desta3d_v3_large_mask.py)、[根病例复核](scripts/audit_desta3d_v3_large_mask_cases.py)。首worker比较CPU/GPU记录时报错，原2预测/12.035秒保留；隔离只修读回设备，配置及重放逐值一致。7CPU合同、240几何、4072margin、90父源汇总、256完整BF16端点hash通过。所有GPU已退出，累计42340.06753310408秒cap=null。

当前结论：固定大幅度确实作用到native，但幅度单独放大没有建立净收益。两个已有span病例只证明可达方向；directionality门未过，reader learnability/distillability未验。保留3D/dual主线；下一context假设需固定operator/location/幅度另登记，本轮未跑。外部pixel基线暂缓，不进OPD/target/64。公开不含权重/视频/标签/caption/预测raw。

## 以下为历史记录，当前状态以本节为准

# 最新：源可控性与同范数位置对照已完成

已按附件完成自由merger→冻结输出列空间→旧mask链，以及同16源的early/late×correct/wrong无更新对照。自由/列空间在两个监督源上改善，但等效latent更新巨大，不能据此声称易学。位置对照未支持“提前mask即可修好”：正确性差值early−late，时间−4.899pp CI跨0、空间+.1298pp CI跨0；错误负控本身退化与全部负尾保留。

先看[位置主表/匿名16例/全部CI](results/desta3d_v3/2026-09-28/SOURCE_LOCATION.json)、[裁决与局限](results/desta3d_v3/2026-09-28/SOURCE_LOCATION.md)、[三层正控表](results/desta3d_v3/2026-09-28/ACTUATION_CHAIN.md)。location144输出/0optimizer，6CPU控制、432几何、7628margin、174+36汇总二核通过。当前所有GPU结束，累计42173.00594093104秒cap=null；源GT诊断非TTA，无target/OPD/64。后续context机制需先固定operator/幅度另登记，当前未跑。外部pixel baseline仍在授权内、须先official smoke。

## 以下为历史记录，当前状态以本节为准

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
