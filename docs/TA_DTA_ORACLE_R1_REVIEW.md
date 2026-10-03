# Oracle Distributional Temporal Adaptation: native head feasibility

**R1 已完成：正确的软时间监督可以通过原生末层梯度更新改善两个数据集的当前-query定位。** 这是一轮显式GT监督的容量/执行诊断，不是无标签方法成绩。确认专家corruption上相对zero-update native，Vid ΔtIoU +8.0634pp、ΔvIoU +5.8219pp；HC +5.2840/+2.6037pp，四个源配对区间均高于零。保留source集中性：Vid确认收益83.9%来自一个source；相对现有A8，HC尚未建立可靠增量且有严重负例。未晋升方法，未启动R2/R3。

## 实际设置与旧实验的区别

TA-STVG官方同域EMA checkpoint，Vid/HC分别使用原Vid-trained/HC2-trained模型；setting是原Paper48 transient deployment corruption，**不是cross-domain**。空间A的1792参数轨迹、框、原始两offset采样、输入像素和25%专家到达位置全部复用。生产注册仍为DeCoTA，与本研究A不同。

原设计每集32开发+16确认来源、每源一query、双序、clean+drop/freeze/blur/occlusion/exposure各5%，总1152到达；仅288个已缓存专家位置实际进行R1（Vid16开发/8确认独立专家源，HC14/7）。其余864到达固定缓存A，完整流读出是缓存模拟，不是新跑一条online stream。来源均历史曝光，confirmation指本轮来源划分，不是fresh test。

只适应原`temp_embed.layers.1`：2×256 weight+2 bias共514名义参数。cached hidden先经冻结MLP首层及ReLU，eval dropout关闭；未把decoder hidden直接当原末层输入。bias整体平移被概率归一化消掉，实际可辨识更新方向至多512。无backbone/suffix/expert新增调用、无decoder/LN/routing/spatial更新。

完整合法i<j joint span分布按两个offset分别归一化，损失平均；保持原生FP32 MAP及两offset物理envelope。没有切换成merged-grid argmax，故收益来自同一个解码接口的参数更新。所有47源验证输入和288目标输入no-update区间精确匹配旧native，目标CUDA/CPU logits最大误差见ROOT_AUDIT。

q_GT为物理边界二维Gaussian，各offset sigma为该offset中位相邻帧距（一原网格cell）；不裁剪GT。使用forward KL(q_GT||pθ)+KL(p0||pθ)，p0每query冻结，普通SGD三步、总是最后一步、无step选择/裁剪/EMA/weight decay。prior KL是软惩罚，不是有严格半径的trust region。GT Gaussian监督正确边界，但不等于固定空间下完美vIoU teacher。

旧native-head实验只有Old8配对受限概率、rank reverse-KL、一次有界更新、未来query收益近零；本轮完整支持/GT forward-KL/3步/当前query/reset是实质不同的可行性诊断。

## 源验证选择：目标上不调参

复用官方train来源派生的31 Vid+16 HC验证查询及hidden（历史曝光），与目标source/media互斥。仅测试预锁五lr，其余beta1/K3/sigma固定；每query复位，按最终source平均tIoU选择，exact tie取较小lr。原95/48 fitting来源未用于训练任何readout。两个lr选择均先seal，随后才开始本轮目标GT teacher。

| 数据集 | 源验证query | 选择lr | native tIoU % | 3步后 tIoU % |
|---|---:|---:|---:|---:|
| Vid | 31 | 0.01 | 49.0091 | 55.0005 |
| HC | 16 | 0.01 | 60.2788 | 72.2506 |

两集均选.01，即这次小网格的上边界。只能称网格内最佳，不称全局最优；本轮没有扩大范围或在目标上再选。完整235个query-lr结果公开。R1直接使用GT Gaussian；附件lambda=.5的PoE构造属于未运行的R2，此轮不用。

## 参数通道主对照：专家corruption，R1 − Native

N是原checkpoint head在同A空间输入上的zero-update native输出；A8是Uniform空间持续学习+原UVTG Fast时间选管。先比较R1−N，才能把原专家readout差异和梯度执行分开。单位pp；CI为10000次配对source-bootstrap。

| 面板 | ΔtIoU | ΔvIoU | v增益/损害cells | >5pp损害 |
|---|---|---|---:|---:|
| vidstg/search (16源/80cells) | +5.2976 [+1.5615, +9.6438] | +2.5021 [+0.1908, +5.4126] | 19/0 | 0 |
| hc2/search (14源/80cells) | +11.3484 [+4.3023, +19.6215] | +5.4854 [+2.3078, +9.1058] | 46/1 | 0 |
| vidstg/confirm (8源/40cells) | +8.0634 [+0.4419, +20.4153] | +5.8219 [+0.0273, +15.5926] | 10/0 | 0 |
| hc2/confirm (7源/40cells) | +5.2840 [+1.5382, +9.4845] | +2.6037 [+0.6606, +5.2052] | 21/0 | 0 |

确认Vid有4/8源正增益、HC6/7；Vid source33占正增益约83.9%，前两源合计99.4%；HC最大source占约52.8%，前两源合计76.4%。两序均正，幅度不同；删除source33后Vid仍正均值，但本轮没有为该删源均值建立独立确认。不能用一个小面板宣布普遍稳定适应。

所有240个corruption专家cell的loss下降；Vid开发52/80、确认11/40区间改变，HC54/80、21/40。三步小更新仍常未跨越离散MAP决策边界。HC开发有一例loss下降而vIoU下降0.5268pp，证明正确边界监督也不保证每条固定空间tube的最终vIoU改善。

## 实际分支参照：专家corruption，R1 − A8

| 面板 | ΔtIoU | ΔvIoU | >5pp损害 |
|---|---|---|---:|
| vidstg/search | -0.3273 [-5.9570, +5.6951] | -0.1595 [-2.6806, +2.2754] | 6 |
| hc2/search | +12.7743 [+4.7852, +21.0101] | +6.5291 [+2.0241, +11.0664] | 6 |
| vidstg/confirm | +10.9086 [+2.8457, +23.1290] | +7.3431 [+1.3315, +17.0894] | 0 |
| hc2/confirm | +3.8592 [-4.7402, +13.6369] | +1.3179 [-3.8930, +7.8855] | 6 |

**GT参数学习有收益，并不等于当前设置可安全替换A8。** Vid开发均值仍略低于A8；HC确认均值正但区间跨零，6/40个专家corruption到达相对A8下降超过5pp。Native通道的零严重损害不能抹掉这个实际参照上的负尾。

## 完整缓存流与clean

| 确认完整corrupt流 | R1−A8 ΔvIoU | R1−A8 ΔtIoU |
|---|---|---|
| Vid 16源/160到达 | +1.8358 [+0.2278, +4.4538] | +2.7271 [+0.4876, +6.1319] |
| HC 16源/160到达 | +0.2484 [-0.8957, +1.6993] | +0.7904 [-1.0679, +3.0812] |

非专家输出逐值保留A，未测试时间feedback向未来位置迁移。上表由真实source聚合得出，不用专家增量×25%代替。Full-flow N只在专家位置取消Fast，其余仍A；teacher_MAP也仅在专家位置干预。GT-time控制覆盖全部到达。

确认clean专家R1−N ΔvIoU：Vid +4.8830 [+0.0000, +14.6490]，HC +1.5619 [+0.4286, +3.2688]。Clean收益说明GT监督并非corruption-specific机制，不单开generic refinement研究线。所有clean、order、nonexpert表在SUMMARY/ROWS完整保存。

## 正控、剩余上限与正负例

| 确认专家corruption | Native vIoU % | R1 % | GT Gaussian MAP % | GT-time % |
|---|---:|---:|---:|---:|
| vidstg | 23.5691 | 29.3911 | 45.4752 | 49.6024 |
| hc2 | 32.4715 | 35.0753 | 48.6351 | 50.4652 |

teacher_MAP是无学习地把GT Gaussian直接送进同一个native支持/解码接口的诊断正控，不是head TTA成绩。它与GT-time差距包含现有采样支持/高斯/双offsetenvelope误差；R1仍远低于teacher_MAP，不能宣布已穷尽末层容量。

- Vid确认source33 occlusion/order2：Native 4.2980%→R1 43.3957%，+39.0977pp；该源驱动大量平均收益。
- HC确认source47 exposure/order1：Native 22.3256%→R1 43.1444%，+20.8189pp。
- HC开发source23 occlusion/order2：Native 11.1294%→R1 10.6025%，−0.5268pp，loss仍下降。
- 相对A8的严重损害：Vid开发source19 frame_drop/order2 A8 67.6521%→R1 18.6517%；HC确认source41 exposure/order2 A8 63.8068%→R1 46.1603%。这些都是已有A8更好的情况，并未因为GT teacher而自动被保留。

CASES保留各数据集/面板相对N和A8的最大收益与最大损害，以及三步loss/位移/指标。零损害面板的最小差值也照实保存，不伪造失败案例。

## 核验、成本与结论范围

6项CPU接口测试通过。根审计独立从joint-marginal交叉熵代数重算523个query-configuration（235source+288target）的全部1569次梯度，并核验每步SGD/参数复位/logits/native解码；1152个A状态/框绑定及所有official dense指标复核通过。根检查数6,044,860，公开标量27,472+tail计数240。公开审计只能复核匿名scalar/选择/汇总，不能凭公开副本重建被排除的private hidden/head weights。

模型通道source计算0.933s、target adaptation 2.197s、dense评分46.531s；这些阶段内时间不含先行metadata/hash验证、checkpoint加载、开发/审计/报告。根独立核验62.356s，公开审计0.676s另计。705 source+864 target backward，共1569次；0GPU初始化、0backbone、0suffix、0专家新调用、0空间更新或跨query时间写入。Private临时优化轨迹36,041,707bytes，未公开权重/hidden/GT坐标/媒体。

启动时旧PyTorch2.0.1不支持mmap，在零预测/零目标GT阶段退出；失败原件已保存，改用已有PyTorch2.7.0的CPU运行，科学配置没有改变。首次合成测试的offset等距tie期望end=9修正为原生早索引end=8，科学数据尚未运行。

**本轮裁决：保留head-only distributional adaptation作为有正证据的梯度接口。** 以GT Gaussian为teacher时能修正原生时间输出，因此不需要先换backbone或扩到decoder/LN。R2的下一问题是同接口上可部署专家teacher能否保留收益；尚未测试。若换成PoE expert teacher失败，不能唯一归因为专家错误，还包含teacher强度/多峰到单个factorized head的投影差异。R1不建立无标签性能、跨源普遍稳定性或online persistence，不恢复旧队列，不改变A/CURRENT。

## 可复现材料

- 协议：`protocols/tastvg_dta_oracle_r1_v1.md`；执行：`docs/tastvg_dta_oracle_r1_v1/EXECUTION.md`。
- 输入/代码锁与private预测barrier：`artifacts/tastvg_dta_oracle_r1_v1`。
- 全部匿名source路径、1152行指标、288条三步轨迹、CI/负尾/cases/三图：`results/tastvg_dta_oracle_r1/2026-10-04`。
- 六测试：`.conda/tubedetr/bin/python -B scripts/test_tastvg_dta_oracle_r1_v1.py`。
- 根：`.conda/tubedetr/bin/python -B scripts/audit_tastvg_dta_oracle_r1_v1.py root`；公开：同脚本加公开结果目录。
