# R90 — 联合准备共享融合与候选绑定路径，再做条件表示恢复TTA

北京时间 2026-11-13 11:29。全部为已暴露开发结果，未访问FINAL。

## 当前裁决

固定主要方法 candidate_joint_tta1：**22.370617%**；相对同模型零步 **+0.002012pp**，相对同信息概率平均 **-0.344375pp**，相对同信息特征平均 **-0.524850pp**。源域结构前提是否通过：**False**；同信息TTA增量是否全为正：**False**。

计算路径确实接通，不再是R89缓存后只训头。这个技术修复与绑定/适应的任务收益分别裁决。固定模型与完整对照无论正负均报告，不以辅助损失下降、参数变化或单个案例宣称成功。独立确认与稳健采用均未成立。

## 实际改变与保留

保留一个Waymo-only3EED epoch100及R76双向结构、原始A256框。仅复用冻结PointNet++/RoBERTa前缀与初始空间假设，所有训练前向重新运行当前融合、冻结但可微的cross-encoder/decoder、局部读取和判别。源域grounding梯度训练现有轻量融合theta和R89同型候选接口eta；无meta学习、新视觉专家、新角色规则或新增完整网络。

对象池化只在指定支持内重新归一化；空支持输出0，禁止偷偷退回全局对象向量。全局信息由原生query/text上下文独立保留。原始点支持来自预测框而非实例GT，原生value已经带上下文，因此“本次池化框外权重为0”不等于“特征只含这个物体”。多个主体/参照假设直到候选条件化读取时才汇总，保持原模型路径。

源域L=完整grounding+两遮挡grounding均值+原物理恢复+.1候选条件表示恢复+.1目标槽几何代理。目标槽仅在源域有IoU>.25且有有效支持的正proposal上，鼓励选择与源目标框相容的own/global假设；不将框内点标成实例真值，不制造参照身份GT。没有可用正代理时，该项为0且样本仍在训练/评价。目标域从不使用这项有标签约束。

条件恢复比较的表示包括query、候选条件化主体/参照读取和own对象读取，经过学习的q/z/t/geo路由投影。测试更新theta及这些投影，最终判别outMLP、原3EED和R76既有RGB编码器权重保持冻结；baseline匹配分支允许更新自身匹配头与theta，query-only仅theta+q。因此参数量不等，不能称等参数对比。

## 源域梯度路径与参数范围

独立受控source检查：冻结cross-encoder输入的grounding梯度范数 **0.024218662**，最后decoder输出梯度 **0.037593592**，image_to_point参数梯度 **0.063008716**。这排除了仅经lateRGB捷径产生梯度的解释。辅助项对eta.z/t/geo梯度分别为 **0.022118193/0.000209104/0.000007099**。所有检查无参数修改，无目标标签。

| 分支 | 源域可训练参数 | 测试可更新参数 |
|---|---:|---:|
| continue_joint | 287458 | 287458 |
| query_joint | 175428 | 120994 |
| candidate_headonly | 72930 | 158339 |
| candidate_joint | 175428 | 158339 |

query_joint源域分配175428个参数，但z/t/geo未被其前向使用，梯度活跃部分为theta102498+query/out35585=138083。其余分支的结构/可训练范围如上；相同源数据、更新次数、验证预算，不是完全相同计算/容量。

## 源域准备与源域受控无标签更新

TRAIN512/VAL128，94/82场景，场景与帧交集0；四分支各2轮、每轮512次AdamW更新、lr1e-4、decay1e-4、clip1，次序相同。共用epoch按四分支零更新源域VAL均值、最早最大值选第1轮。在源域诊断或目标域成绩前已封存，辅助更新后成绩从未用于模型选择。源验证集已用于历史研发/选权重，不能称全新独立确认。

R76源域初始化mIoU为47.385454%。源域全部训练曲线：

| 分支 | epoch | 完整mIoU % | 遮挡平均mIoU % | 完整训练loss | 条件恢复loss | 目标槽proxy loss |
|---|---:|---:|---:|---:|---:|---:|
| continue_joint | 1 | 45.803676 | 44.860438 | 3.885530 | 0.001536 | 0.000000 |
| continue_joint | 2 | 45.395239 | 45.654425 | 3.851909 | 0.001417 | 0.000000 |
| query_joint | 1 | 46.040059 | 44.907005 | 3.888557 | 0.007752 | 0.000000 |
| query_joint | 2 | 45.558155 | 45.280660 | 3.884120 | 0.004419 | 0.000000 |
| candidate_headonly | 1 | 47.432991 | 46.631001 | 3.884293 | 0.015561 | 0.413453 |
| candidate_headonly | 2 | 47.381185 | 46.713074 | 3.881958 | 0.018201 | 0.275643 |
| candidate_joint | 1 | 46.040059 | 44.890634 | 3.888310 | 0.013909 | 0.422156 |
| candidate_joint | 2 | 45.888943 | 45.166208 | 3.883245 | 0.013896 | 0.313770 |

选定的四个模型在VAL128先运行完全无标签的内部更新并封存，再单独连接source IoU。下表不是额外训练，也不拿source GT选择步长/接受更新。

| 源域读出/更新 | mIoU % | Acc@25 % | Acc@50 % |
|---|---:|---:|---:|
| continue_joint_frozen | 45.803676 | 85.156250 | 43.750000 |
| continue_joint_mean3 | 45.446175 | 84.375000 | 42.968750 |
| continue_joint_feature3 | 44.997180 | 83.593750 | 42.968750 |
| continue_joint_tta1 | 45.803676 | 85.156250 | 43.750000 |
| query_joint_frozen | 46.040059 | 85.156250 | 43.750000 |
| query_joint_mean3 | 44.790684 | 82.812500 | 42.187500 |
| query_joint_feature3 | 44.355319 | 82.031250 | 42.187500 |
| query_joint_tta1 | 45.982774 | 85.156250 | 43.750000 |
| candidate_headonly_frozen | 47.432991 | 85.937500 | 46.093750 |
| candidate_headonly_mean3 | 47.129460 | 85.937500 | 45.312500 |
| candidate_headonly_feature3 | 46.645836 | 85.156250 | 45.312500 |
| candidate_headonly_tta1 | 47.618097 | 85.937500 | 46.093750 |
| candidate_joint_frozen | 46.040059 | 85.156250 | 43.750000 |
| candidate_joint_mean3 | 44.802633 | 82.812500 | 42.187500 |
| candidate_joint_feature3 | 44.367268 | 82.031250 | 42.187500 |
| candidate_joint_tta1 | 45.982774 | 85.156250 | 43.750000 |
| candidate_joint_tta3 | 45.982774 | 85.156250 | 43.750000 |
| no_ref_frozen | 46.040059 | 85.156250 | 43.750000 |
| soft_pool_frozen | 46.040059 | 85.156250 | 43.750000 |
| candidate_joint_bypass1 | 45.973245 | 85.156250 | 43.750000 |
| candidate_joint_physical1 | 45.982774 | 85.156250 | 43.750000 |
| candidate_joint_shuffle1 | 45.973245 | 85.156250 | 43.750000 |

封存后source GT梯度仅作方向诊断：正cos表示局部辅助下降方向与grounding下降方向相容，不能替代离散选框任务结果；负cos表示局部冲突。不会据此过滤目标样本/更新。

| 更新目标 | 有效n | cos均值 | cos中位数 | 正/负数 |
|---|---:|---:|---:|---:|
| full | 128 | -0.038632125 | -0.063020647 | 59/69 |
| bypass | 128 | -0.044412955 | -0.059141967 | 58/70 |
| physical | 128 | -0.050287558 | -0.069594845 | 55/73 |

源域若未优于query-only/只训头/撤回参照，对目标域的任何正负结果都不能自动解释为可靠新绑定；不把失败仅推给TTA。不追加source epoch或目标lr网格挽救本轮。

## DEV336全部方法

Drone/Quad各168条、五个历史场景，主指标两平台等权mIoU。所有失败、UNKNOWN、回退、零更新保留。每输入恢复source；固定1/3步、SGD lr.1/clip1及9档Armijo回溯，接收只看同一停止梯度目标，不读取任务IoU。物理恢复+条件恢复+.1完整条件表示保持与预测KL。对照bypass只替换为聚合前区域目标；physical去掉条件恢复；shuffle打乱candidate条件目标对应；三者共享其余条件，源标签不进入目标更新。

同信息零更新两种：完整+两遮挡的概率平均；同三次前向的原生query/text投影平均，再加平均条件读出输入经冻结判别头的结果。后者没有额外模型调用或逐输入搜索，不称为原R59精确复现。历史source、R76、FrozenABC、R59及三条R89基线直接复用其封存预测；其余26条新端点重新计算并逐条原生重放。全部33条统一连接相同已保存A256 IoU，未重新用GT挑框。

| 方法 | Drone mIoU % | Quad mIoU % | 等权mIoU % | 等权Acc25 % | 等权Acc50 % |
|---|---:|---:|---:|---:|---:|
| source | 15.539892 | 28.120770 | 21.830331 | 40.773810 | 21.130952 |
| frozen_bi | 17.083330 | 28.139863 | 22.611596 | 41.071429 | 21.428571 |
| frozen_ABC | 15.736332 | 28.868588 | 22.302460 | 42.261905 | 21.428571 |
| R59 | 16.385568 | 28.836664 | 22.611116 | 43.154762 | 21.726190 |
| R89_candidate_frozen | 17.061770 | 28.068796 | 22.565283 | 41.071429 | 21.130952 |
| R89_candidate_tta1 | 17.209577 | 28.068796 | 22.639187 | 41.071429 | 21.130952 |
| R89_continue_tta1 | 17.559059 | 28.097869 | 22.828464 | 41.369048 | 21.130952 |
| continue_joint_frozen | 17.001706 | 28.166731 | 22.584218 | 40.178571 | 20.833333 |
| continue_joint_mean3 | 16.901333 | 28.413725 | 22.657529 | 39.880952 | 21.130952 |
| continue_joint_feature3 | 16.901333 | 28.496771 | 22.699052 | 40.476190 | 20.833333 |
| continue_joint_tta1 | 17.002524 | 28.166731 | 22.584628 | 40.178571 | 21.130952 |
| query_joint_frozen | 16.805435 | 27.931775 | 22.368605 | 41.369048 | 20.833333 |
| query_joint_mean3 | 17.266161 | 28.153218 | 22.709689 | 41.369048 | 21.130952 |
| query_joint_feature3 | 17.301687 | 28.477990 | 22.889839 | 41.964286 | 21.130952 |
| query_joint_tta1 | 16.879742 | 27.861492 | 22.370617 | 41.369048 | 21.130952 |
| candidate_headonly_frozen | 17.199782 | 28.287681 | 22.743731 | 41.369048 | 21.130952 |
| candidate_headonly_mean3 | 16.694763 | 28.648101 | 22.671432 | 41.369048 | 21.130952 |
| candidate_headonly_feature3 | 16.694763 | 28.499141 | 22.596952 | 41.666667 | 20.833333 |
| candidate_headonly_tta1 | 17.198565 | 28.287681 | 22.743123 | 41.369048 | 21.130952 |
| candidate_joint_frozen | 16.805435 | 27.931775 | 22.368605 | 41.369048 | 20.833333 |
| candidate_joint_mean3 | 17.265508 | 28.164475 | 22.714992 | 41.369048 | 21.130952 |
| candidate_joint_feature3 | 17.301687 | 28.489247 | 22.895467 | 41.964286 | 21.130952 |
| candidate_joint_tta1 | 16.879742 | 27.861492 | 22.370617 | 41.369048 | 21.130952 |
| candidate_joint_tta3 | 16.801784 | 27.861492 | 22.331638 | 41.369048 | 20.833333 |
| no_ref_frozen | 16.805435 | 27.931775 | 22.368605 | 41.369048 | 20.833333 |
| no_ref_tta1 | 16.879742 | 27.861492 | 22.370617 | 41.369048 | 21.130952 |
| soft_pool_frozen | 16.805435 | 27.931775 | 22.368605 | 41.369048 | 20.833333 |
| soft_pool_tta1 | 16.879742 | 27.861492 | 22.370617 | 41.369048 | 21.130952 |
| wrong_space_frozen | 16.805435 | 27.931775 | 22.368605 | 41.369048 | 20.833333 |
| wrong_space_tta1 | 16.870435 | 27.861492 | 22.365963 | 41.369048 | 21.130952 |
| candidate_joint_bypass1 | 16.879742 | 27.861492 | 22.370617 | 41.369048 | 21.130952 |
| candidate_joint_physical1 | 16.879742 | 27.861492 | 22.370617 | 41.369048 | 21.130952 |
| candidate_joint_shuffle1 | 16.857928 | 27.861492 | 22.359710 | 41.071429 | 21.130952 |

## 完整配对组件与所有负例

no_ref撤回新接口的显式参照读取，原生全局上下文仍在；soft_pool用R89式.05框外底权重，是相同新权重下的推理撤回，不是重训替代架构；wrong_space置换局部支持点，原RGB–LiDAR输入对应/物理目标保留；shuffle则打乱条件向量目标候选对应。都不是实例身份真值。

| 比较 | 方法 | 对照 | 平台 | Δpp | 改善/恶化/不变 | 最大单例正/负贡献pp |
|---|---|---|---|---:|---:|---:|
| source_preparation_continue_joint | candidate_joint_frozen | continue_joint_frozen | balanced | -0.215614 | 22/17/297 | +0.037520/-0.132602 |
| source_preparation_continue_joint | candidate_joint_frozen | continue_joint_frozen | drone | -0.196271 | 18/12/138 | +0.075041/-0.265204 |
| source_preparation_continue_joint | candidate_joint_frozen | continue_joint_frozen | quad | -0.234956 | 4/5/159 | +0.070283/-0.253073 |
| source_preparation_query_joint | candidate_joint_frozen | query_joint_frozen | balanced | +0.000000 | 0/0/336 | +0.000000/+0.000000 |
| source_preparation_query_joint | candidate_joint_frozen | query_joint_frozen | drone | +0.000000 | 0/0/168 | +0.000000/+0.000000 |
| source_preparation_query_joint | candidate_joint_frozen | query_joint_frozen | quad | +0.000000 | 0/0/168 | +0.000000/+0.000000 |
| source_preparation_candidate_headonly | candidate_joint_frozen | candidate_headonly_frozen | balanced | -0.375126 | 9/16/311 | +0.035141/-0.126536 |
| source_preparation_candidate_headonly | candidate_joint_frozen | candidate_headonly_frozen | drone | -0.394347 | 6/11/151 | +0.020597/-0.152199 |
| source_preparation_candidate_headonly | candidate_joint_frozen | candidate_headonly_frozen | quad | -0.355906 | 3/5/160 | +0.070283/-0.253073 |
| new_source_prepared_structure | candidate_joint_frozen | frozen_bi | balanced | -0.242991 | 10/16/310 | +0.094792/-0.126536 |
| new_source_prepared_structure | candidate_joint_frozen | frozen_bi | drone | -0.277895 | 6/14/148 | +0.189584/-0.152199 |
| new_source_prepared_structure | candidate_joint_frozen | frozen_bi | quad | -0.208088 | 4/2/162 | +0.070283/-0.253073 |
| adaptation_or_control_candidate_joint_frozen | candidate_joint_tta1 | candidate_joint_frozen | balanced | +0.002012 | 2/1/333 | +0.032500/-0.035141 |
| adaptation_or_control_candidate_joint_frozen | candidate_joint_tta1 | candidate_joint_frozen | drone | +0.074307 | 2/0/166 | +0.065000/+0.000000 |
| adaptation_or_control_candidate_joint_frozen | candidate_joint_tta1 | candidate_joint_frozen | quad | -0.070283 | 0/1/167 | +0.000000/-0.070283 |
| adaptation_or_control_candidate_joint_mean3 | candidate_joint_tta1 | candidate_joint_mean3 | balanced | -0.344375 | 22/16/298 | +0.040049/-0.170787 |
| adaptation_or_control_candidate_joint_mean3 | candidate_joint_tta1 | candidate_joint_mean3 | drone | -0.385767 | 12/9/147 | +0.053356/-0.265204 |
| adaptation_or_control_candidate_joint_mean3 | candidate_joint_tta1 | candidate_joint_mean3 | quad | -0.302982 | 10/7/151 | +0.080098/-0.341574 |
| adaptation_or_control_candidate_joint_feature3 | candidate_joint_tta1 | candidate_joint_feature3 | balanced | -0.524850 | 24/19/293 | +0.081457/-0.170787 |
| adaptation_or_control_candidate_joint_feature3 | candidate_joint_tta1 | candidate_joint_feature3 | drone | -0.421946 | 12/9/147 | +0.052329/-0.265204 |
| adaptation_or_control_candidate_joint_feature3 | candidate_joint_tta1 | candidate_joint_feature3 | quad | -0.627754 | 12/10/146 | +0.162913/-0.341574 |
| adaptation_or_control_R59 | candidate_joint_tta1 | R59 | balanced | -0.240499 | 43/60/233 | +0.234305/-0.168124 |
| adaptation_or_control_R59 | candidate_joint_tta1 | R59 | drone | +0.494174 | 25/38/105 | +0.468611/-0.335152 |
| adaptation_or_control_R59 | candidate_joint_tta1 | R59 | quad | -0.975172 | 18/22/128 | +0.292609/-0.336248 |
| adaptation_or_control_R89_continue_tta1 | candidate_joint_tta1 | R89_continue_tta1 | balanced | -0.457847 | 14/21/301 | +0.047379/-0.132602 |
| adaptation_or_control_R89_continue_tta1 | candidate_joint_tta1 | R89_continue_tta1 | drone | -0.679318 | 12/14/142 | +0.094759/-0.265204 |
| adaptation_or_control_R89_continue_tta1 | candidate_joint_tta1 | R89_continue_tta1 | quad | -0.236376 | 2/7/159 | +0.022851/-0.253073 |
| adaptation_or_control_continue_joint_tta1 | candidate_joint_tta1 | continue_joint_tta1 | balanced | -0.214011 | 19/15/302 | +0.037520/-0.132602 |
| adaptation_or_control_continue_joint_tta1 | candidate_joint_tta1 | continue_joint_tta1 | drone | -0.122783 | 16/10/142 | +0.075041/-0.265204 |
| adaptation_or_control_continue_joint_tta1 | candidate_joint_tta1 | continue_joint_tta1 | quad | -0.305238 | 3/5/160 | +0.025087/-0.253073 |
| adaptation_or_control_query_joint_tta1 | candidate_joint_tta1 | query_joint_tta1 | balanced | +0.000000 | 0/0/336 | +0.000000/+0.000000 |
| adaptation_or_control_query_joint_tta1 | candidate_joint_tta1 | query_joint_tta1 | drone | +0.000000 | 0/0/168 | +0.000000/+0.000000 |
| adaptation_or_control_query_joint_tta1 | candidate_joint_tta1 | query_joint_tta1 | quad | +0.000000 | 0/0/168 | +0.000000/+0.000000 |
| adaptation_or_control_candidate_headonly_tta1 | candidate_joint_tta1 | candidate_headonly_tta1 | balanced | -0.372506 | 9/14/313 | +0.011425/-0.126536 |
| adaptation_or_control_candidate_headonly_tta1 | candidate_joint_tta1 | candidate_headonly_tta1 | drone | -0.318823 | 7/9/152 | +0.020597/-0.152199 |
| adaptation_or_control_candidate_headonly_tta1 | candidate_joint_tta1 | candidate_headonly_tta1 | quad | -0.426188 | 2/5/161 | +0.022851/-0.253073 |
| adaptation_or_control_candidate_joint_bypass1 | candidate_joint_tta1 | candidate_joint_bypass1 | balanced | +0.000000 | 0/0/336 | +0.000000/+0.000000 |
| adaptation_or_control_candidate_joint_bypass1 | candidate_joint_tta1 | candidate_joint_bypass1 | drone | +0.000000 | 0/0/168 | +0.000000/+0.000000 |
| adaptation_or_control_candidate_joint_bypass1 | candidate_joint_tta1 | candidate_joint_bypass1 | quad | +0.000000 | 0/0/168 | +0.000000/+0.000000 |
| adaptation_or_control_candidate_joint_physical1 | candidate_joint_tta1 | candidate_joint_physical1 | balanced | +0.000000 | 0/0/336 | +0.000000/+0.000000 |
| adaptation_or_control_candidate_joint_physical1 | candidate_joint_tta1 | candidate_joint_physical1 | drone | +0.000000 | 0/0/168 | +0.000000/+0.000000 |
| adaptation_or_control_candidate_joint_physical1 | candidate_joint_tta1 | candidate_joint_physical1 | quad | +0.000000 | 0/0/168 | +0.000000/+0.000000 |
| adaptation_or_control_candidate_joint_shuffle1 | candidate_joint_tta1 | candidate_joint_shuffle1 | balanced | +0.010907 | 2/0/334 | +0.010298/+0.000000 |
| adaptation_or_control_candidate_joint_shuffle1 | candidate_joint_tta1 | candidate_joint_shuffle1 | drone | +0.021813 | 2/0/166 | +0.020597/+0.000000 |
| adaptation_or_control_candidate_joint_shuffle1 | candidate_joint_tta1 | candidate_joint_shuffle1 | quad | +0.000000 | 0/0/168 | +0.000000/+0.000000 |
| no_ref | candidate_joint_frozen | no_ref_frozen | balanced | +0.000000 | 0/0/336 | +0.000000/+0.000000 |
| no_ref | candidate_joint_frozen | no_ref_frozen | drone | +0.000000 | 0/0/168 | +0.000000/+0.000000 |
| no_ref | candidate_joint_frozen | no_ref_frozen | quad | +0.000000 | 0/0/168 | +0.000000/+0.000000 |
| no_ref | candidate_joint_tta1 | no_ref_tta1 | balanced | +0.000000 | 0/0/336 | +0.000000/+0.000000 |
| no_ref | candidate_joint_tta1 | no_ref_tta1 | drone | +0.000000 | 0/0/168 | +0.000000/+0.000000 |
| no_ref | candidate_joint_tta1 | no_ref_tta1 | quad | +0.000000 | 0/0/168 | +0.000000/+0.000000 |
| soft_pool | candidate_joint_frozen | soft_pool_frozen | balanced | +0.000000 | 0/0/336 | +0.000000/+0.000000 |
| soft_pool | candidate_joint_frozen | soft_pool_frozen | drone | +0.000000 | 0/0/168 | +0.000000/+0.000000 |
| soft_pool | candidate_joint_frozen | soft_pool_frozen | quad | +0.000000 | 0/0/168 | +0.000000/+0.000000 |
| soft_pool | candidate_joint_tta1 | soft_pool_tta1 | balanced | +0.000000 | 0/0/336 | +0.000000/+0.000000 |
| soft_pool | candidate_joint_tta1 | soft_pool_tta1 | drone | +0.000000 | 0/0/168 | +0.000000/+0.000000 |
| soft_pool | candidate_joint_tta1 | soft_pool_tta1 | quad | +0.000000 | 0/0/168 | +0.000000/+0.000000 |
| wrong_space | candidate_joint_frozen | wrong_space_frozen | balanced | +0.000000 | 0/0/336 | +0.000000/+0.000000 |
| wrong_space | candidate_joint_frozen | wrong_space_frozen | drone | +0.000000 | 0/0/168 | +0.000000/+0.000000 |
| wrong_space | candidate_joint_frozen | wrong_space_frozen | quad | +0.000000 | 0/0/168 | +0.000000/+0.000000 |
| wrong_space | candidate_joint_tta1 | wrong_space_tta1 | balanced | +0.004653 | 1/0/335 | +0.004653/+0.000000 |
| wrong_space | candidate_joint_tta1 | wrong_space_tta1 | drone | +0.009307 | 1/0/167 | +0.009307/+0.000000 |
| wrong_space | candidate_joint_tta1 | wrong_space_tta1 | quad | +0.000000 | 0/0/168 | +0.000000/+0.000000 |
| secondary_three_steps | candidate_joint_tta3 | candidate_joint_tta1 | balanced | -0.038979 | 0/2/334 | +0.000000/-0.032500 |
| secondary_three_steps | candidate_joint_tta3 | candidate_joint_tta1 | drone | -0.077957 | 0/2/166 | +0.000000/-0.065000 |
| secondary_three_steps | candidate_joint_tta3 | candidate_joint_tta1 | quad | +0.000000 | 0/0/168 | +0.000000/+0.000000 |

## 场景聚类边界

| 平台/场景 | n | 方法 | mIoU % |
|---|---:|---|---:|
| drone/D_S1 | 106 | frozen_bi | 13.449026 |
| drone/D_S1 | 106 | R59 | 11.761612 |
| drone/D_S1 | 106 | continue_joint_tta1 | 13.636613 |
| drone/D_S1 | 106 | query_joint_tta1 | 13.257626 |
| drone/D_S1 | 106 | candidate_headonly_tta1 | 13.658612 |
| drone/D_S1 | 106 | candidate_joint_frozen | 13.154607 |
| drone/D_S1 | 106 | candidate_joint_feature3 | 13.777952 |
| drone/D_S1 | 106 | candidate_joint_tta1 | 13.257626 |
| drone/D_S2 | 62 | frozen_bi | 23.296817 |
| drone/D_S2 | 62 | R59 | 24.291041 |
| drone/D_S2 | 62 | continue_joint_tta1 | 22.757147 |
| drone/D_S2 | 62 | query_joint_tta1 | 23.072391 |
| drone/D_S2 | 62 | candidate_headonly_tta1 | 23.250744 |
| drone/D_S2 | 62 | candidate_joint_frozen | 23.047173 |
| drone/D_S2 | 62 | candidate_joint_feature3 | 23.326138 |
| drone/D_S2 | 62 | candidate_joint_tta1 | 23.072391 |
| quad/Q_S1 | 62 | frozen_bi | 10.447889 |
| quad/Q_S1 | 62 | R59 | 9.632116 |
| quad/Q_S1 | 62 | continue_joint_tta1 | 10.744790 |
| quad/Q_S1 | 62 | query_joint_tta1 | 10.603438 |
| quad/Q_S1 | 62 | candidate_headonly_tta1 | 10.848428 |
| quad/Q_S1 | 62 | candidate_joint_frozen | 10.603438 |
| quad/Q_S1 | 62 | candidate_joint_feature3 | 10.717904 |
| quad/Q_S1 | 62 | candidate_joint_tta1 | 10.603438 |
| quad/Q_S2 | 49 | frozen_bi | 34.118085 |
| quad/Q_S2 | 49 | R59 | 36.067581 |
| quad/Q_S2 | 49 | continue_joint_tta1 | 33.834533 |
| quad/Q_S2 | 49 | query_joint_tta1 | 32.966856 |
| quad/Q_S2 | 49 | candidate_headonly_tta1 | 34.118085 |
| quad/Q_S2 | 49 | candidate_joint_frozen | 33.207824 |
| quad/Q_S2 | 49 | candidate_joint_feature3 | 33.986099 |
| quad/Q_S2 | 49 | candidate_joint_tta1 | 32.966856 |
| quad/Q_S3 | 57 | frozen_bi | 42.244590 |
| quad/Q_S3 | 57 | R59 | 43.509770 |
| quad/Q_S3 | 57 | continue_joint_tta1 | 42.244590 |
| quad/Q_S3 | 57 | query_joint_tta1 | 42.244590 |
| quad/Q_S3 | 57 | candidate_headonly_tta1 | 42.244590 |
| quad/Q_S3 | 57 | candidate_joint_frozen | 42.244590 |
| quad/Q_S3 | 57 | candidate_joint_feature3 | 43.094115 |
| quad/Q_S3 | 57 | candidate_joint_tta1 | 42.244590 |

不将同场景帧/描述当独立场景证据，不因某场景下降就一票否决，也不以这些历史场景做全新总体CI。

## 案例与机制

全体最大收益/损伤、绝对高分、可恢复错误按固定规则选图；保留R89固定正负例。案例只解释结果，不改变方法。原句、RGB、点云、坐标和query索引均只在本地私有图册；本地人工核验见CASE_REVIEW。

| 案例 | 平台 | 原R76 | 新联合冻结 | 概率均值 | 特征均值 | 一步 | 三步 |
|---|---|---:|---:|---:|---:|---:|---:|
| CASE001 | drone | 0.000000 | 0.318500 | 0.297039 | 0.297039 | 0.318500 | 0.318500 |
| CASE002 | quad | 0.327987 | 0.446062 | 0.327987 | 0.327987 | 0.327987 | 0.327987 |
| CASE003 | quad | 0.425162 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| CASE004 | drone | 0.255694 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| CASE005 | drone | 0.523128 | 0.413929 | 0.523128 | 0.523128 | 0.523128 | 0.413929 |
| CASE006 | drone | 0.203502 | 0.187866 | 0.203502 | 0.203502 | 0.203502 | 0.203502 |
| CASE007 | quad | 0.581626 | 0.581626 | 0.581626 | 0.307932 | 0.581626 | 0.581626 |
| CASE008 | quad | 0.607324 | 0.607324 | 0.472758 | 0.472758 | 0.607324 | 0.607324 |
| CASE009 | quad | 0.000000 | 0.000000 | 0.573844 | 0.573844 | 0.000000 | 0.000000 |
| CASE010 | quad | 0.000000 | 0.000000 | 0.000000 | 0.466849 | 0.000000 | 0.000000 |
| CASE011 | quad | 0.812701 | 0.812701 | 0.812701 | 0.812701 | 0.812701 | 0.812701 |
| CASE012 | quad | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## 数值审计、成本及实际范围

目标原生独立重放读出 8736 个、实际loss检查 20160 个，概率最大误差 0，loss最大误差 0。接受后总loss增加数0；固定原A256/最终判别头及原模型不变。源域初次更新审计因requires_grad重放状态不一致出现1.73e-10概率差；修正审计状态后误差0，原预测/更新规则/容差未改，失败记录保留。

四分支真实source训练合计 538.158s，使用既有冻结前缀，非整模型从零训练耗时。目标33臂合集 1046.692s，峰值GPU已分配张量 1586376192bytes，包含全部对照和保存，并非单方法延迟。SOURCE与TARGET运行均单GPU互斥；source内层/目标更新均无任务标签，source有标签训练与封存后诊断明确分开。

已暴露DEV仅用于本次固定算法评价，不再按目标标签调参。新支持/辅助与原型/源数据量、冻结RGB编码器、有限初始hypotheses、参数组优化、source checkpoint选择均限定解释范围；未排除更充分的准备条件，不能将这一轮负结果泛化为所有联合绑定或TTA不可行。FINAL未动；不排队后续实验。

## 总体机制裁决与逐例审查边界

源域前提未通过：联合绑定零步46.040059%，与query-only选框mIoU相同，低于仅训练接口47.432991%及R76初始化47.385454%；联合无标签一步45.982774%也低于自身零步。真实梯度到达新接口不等于已有可靠绑定。

DEV336上，联合结构零步相对query-only为+0.000000pp；完整一步相对自身零步+0.002012pp（2改善/1恶化/333不变），相对同信息表示融合-0.524850pp（24改善/19恶化）。相对R59为-0.240499pp。三步仅作预先锁定的次要结果，不据目标成绩替换主要方法。

条件表示恢复相对纯物理恢复+0.000000pp，相对打乱对应+0.010907pp，相对去参照+0.000000pp。这些是固定条件下的任务差分；任一接近零都不支持把梯度非零解释为该组件有实质贡献。错配控制仍可能造成小幅一般扰动，不能仅凭一项微弱差分宣布真实对象绑定。

完整一步接受336/336条；条件恢复项下降310条，物理项下降336条。未参与构造/接受的第三遮挡，条件误差均值0.014475131→0.013645571，304条改善。辅助任务层面的这些结果与离散选框收益分开判断，不以其代替grounding。

角色分层只作封存后描述，子集pooled按该子集实际样本平均，不冒充平台等权总体；主体/参照解析成功不等于实例身份核验成功。

| 子集 | 平台 | n | 一步相对零步Δpp | 改善/恶化 |
|---|---|---:|---:|---:|
| all | pooled | 336 | +0.002012 | 2/1 |
| all | drone | 168 | +0.074307 | 2/0 |
| all | quad | 168 | -0.070283 | 0/1 |
| reference_available | pooled | 222 | -0.046144 | 1/1 |
| reference_available | drone | 114 | +0.013716 | 1/0 |
| reference_available | quad | 108 | -0.109328 | 0/1 |
| no_reference_available | pooled | 114 | +0.095789 | 1/0 |
| no_reference_available | drone | 54 | +0.202221 | 1/0 |
| no_reference_available | quad | 60 | +0.000000 | 0/0 |
| original_oracle_above25 | pooled | 268 | +0.002523 | 2/1 |
| original_oracle_above25 | drone | 126 | +0.099076 | 2/0 |
| original_oracle_above25 | quad | 142 | -0.083151 | 0/1 |
| original_oracle_at_most25 | pooled | 68 | +0.000000 | 0/0 |
| original_oracle_at_most25 | drone | 42 | +0.000000 | 0/0 |
| original_oracle_at_most25 | quad | 26 | +0.000000 | 0/0 |

源域proxy资格事后统计：TRAIN512中510条A256 oracle>25，480条可用目标槽代理；VAL128中127条oracle>25，119条可用代理。监督没有因大部分样本无正proposal而整体失效；但代理仍不能提供参照身份或属性归属真值。

本轮采纳的是实现修复：冻结前缀缓存、可微冻结后缀、theta/eta联合源域监督、严格支持池化、候选条件辅助和可更新router。**不采纳“可靠角色绑定及稳健TTA收益已经成立”的科学结论。** 不追加训练轮数、目标阈值或学习率搜索。后续若继续，应先让源域绑定与候选证据呈现超过简单接口的任务价值；这不是本轮已验证的后续算法，也没有为其启动实验。

本地CASE_REVIEW保留全部选定正负例、固定R89回访和UNKNOWN。IoU提高/降低、同框/换query、代理支持分布与真实物体身份是不同证据；没有独立身份核验时，不将“换query”写成“换物体”，不将attention/support写成实例mask。公开包仅含匿名标量和代码，完整描述、媒体、坐标和内部特征不发布。

## 原图复核后的具体归因

17页图册包含15个唯一观测，两个旧页回访与新页重合，不能当独立样本累加。12新页概览后，放大11新页和1旧页；其余页的身份保留UNKNOWN，详情在本地CASE_REVIEW及匿名审查表。

两条一步TTA改善CASE005/006均精确恢复R76原query；放大后主要是同一车辆附近的近似框切换，直接表示融合也恢复，不能证明新实例绑定。唯一一步恶化CASE002也回到R76原query，两框主要包围同一行人范围。CASE003/004则在联合零步前向已经从主体区域转向另一行人/邻车区域；随后TTA没有救回。CASE009/010由同信息直接融合选到更有重叠的候选，TTA仍未实现相应选择。CASE011是所有状态一致的稳定正例；CASE012仍将主体与参照类别混淆。CASE001无空间角色，只支持query后备路径的静态改进。固定R89_CASE005的旧适应正例在本轮只由直接融合保留。

保存分数进一步区分“模块未执行”与“模块没有任务价值”：撤去参照的零步分布最大绝对差均值为0.000230892，改变1个query但0个IoU；旧式池化和错位支持零步分别改变0个query。完整一步与纯物理/绕过条件表示相比，概率分布均确有数值差，但336条Top1 query完全相同。这说明新增路径在计算，尚未推动决定性的目标排序；不能再用参数非零更新作为其有效性证明。

零更新表示融合的22.895467%可保留为开发参照；但query-only分支同样融合已达22.889839%，两者仅差0.005628pp，不能将这一较强融合结果归功于新空间绑定。主算法与两个直接零更新对照的差分为负，且源域绑定前提未通过，本轮不以此零更新对照替换成一个“成功TTA”结论。
