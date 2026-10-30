# R8 独立实验协议审计（实现前草案）

R7 的事实是 full5912 三种 G 都没有产生可用更新，而不是参数回归机制已在真实目标反馈上失败。R7 人工回归头正控仅 strict FP32 条件通过；原模式的全 query 等价误差仍超过已登记 5e-5m。R8 必须用同一新精度下重新计算的 source，不能把 R7 frozen 分数作为新方法的配对基线。

## 实际范围与测量分解

源模型固定为已审计 Waymo-only epoch100，目标为 Drone2984/Quad2928 条官方 VAL。它们未进入 source 训练，但已在 R5/R7 评过，不是新的研究确认集。目标仅有 2 个 Drone、3 个 Quad 场景簇。不得把表达、帧、query、crop 或方法臂当成独立场景。

G author 已明确：q0 是原纯文本 source argmax；q1 是语义选中去重组中原 textscore 最高的 canonical **原始 proposal**，不是聚合后的新框。所有适应在 q1 的原始冻结特征上执行，1/3 步各从同一 source 权重恢复。更新后不重新选 query。每个反馈模式分别保留：

1. matched strict-FP32 source q0（共享）；
2. association-only：q1 原始 source 框；
3. direct pseudo：q1 的非学习组内中心共识框，无参数更新；
4. parameter1、parameter3：q1 回归更新；
5. exact output-space1、output-space3：同信息、同学习率的冻结特征 Gram 等价框步。

三个模式使用同一次 source、同一候选准备、同一投影；共享 q0 加每模式六臂，共 19 个输出。q1 可因 RGB/PC/shuffle 模式不同而不同，必须逐模式存储，不允许沿用 R7 的“所有 query 都必须 argmax”断言。w=0 时应回退 q1 原始框；只有关联本身失败/无候选时 q1 才回退 q0。尺寸 prior 对应 q1，不是 q0。

必须分别报告 association−q0、directG−association、parameter−association、parameter−directG 和 parameter−outputspace。换 query、组内聚合、部分框步以及参数更新的收益不能合并归因。简单 cached-final-linear SGD 的等价框步本来就是强同信息对照。

## 机制反例及必要边界

- 多个 proposal 来自同一模型和点云，组内 robust center consensus 是相关 source 预测的自聚合，不是独立几何真值。成员数不能解释为独立观测数。directG 对照承担全部聚合信息。
- 观测支持 Jaccard、AABB overlap 与空间 complete-link 仍可能将紧邻实体合并，或把一个实体分裂成多个组。组 ID 只是算法分组，不是真实实例 ID。候选/选中组 oracle 只能封存后用 target IoU 做覆盖诊断；不能以 GT 纯度选择关联或修改聚类。
- 冻结 CLIP tight/context 与 full-caption cosine 仅提供语义证据。它不是通用 2D grounding、关系推理或 3D teacher。必须记录模型/processor/tokenizer身份、实际文本截断（包括 CLIP77-token限制）、tight/context裁剪及融合规则、不可见/裁剪过小/背景支配/相同类别多实例等失败条件。
- 原始图片分辨率、K/E/平台预旋转、畸变缺失约定仍需按已审计接口处理。不得根据 GT 去修正标定；缺 D 的 assume-zero 是显式近似而非已恢复物理无畸变。
- 旧 adapter 的 loss=sum(w×SmoothL1)/sum(w)。统一 global trust 乘所有有效轴会被分母抵消，只能控制 eligibility，不能宣称连续缩小更新。若 driver 预锁学习率再乘 trust，应记录 actual lr、trust、q1 feature norm 和所有对照的相同规则；未锁前不能写作已采用方案。
- 原生 yaw0；当前尺寸权重全0、保留 q1 shape prior。局部可见点或 source 组的聚合不得宣称恢复隐藏完整物体尺寸。

## 精度、复现与门禁

所有真实 source、关联、回归和对照使用 cuDNN TF32=False、matmul TF32=False、cuDNN benchmark=False；其余标志和非决定性算子状态如实记录。该组合得到 R7 工程正控支持，但并没有单独隔离两个标志的因果贡献，也不保证跨进程 query 槽身份。

smoke 应按固定 hash 覆盖全部五个场景，在 source/GT 评分前锁定，并只查看无标签工程信息：真实 source 重放、CLIP冻结与 ROI 对应、q0/q1、去重成员、投影、G接受、非零伪目标、真实梯度/参数变化、等价框步及恢复。非零 G/更新不足可作为机制停止条件，但不能为提高比例按 GT 放宽门槛。full 配置在新 GT 评分前冻结，完整预测封存后才授权评分；smoke 不先产生新 IoU 来影响 full。

所有真实 source 和各臂必须共用本样本同一次 source cache。R7 已观测跨进程同 P 仍有 pool 槽漂移；不能复用旧池或按索引跨进程宣称同一物理 proposal。smoke/full 重复样本如有漂移应另记，不修改全量预测或替换基线。

## 独立评分与私有数据边界

输入/适应只用 P[:,:3]、RGB、全文 q、标定白名单和 source proposals；GT class/positive_map 不进入 G。包含 ground_info 的相机容器采用流式白名单，禁止将其语义字段作为在线输入。R5 私有 scoring index 仅在明确 stage CPU_SCORE_RELEASE 且全量 seal/code/sourcecheckpoint/CLIP资产与清单 hash 已核后用于离线目标框评分。

任务 IoU 为官方 source/model 坐标下 native yaw0，对 evaluation size 作原协议最小1e-6，仅评分用。无 PCA、GT yaw 或 GT query 选择。所有方法选中框用独立3D convex复核；全256 source pool覆盖用已审计 batch float64 几何。K>1 与 cluster-best 为 GT辅助诊断，不是在线选择器或物理实例标签。

平台分别报告微平均、等场景平均和 scene-cluster paired delta；空分母保留 null，全部失败、跳过、q变化、非零伪目标、执行步骤和真实参数变化分别统计。q0 baseline与q1 association baseline的比较掩码明确。所有负向例保留；公开文件仅 aggregate/scalar/hash，原图、caption、P、GT、原始候选和 teacher-like 分布不直接公开。

这是实现前协议草案，不是 R8 代码通过或真实 TTA 有效性结论。后续独立审计绑定实际源码、计划、缓存与封存收据。
