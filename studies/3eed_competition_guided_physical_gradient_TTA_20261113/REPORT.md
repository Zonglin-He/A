# R93 — 固定R91主线，让竞争证据直接约束基础分数的更新方向

北京时间 2026-11-13 15:23。全DEV336为历史暴露开发，2Drone/3Quad场景，无独立确认、无FINAL。

固定主要方法 competition_projected_bt：**22.915325%**；相对自身R91零步-0.116720pp，相对同证据直接使用+0.078064pp，相对原R92物理BT +0.042932pp。

## 固定内容与本轮新检验

唯一主线R91_R76_full_disc，保持源域已选权重、双向结构、主体/属性/参照读取、新判别接口和原A256不变。不重训source，不换encoder/decoder/matching，不用外部视觉专家或目标GT。只更新原102498个Adapter/交互投影/门控。沿用R92的静态当前输入残差r0，基础b(theta)可变化。R92显示BT损伤可由基础分数变化复现；R93因此新增能直接对基础分数产生非零梯度的竞争证据，而不继续加零起点残差保持。

## 弱监督的来源、身份边界与同信息直接对照

零步同一已训练R91模型读取完整输入和原互补遮挡76001/2，三视图使用被保留的相同空间锚点，不跨描述沿用query编号。按完整与两遮挡均值排名交替，至多4个空间假设，保留去重：框IoU>=.9，或框IoU>=.5且点Jaccard>=.5视为重复假设。合格候选对要求各有>=2可见支持点、相互框IoU<=.1和点Jaccard<=.1。它们只是空间支持分离的假设，稀疏/偏框可能仍混物体；不能称GT实例。

仅当两个遮挡下的完整候选分数差和两个遮挡下的竞争残差差值均同号且绝对值>1e-8时接受弱成对证据。目标t_ij取两遮挡完整分数差均值，停止梯度。没有要求随机描述让排名反转，没有把当前第一名当正确对象；接受偏好可以反对它。但同模型可以共同依赖错误证据，因此一致性不是正确性认证。缺证据记UNKNOWN，仍走原物理更新而非回退冻结模型，保留总分分母。

任务loss为0.5*mean[((btheta+r0)_i-(btheta+r0)_j-t_ij)^2]，真实梯度穿过冻结3EED后缀到原轻量theta。主方法没有把二选一偏好升级成全候选正确标签。对照current_top1在相同反馈覆盖下刻意使用原第一名CE，用于检查是否仅在保护已有判断；它不是主方法。反向偏好控制在相同候选对上把t取负，不改变覆盖或更新预算。

无更新直接使用：在原全256维分数s0上，求满足同一接受成对间隔的最近向量s，通过小型incidence矩阵最小范数解；未涉及的候选不获无依据共同加分。它不训练参数、也不宣称局部偏好足以确定全局身份。冻结三观测概率/表示平均是另一组同视图无更新参照；R59/ABC沿用历史不同采样，不冒充这组遮挡。

## 更新方向、幅度和验收

gP为原物理恢复0.1P+0.1RGB+0.01depth梯度，gC为竞争pair梯度。c=stopgrad(||gP||/||gC||)，仅作初始梯度尺度平衡，不用GT调系数。主方法先去掉gP中与gC冲突的分量，再加c*gC；sum对照不投影，only对照只用竞争梯度；无合格信号时均使用物理更新。wrong_binding只错配原物理绑定，竞争证据不变。

所有方法提案的L2半径与该输入原物理raw步骤相同，避免通过增加步幅获益。raw直接应用，BT仍为原9档scale=2^-k，Armijo c1e-4/tol1e-10。验收目标为physical_weight*physical+c*task+原初始输出KL/0.1query保持，预测和保持使用同一btheta+r0。Armijo方向导数用真正目标梯度gP+c*gC，不误用手术后的向量。竞争更新可能暂时增加物理误差，不能把它暗改成只允许恢复下降。第三遮挡76003只在独立审计后测主方法恢复，从不参与接受。

这是借鉴PCGrad冲突投影思想的非对称、同半径任务迁移，未复现原PCGrad双边投影或GEM标签记忆，不继承其理论保证。即使优化满足目标，也可能拟合错误偏好。

## 全DEV336任务结果

| 方法 | Drone mIoU % | Quad mIoU % | 等权mIoU % | 等权Acc25 % | 等权Acc50 % |
|---|---:|---:|---:|---:|---:|
| source | 15.539892 | 28.120770 | 21.830331 | 40.773810 | 21.130952 |
| R76_frozen | 17.083330 | 28.139863 | 22.611596 | 41.071429 | 21.428571 |
| ABC | 15.736332 | 28.868588 | 22.302460 | 42.261905 | 21.428571 |
| R59 | 16.385568 | 28.836664 | 22.611116 | 43.154762 | 21.726190 |
| R80_raw | 17.128930 | 28.791222 | 22.960076 | 42.261905 | 21.130952 |
| R82_bt | 17.358929 | 28.163165 | 22.761047 | 41.666667 | 21.428571 |
| R91_frozen | 17.346189 | 28.717901 | 23.032045 | 41.964286 | 21.726190 |
| R91_mean3 | 17.145428 | 28.802929 | 22.974178 | 41.964286 | 21.726190 |
| R91_feature3 | 17.145428 | 28.802929 | 22.974178 | 41.964286 | 21.726190 |
| R92_bt | 17.073866 | 28.670921 | 22.872394 | 41.964286 | 21.726190 |
| R92_raw_static | 17.039432 | 29.069303 | 23.054367 | 42.559524 | 21.428571 |
| competition_direct | 17.026903 | 28.647619 | 22.837261 | 41.666667 | 21.726190 |
| reverse_preference_direct | 15.070138 | 19.844674 | 17.457406 | 33.630952 | 14.880952 |
| competition_projected_raw | 16.941057 | 29.466507 | 23.203782 | 42.559524 | 21.428571 |
| competition_projected_bt | 17.159730 | 28.670921 | 22.915325 | 41.964286 | 21.726190 |
| competition_sum_raw | 16.941057 | 29.466507 | 23.203782 | 42.559524 | 21.428571 |
| competition_sum_bt | 17.159730 | 28.670921 | 22.915325 | 41.964286 | 21.726190 |
| competition_only_raw | 17.150753 | 29.271380 | 23.211066 | 42.857143 | 21.130952 |
| competition_only_bt | 17.343488 | 28.670921 | 23.007204 | 42.261905 | 21.726190 |
| current_top1_raw | 17.369719 | 29.247600 | 23.308659 | 42.857143 | 22.023810 |
| current_top1_bt | 17.343488 | 28.741204 | 23.042346 | 42.261905 | 21.726190 |
| reverse_preference_raw | 17.252870 | 28.771415 | 23.012143 | 42.559524 | 20.833333 |
| reverse_preference_bt | 17.426762 | 28.670921 | 23.048842 | 42.261905 | 21.726190 |
| wrong_binding_raw | 16.727005 | 29.433350 | 23.080177 | 41.964286 | 21.726190 |
| wrong_binding_bt | 17.275936 | 28.670921 | 22.973428 | 41.964286 | 21.726190 |

## 配对净收益与所有负例

| 类型 | 方法 | 对照 | 平台 | Δpp | 改善/恶化/不变 | 换query |
|---|---|---|---|---:|---:|---:|
| primary | competition_projected_bt | R91_frozen | balanced | -0.116720 | 2/3/331 | 9 |
| primary | competition_projected_bt | R91_frozen | drone | -0.186460 | 1/2/165 | 7 |
| primary | competition_projected_bt | R91_frozen | quad | -0.046980 | 1/1/166 | 2 |
| primary | competition_projected_bt | R91_mean3 | balanced | -0.058853 | 1/3/332 | 15 |
| primary | competition_projected_bt | R91_mean3 | drone | +0.014301 | 1/1/166 | 10 |
| primary | competition_projected_bt | R91_mean3 | quad | -0.132008 | 0/2/166 | 5 |
| primary | competition_projected_bt | R91_feature3 | balanced | -0.058853 | 1/3/332 | 14 |
| primary | competition_projected_bt | R91_feature3 | drone | +0.014301 | 1/1/166 | 10 |
| primary | competition_projected_bt | R91_feature3 | quad | -0.132008 | 0/2/166 | 4 |
| primary | competition_projected_bt | competition_direct | balanced | +0.078064 | 2/2/332 | 12 |
| primary | competition_projected_bt | competition_direct | drone | +0.132826 | 1/2/165 | 10 |
| primary | competition_projected_bt | competition_direct | quad | +0.023302 | 1/0/167 | 2 |
| primary | competition_projected_bt | R92_bt | balanced | +0.042932 | 1/0/335 | 6 |
| primary | competition_projected_bt | R92_bt | drone | +0.085864 | 1/0/167 | 5 |
| primary | competition_projected_bt | R92_bt | quad | +0.000000 | 0/0/168 | 1 |
| primary | competition_projected_bt | R92_raw_static | balanced | -0.139042 | 9/10/317 | 46 |
| primary | competition_projected_bt | R92_raw_static | drone | +0.120298 | 8/5/155 | 30 |
| primary | competition_projected_bt | R92_raw_static | quad | -0.398382 | 1/5/162 | 16 |
| primary | competition_projected_bt | competition_sum_bt | balanced | +0.000000 | 0/0/336 | 0 |
| primary | competition_projected_bt | competition_sum_bt | drone | +0.000000 | 0/0/168 | 0 |
| primary | competition_projected_bt | competition_sum_bt | quad | +0.000000 | 0/0/168 | 0 |
| primary | competition_projected_bt | competition_only_bt | balanced | -0.091879 | 1/1/334 | 4 |
| primary | competition_projected_bt | competition_only_bt | drone | -0.183758 | 1/1/166 | 4 |
| primary | competition_projected_bt | competition_only_bt | quad | +0.000000 | 0/0/168 | 0 |
| primary | competition_projected_bt | current_top1_bt | balanced | -0.127020 | 1/2/333 | 5 |
| primary | competition_projected_bt | current_top1_bt | drone | -0.183758 | 1/1/166 | 4 |
| primary | competition_projected_bt | current_top1_bt | quad | -0.070283 | 0/1/167 | 1 |
| primary | competition_projected_bt | reverse_preference_bt | balanced | -0.133516 | 1/1/334 | 4 |
| primary | competition_projected_bt | reverse_preference_bt | drone | -0.267032 | 1/1/166 | 3 |
| primary | competition_projected_bt | reverse_preference_bt | quad | +0.000000 | 0/0/168 | 1 |
| primary | competition_projected_bt | wrong_binding_bt | balanced | -0.058103 | 2/1/333 | 9 |
| primary | competition_projected_bt | wrong_binding_bt | drone | -0.116206 | 2/1/165 | 9 |
| primary | competition_projected_bt | wrong_binding_bt | quad | +0.000000 | 0/0/168 | 0 |
| primary | competition_projected_bt | competition_projected_raw | balanced | -0.288457 | 13/12/311 | 48 |
| primary | competition_projected_bt | competition_projected_raw | drone | +0.218673 | 11/4/153 | 30 |
| primary | competition_projected_bt | competition_projected_raw | quad | -0.795586 | 2/8/158 | 18 |
| direct_feedback | competition_direct | R91_frozen | balanced | -0.194784 | 2/3/331 | 11 |
| direct_feedback | competition_direct | R91_frozen | drone | -0.319286 | 2/2/164 | 9 |
| direct_feedback | competition_direct | R91_frozen | quad | -0.070283 | 0/1/167 | 2 |
| direct_preference_direction | competition_direct | reverse_preference_direct | balanced | +5.379855 | 64/18/254 | 121 |
| direct_preference_direction | competition_direct | reverse_preference_direct | drone | +1.956766 | 16/7/145 | 41 |
| direct_preference_direction | competition_direct | reverse_preference_direct | quad | +8.802945 | 48/11/109 | 80 |
| physical_contribution | competition_sum_bt | competition_only_bt | balanced | -0.091879 | 1/1/334 | 4 |
| physical_contribution | competition_sum_bt | competition_only_bt | drone | -0.183758 | 1/1/166 | 4 |
| physical_contribution | competition_sum_bt | competition_only_bt | quad | +0.000000 | 0/0/168 | 0 |

## 反馈覆盖与可纠错机会

覆盖与辅助loss只解释任务结果，不据此采用方法或筛掉失败。反馈可能只约束无关低分项；偏好反对零步第一名的数量和直接读出换第一名的数量单独列出，不把全部接受pair称困难纠错。

| 平台 | n | 空间资格样本 | 接受反馈样本 | pair数 | 含反对原第一名偏好的样本 | 直接换Top1 |
|---|---:|---:|---:|---:|---:|---:|
| pooled | 336 | 219 | 166 | 366 | 2 | 11 |
| drone | 168 | 97 | 69 | 143 | 2 | 9 |
| quad | 168 | 122 | 97 | 223 | 0 | 2 |

## 梯度、验收与方法组件

| 方法 | task激活样本 | 梯度冲突样本 | cosine均值(task激活) | 接受BT数 | 平均接受scale | 初始/端点目标均值 |
|---|---:|---:|---:|---:|---:|---:|
| competition_projected | 166 | 70 | 0.05825822 | 336 | 0.26040504 | 0.03505634/0.03482026 |
| competition_sum | 166 | 70 | 0.05825822 | 336 | 0.27219355 | 0.03505634/0.03480391 |
| competition_only | 166 | 70 | 0.05825822 | 336 | 0.22005208 | 0.01675794/0.01657985 |
| current_top1 | 166 | 106 | -0.09229722 | 336 | 0.28547015 | 0.05022019/0.04994107 |
| reverse_preference | 166 | 72 | 0.07324399 | 336 | 0.29853748 | 0.03891273/0.03861450 |
| wrong_binding | 166 | 68 | 0.07782432 | 336 | 0.34597052 | 0.05314008/0.05278831 |

主方法的第三遮挡物理恢复在331/336条改善。它是同一观测的未拟合遮挡，不是新相机视角或独立场景。

## 五个历史场景

| 平台/场景 | n | 方法 | mIoU % |
|---|---:|---|---:|
| drone/D_S1 | 106 | R91_frozen | 13.989152 |
| drone/D_S1 | 106 | R92_bt | 13.557545 |
| drone/D_S1 | 106 | competition_direct | 13.456692 |
| drone/D_S1 | 106 | competition_projected_bt | 13.693631 |
| drone/D_S1 | 106 | competition_only_bt | 13.984870 |
| drone/D_S1 | 106 | current_top1_bt | 13.984870 |
| drone/D_S1 | 106 | reverse_preference_bt | 14.116852 |
| drone/D_S2 | 62 | R91_frozen | 23.085641 |
| drone/D_S2 | 62 | R92_bt | 23.085641 |
| drone/D_S2 | 62 | competition_direct | 23.130813 |
| drone/D_S2 | 62 | competition_projected_bt | 23.085641 |
| drone/D_S2 | 62 | competition_only_bt | 23.085641 |
| drone/D_S2 | 62 | current_top1_bt | 23.085641 |
| drone/D_S2 | 62 | reverse_preference_bt | 23.085641 |
| quad/Q_S1 | 62 | R91_frozen | 10.708624 |
| quad/Q_S1 | 62 | R92_bt | 10.771765 |
| quad/Q_S1 | 62 | competition_direct | 10.708624 |
| quad/Q_S1 | 62 | competition_projected_bt | 10.771765 |
| quad/Q_S1 | 62 | competition_only_bt | 10.771765 |
| quad/Q_S1 | 62 | current_top1_bt | 10.771765 |
| quad/Q_S1 | 62 | reverse_preference_bt | 10.771765 |
| quad/Q_S2 | 49 | R91_frozen | 35.770023 |
| quad/Q_S2 | 49 | R92_bt | 35.529054 |
| quad/Q_S2 | 49 | competition_direct | 35.529054 |
| quad/Q_S2 | 49 | competition_projected_bt | 35.529054 |
| quad/Q_S2 | 49 | competition_only_bt | 35.529054 |
| quad/Q_S2 | 49 | current_top1_bt | 35.770023 |
| quad/Q_S2 | 49 | reverse_preference_bt | 35.529054 |
| quad/Q_S3 | 57 | R91_frozen | 42.244590 |
| quad/Q_S3 | 57 | R92_bt | 42.244590 |
| quad/Q_S3 | 57 | competition_direct | 42.244590 |
| quad/Q_S3 | 57 | competition_projected_bt | 42.244590 |
| quad/Q_S3 | 57 | competition_only_bt | 42.244590 |
| quad/Q_S3 | 57 | current_top1_bt | 42.244590 |
| quad/Q_S3 | 57 | reverse_preference_bt | 42.244590 |

## 固定回访与总体正负尾部

固定回访R92_CASE002/003/004/005（分别涉及真实实例误改、旧raw纠错被BT压回、同物体框改善和退化），再按主方法/直接使用/旧物理BT的总体最大正负贡献各至多两条选页。案例只解释，不生成部署条件。

| 案例 | 平台 | 接受pairs | 零步 | 旧物理BT | 同证据直接 | 主TTA | currentTop1控制 | reverse控制 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CASE001 | drone | 4 | 0.452964 | 0.000000 | 0.000000 | 0.000000 | 0.452964 | 0.452964 |
| CASE002 | quad | 1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| CASE003 | quad | 0 | 0.233223 | 0.272370 | 0.233223 | 0.272370 | 0.272370 | 0.272370 |
| CASE004 | quad | 1 | 0.446062 | 0.327987 | 0.327987 | 0.327987 | 0.446062 | 0.327987 |
| CASE005 | drone | 1 | 0.000000 | 0.000000 | 0.144251 | 0.144251 | 0.000000 | 0.144251 |
| CASE006 | drone | 2 | 0.255694 | 0.255694 | 0.000000 | 0.255694 | 0.255694 | 0.255694 |
| CASE007 | drone | 1 | 0.246800 | 0.246800 | 0.274807 | 0.246800 | 0.246800 | 0.246800 |
| CASE008 | drone | 0 | 0.005975 | 0.001437 | 0.005975 | 0.001437 | 0.001437 | 0.001437 |

## 测量与成本边界

预测封存后，独立原生审计检查4704读出端点和7267实际trial；概率最大误差1.11022302463e-15、参数0、loss0、Armijo界0、梯度/独立NumPy投影1.49011611938e-08，Top1不一致0。FP32容差1e-6/参数1e-7，Top1必须相同。

生产整组25臂317.456s，trial原生前向7267，trial中物理loss例程14534。计数native_proposal_calls=1008实际表示三类提案例程（正确/错配物理、竞争梯度）的调用次数，不是其内部所有native前向数。初始三视图复用同权重R91缓存，独立审计重新生成；该时间不是冷启动部署端到端时延。

不新增源域训练、专家或目标标签。全部样本、UNKNOWN（本轮指无合格弱偏好）、物理后备、拒绝trial与负比较保留。没有为了不下降大面积恢复冻结模型。全DEV已反复暴露，场景聚类列出但不构造独立总体CI。单一配置结果不改变已选R91架构；若投影不优于sum，不能把收益归于梯度手术；若反向偏好同样有效，不能声称语义反馈可靠；若only优于联合，物理项增量未建立。当前物体身份只能由私有原图核验，不能从IoU排序单独充当身份GT。

## 本轮裁决：把收益与组件依据分开

预先固定主方法22.915325%，相对零步-0.116720pp，相对同信息直接使用+0.078064pp；不在读完开发分数后把最高分对照改名为主方法。保留所有raw/BT负结果，未再搜索系数、层或门槛。

投影相对直接梯度相加+0.000000pp；物理联合相对只用竞争任务-0.091879pp；正确偏好相对反向偏好-0.133516pp；相对当前第一名保护-0.127020pp。这些是总体任务增量，不能以冲突个数或loss下降替代。

反馈166/336，接受366对，但只有2条输入出现反对原第一名的偏好。此设计大部分是在确认/校准已有偏好，尚不是广覆盖的实例纠错监督。直接使用只改11条Top1；局部pair不覆盖第三候选时，直接使用也可能抬出未核验的第三候选，不把它自动设为负例。

仅在预测封存后的IoU诊断：接受pair中偏好较高IoU者190对、偏好较低IoU者79对，两者都零97对。IoU只是相对定位效用，不能单独认证主体身份；这些结果没有进入接收、更新或停止。

| 描述性子集 | 方法 | 对照 | n | Δpp | 改善/恶化 | 换query |
|---|---|---|---:|---:|---:|---:|
| feedback | competition_projected_bt | R91_frozen | 166 | -0.257101 | 1/2 | 5 |
| feedback | competition_projected_bt | competition_direct | 166 | +0.137161 | 1/1 | 8 |
| feedback | competition_direct | R91_frozen | 166 | -0.394262 | 2/3 | 11 |
| feedback | competition_projected_bt | R92_bt | 166 | +0.086898 | 1/0 | 6 |
| no_feedback | competition_projected_bt | R91_frozen | 170 | +0.020358 | 1/1 | 4 |
| no_feedback | competition_projected_bt | competition_direct | 170 | +0.020358 | 1/1 | 4 |
| no_feedback | competition_direct | R91_frozen | 170 | +0.000000 | 0/0 | 0 |
| no_feedback | competition_projected_bt | R92_bt | 170 | +0.000000 | 0/0 | 0 |
| opposes_zero | competition_projected_bt | R91_frozen | 2 | -22.648195 | 0/1 | 1 |
| opposes_zero | competition_projected_bt | competition_direct | 2 | +0.000000 | 0/0 | 1 |
| opposes_zero | competition_direct | R91_frozen | 2 | -22.648195 | 0/1 | 2 |
| opposes_zero | competition_projected_bt | R92_bt | 2 | +0.000000 | 0/0 | 0 |

该结果只评价本轮固定弱偏好、梯度平衡、同半径方向与验收组合，不否定所有任务相关TTA或既有R91源域架构。继续保留R91_R76_full_disc作为唯一架构锚点；本轮无独立确认，不宣布稳健采用，不自动启动后续训练/搜索。若反馈不能触及错误第一名，或反向/当前第一名控制不差，首先缺的是独立于原排序的有效实例纠错证据，而不是再提高任务梯度权重。

## 真实原图复核与未采用的raw正信号

主方法与sum在70/336条提案实际不同，23/336条接受scale改变、概率最大差0.00642822855，最终Top1差0/336。梯度手术确实执行，但未提供额外选框收益。

预注册raw为23.203782%，相对零步+0.171737pp，14改善/16恶化。Drone 16.941057%、Quad 29.466507%；前者比零步更低、后者提高。最大单例正贡献+0.184689pp。保留此开发正结果，但current_top1_raw更高，sum_raw与主raw相同，不能归因于正确竞争偏好或投影。没有事后将raw替换成锁定BT主方法。

| 案例 | 原图身份变化/不确定性 | 实际观察 |
|---|---|---|
| CASE001 | different_visible_objects | 原近处车辆仍被主TTA和直接偏好改到远处上沿区域。task-only和反向控制保住原框。该例接受反馈真实反对原第一名，故错误不只是物理项单独造成。 |
| CASE002 | same_candidate_wrong_visible_person | 主BT和直接使用仍选中另一行人；raw仍可改到左侧主体。唯一接受偏好不包含原第一名，未覆盖最终关键竞争者。不能由两候选偏好推出它超过未监督第一名。 |
| CASE003 | same_visible_object_duplicate_boxes | 同一SUV附近的重复框由0.2332改到0.2724，沿用原物理后备；该例没有接受pair，不归因于新增竞争监督。 |
| CASE004 | same_visible_object_duplicate_boxes | 仍是左侧同一行人附近较差重复框，0.4461降到0.3280。只保护原第一名的控制能避免，反向偏好BT仍未避免。 |
| CASE005 | different_visible_regions_target_identity_partial | 主TTA和直接读出从远处背景回到右侧车辆附近，但仅0.1443，目标边缘裁切，精确物体身份保留部分未知。胜出的是接受pair之外的第三候选，反向BT也得到相同框，不能认证为正确偏好带来的身份纠错。 |
| CASE006 | same_candidate | 主BT保持白色车辆原框，直接使用偏好则转向前景深色车辆并降到零。此项TTA相对直接的改善是避免直接重排损伤，没有超过零步。 |
| CASE007 | same_candidate | 主BT保留原SUV框；直接使用改到同一SUV更合适的重复框，0.2468到0.2748，跨过Acc25。参数更新没有执行这份有益变化。 |
| CASE008 | same_visible_object_duplicate_boxes | 同一前景车辆区域两个很差重复框从0.0060到0.0014；高IoU候选一直存在但排名很低。无接受pair，属于原物理后备，不能叫新增语义身份错误。 |

关键两例没有同时解决：原近处车辆的误改仍在；原可由raw纠正的行人，BT仍压回。大部分接受pair确认既有排序，少数真实反对第一名的反馈仍可指错。因此，不把下一步重新简化成“增大梯度”或“保护第一名”；首先需要在模型已有候选竞争中提供能覆盖实际错误第一名、且不依赖其同源自信的可测量纠错依据。当前结论不要求换架构、重训source或另扫层位。

整理案例统计时首次保存遇到NumPy整数不能直接写入JSON；保留失败版本与返回码，显式转换计数后重试完成。该问题发生在封存、原生审计和任务评分之后，未改变预测、算法、容差或评价分母。
