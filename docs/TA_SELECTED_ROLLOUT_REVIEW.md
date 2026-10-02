# Selected Student Rollout：完整参数梯度实现专家选中的学生 tube

**实际结果（未来 corrupt nonexpert）**：vidstg G−A -0.1705 [-0.6130, +0.2333] pp；hc2 G−A -0.1213 [-0.2320, -0.0150] pp。区间为95%配对source-bootstrap；裁决限于本次曝光开发面板。

本轮仅A/G。A原Rank-RKL；G由专家奖励first-argmax选一个student-generated tube，用停止梯度的该tube与central之间native L1+GIoU求完整1792参数VJP。G的更新长度仍取自己当前状态的lr×原RKL梯度范数，方向换成负selected-target梯度；不沿probe轴，不复制旧A轨迹范数。central被选中或exact flat时G精确no-op。

各数据集原32历史开发来源、一query/源、双序、clean+五类5%瞬时部署corruption、25%专家；384到达/臂/数据集，共1536。官方同域checkpoint、原Paper48帧/像素、两offset、H与Sa2VA/UniversalVTG缓存、九probe、1792空间参数不变。Vid K1/HC K8和各自封存lr/teacher温度保留；空间state流内继承，当前输出先于更新，时间Fast rerank不变。

全部四stream预测共同封存后才CPU读已曝光GT。10000次配对source-bootstrap，seed20261001；不当作fresh全量。A每个384到达逐值复现前轮A。first exact argmax处理并列，flat精确spread=0；RKL旧rank tie tolerance1e-12未改，二者含义分开。

## vidstg

封存参数：`{"lr": 0.033761698432507946, "rho": 0.05, "teacher_temperature": 0.34902548789596055, "steps": 1, "student_temperature": 1.0, "direction_count": 4}`。本轮下一步开发优先臂 **A**，只按预登记未来nonexpert均值规则，未晋升生产。

| 臂 | 全部corrupt ΔvIoU vs Frozen (pp,95%CI) | 未来nonexpert ΔvIoU vs Frozen | 未来 ΔvIoU vs A |
|---|---:|---:|---:|
| A | +1.2384 [-0.4533, +3.3695] | +0.6652 [-0.4635, +2.3858] | — |
| G | +1.0245 [-0.5125, +2.9187] | +0.4947 [-0.4495, +1.9196] | -0.1705 [-0.6130, +0.2333] |

| 臂 | 相对A未来gross gain/loss (pp) | 相对Frozen未来gross gain/loss (pp) | 相对A未来arrival损害>5pp | 首步唯一有益首选但更新受损 |
|---|---:|---:|---:|---:|
| A | 0.0000/0.0000 | 1.2209/0.5557 | 0/240 | 3/75 |
| G | 0.3745/0.5450 | 1.1054/0.6107 | 4/240 | 3/75 |

| 臂 | 平均框坐标绝对变化 | output-space cosine平均/正/负/未定义 | selected距离下降/eligible | 平均step固定时间GT ΔvIoU(pp) |
|---|---:|---:|---:|---:|
| A | 0.02251689 | +0.628393/60/4/11 | 5/75 | +0.937705 |
| G | 0.01974996 | +0.730557/64/0/11 | 7/75 | +0.410280 |

G no-op原因计数：`{"flat_rewards": 11, "central_selected": 0, "zero_selected_gradient": 0, "zero_rkl_magnitude": 0}`；幅度可匹配step 64/75。零output或零target方向余弦记null并另报，不置零混进均值。HC八步都保存动态候选/梯度/目标，非空no-op不早停。

## hc2

封存参数：`{"lr": 0.006097133675874025, "rho": 0.05, "teacher_temperature": 1.0, "steps": 8, "student_temperature": 1.0, "direction_count": 4}`。本轮下一步开发优先臂 **A**，只按预登记未来nonexpert均值规则，未晋升生产。

| 臂 | 全部corrupt ΔvIoU vs Frozen (pp,95%CI) | 未来nonexpert ΔvIoU vs Frozen | 未来 ΔvIoU vs A |
|---|---:|---:|---:|
| A | +1.0970 [-1.3329, +4.4006] | +1.3149 [-0.9088, +4.6787] | — |
| G | +0.9897 [-1.4133, +4.2656] | +1.1936 [-1.0020, +4.5076] | -0.1213 [-0.2320, -0.0150] |

| 臂 | 相对A未来gross gain/loss (pp) | 相对Frozen未来gross gain/loss (pp) | 相对A未来arrival损害>5pp | 首步唯一有益首选但更新受损 |
|---|---:|---:|---:|---:|
| A | 0.0000/0.0000 | 2.1065/0.7915 | 0/240 | 2/75 |
| G | 0.0744/0.1957 | 2.0531/0.8594 | 0/240 | 12/75 |

| 臂 | 平均框坐标绝对变化 | output-space cosine平均/正/负/未定义 | selected距离下降/eligible | 平均step固定时间GT ΔvIoU(pp) |
|---|---:|---:|---:|---:|
| A | 0.00108483 | +0.725272/566/31/3 | 559/600 | +0.044522 |
| G | 0.00112855 | +0.744948/582/18/0 | 570/600 | +0.023876 |

G no-op原因计数：`{"flat_rewards": 0, "central_selected": 0, "zero_selected_gradient": 0, "zero_rkl_magnitude": 0}`；幅度可匹配step 600/600。零output或零target方向余弦记null并另报，不置零混进均值。HC八步都保存动态候选/梯度/目标，非空no-op不早停。

## 本轮机制判断

两集开发优先仍是A。G−A在Vid为负均值且95%区间跨0；HC为负均值且本面板95%配对区间在0以下。不能将曝光开发的区间当作fresh验证或排除所有selected-rollout方法。

Vid G64次有效移动全部positive cosine（平均.73056），但只7次缩短selected几何距离，57次增大；A为5次下降。方向为正仍可能超过近邻目标或沿非欧氏损失路径移动，不能由cosine单独认定已实现有益的tube纠正。G平均局部GT gain降到+.41028pp（A+.93770），局部gross loss .55201pp（A .18354），并新增1个>5pp局部负步；future相对A4/240到达损害>5pp。

HC G582/600步positive cosine、570/600步selected距离下降（A559/600），但固定时间GT受损167→261，selected目标GT有益而执行有害19→104；首步唯一有益首选却更新受损2→12。selected距离下降同时GT损害143→242。框运动并未像E/F缩小13–18倍，平均movement .00112855对A .00108483。因此本结果不支持把问题只归结为四轴低敏感性或无法移动decoder输出；匹配RKL幅度、完整clip几何与稀疏事件支持、critic正确性/共享state转移仍是竞争解释。

上述局部计数来自各自不同演化轨迹，不是同一teacher实例的因果替换。若下一步研究event-conditioned证据，需另锁专家取帧/时间支持及对照；本轮没有据GT选择在线frame、gate或自动新作业。

primary未来子集各240到达，覆盖Vid32来源、HC30来源；HC另2来源在两序均处于expert位置，不进future主指标，仍保留在全组结果。

## 测量含义与判断边界

box movement为封存normalized cxcywh逐帧四坐标的FP64平均绝对差；cosine比较post−central与selected−central。独立CPU重建全部FP32参数执行、raw rewards/ranks、两个几何目标、functional scalars、状态链与dense指标。selected D下降/方向为正仍不等于GT改善；固定时间GT只解释局部动作，不替换当前pre-update线上输出。

G保留专家选择的student hard target，不直接拟合Sa2VA box，但仍是自生成target学习。两条轨迹分叉后自己的RKL范数不同，central/flat no-op又改变更新次数；胜出只能支持这整个执行规则，不能唯一归因为方向或声称跨臂每step长度完全相同。负结果也不排除其他尺度、函数参数化或证据支持方式。

完整1792D VJP避免四轴限制，但有限步长、L1/GIoU非光滑、native routing及critic误选仍可能使输出方向和任务效用分离。坏步减少需与gross gain/loss和负尾共同看；不以单个坏例或条件GT分组构造线上阈值。

两集均为原多轮历史曝光32来源的小开发面板，未加独立确认或全量。clean/control及所有arrival匿名结果、正负cases、K8每步和整个arrival效用均保留。优先臂选择按strict primary mean（1e-12 tie保留A），不等于统计显著性或论文普适结论。

G仍计算RKL counterfactual以及eligible noncentral selected-target额外backward，没有backward效率宣称。零新backbone/专家推理；完整replay/provider/两类backward和worker wall见COSTS.json。wall含载入、I/O和检查，不称纯GPU kernel。

无GT smoke首轮diagnostic日志先FP32相减产生独立FP64核对差异，保存functional_logging_001并仅改日志精度，正式预测前revision001重新pin，模型/目标/更新无改变。

文献联系仅为机制动机：[KL-free OPD](https://arxiv.org/abs/2609.33791)分析LLM token更新方向；[Best-of-N teacher rollout](https://arxiv.org/abs/2605.09725)使用teacher轨迹correctness优先选择并含GT recovery。本轮student tube selector不继承其保证，也不使用GT在线过滤。

本轮不自动启动事件条件Sa2VA取帧、top2、时间head、critic训练、超参grid、新模型/baseline或旧fullquery队列；保持生产注册与所有旧证据。

