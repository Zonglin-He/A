# Negative evidence: matched isolated spatial writes and reset-u control

本轮四臂同 A 更新前状态的专家位置对照、下一 query 单次迁移，以及独立 reset-u 在线对照已完成封存、后验评分和独立审计。裁决见 DECISION.json；所有区间、负尾、参数变化强度和相反案例一并保留。当前空间反事实读出不等于已部署的 prequential 输出，单次未来迁移也不等于负证据完整在线流。

每集32开发源、16源互斥确认、每源一query、双序、clean及五种5% corruption。总1152到达，288专家位置含240corrupt/48clean；确认均有历史曝光。原官方同域EMA checkpoint、Paper48输入像素、缓存H与Uniform Sa2VA、source半径rho .05/D4的九probe、1792空间参数保持。Vid包 lr .033761698432507946 / teacher T .34902548789596055 / K1，HC包 lr .006097133675874025 / T1 / K8。rank_cross仅交换lr/T/K整包，不能确定是哪一个超参数造成差异；本数据集L1/GIoU系数Vid5/3、HC5/4保持。

当前/未来/每步空间GT诊断都固定该query的原封存A final interval。GT事件内有密集标注的实际观察帧另报 spatial IoU；无交集标为未知。完整事件sIoU仍保留原HC最后一标注帧inclusive，与时间span/scorer半开端点分开。正确性阈值按官方vIoU严格大于.3/.5。

## 当前专家位置：同前态、固定A时间

下表为corruption；source宏平均、10000次配对bootstrap seed20261004，单位pp。原Rank必须复现旧A的梯度/状态；新臂更新后当前框是隔离空间反事实读出。

| 面板 | arm−Rank native | ΔvIoU [95%CI] | >5pp / >20pp损害 | gross gain / loss pp |
|---|---|---:|---:|---:|
| vidstg/search (16源/80cells) | rank_cross | +0.1325 [-0.6635, +1.0452] | 0/0 | 0.5456/0.4130 |
| vidstg/search (16源/80cells) | negative_global | -0.8194 [-1.7278, -0.0758] | 6/0 | 0.1965/1.0159 |
| vidstg/search (16源/80cells) | negative_local | -0.8186 [-1.7266, -0.0754] | 6/0 | 0.1967/1.0153 |
| vidstg/confirm (8源/40cells) | rank_cross | -0.6734 [-1.3253, -0.0585] | 0/0 | 0.0806/0.7539 |
| vidstg/confirm (8源/40cells) | negative_global | -1.3049 [-2.5143, -0.1681] | 1/0 | 0.1106/1.4155 |
| vidstg/confirm (8源/40cells) | negative_local | -1.3044 [-2.5137, -0.1677] | 1/0 | 0.1107/1.4151 |
| hc2/search (14源/80cells) | rank_cross | +0.4924 [+0.0651, +1.0047] | 0/0 | 0.6059/0.1134 |
| hc2/search (14源/80cells) | negative_global | -0.3188 [-0.7595, +0.0824] | 0/0 | 0.1828/0.5016 |
| hc2/search (14源/80cells) | negative_local | -0.3183 [-0.7591, +0.0833] | 0/0 | 0.1832/0.5015 |
| hc2/confirm (7源/40cells) | rank_cross | -0.4249 [-1.3292, +0.5024] | 0/0 | 0.2643/0.6892 |
| hc2/confirm (7源/40cells) | negative_global | -0.6810 [-2.3008, +0.5480] | 1/0 | 0.3271/1.0081 |
| hc2/confirm (7源/40cells) | negative_local | -0.6418 [-2.2986, +0.6509] | 1/0 | 0.3656/1.0075 |

negative_local−negative_global直接比较同帧分布与全clip分布；两者lambda1、native lr/K相同。它改变损失空间支持，不是仅在同一候选上排序；不同arm后续K步状态与支持会自然分叉。负证据与Rank还同时改变teacher形式和有效梯度强度，损害减少不能唯一归因负证据语义。没有幅度匹配臂，也没有lambda扫参。

| 面板 | local−global ΔvIoU | global−更新前 ΔvIoU | local−更新前 ΔvIoU |
|---|---:|---:|---:|
| vidstg/search | +0.0009 [+0.0001, +0.0018] | -0.0000 [-0.0015, +0.0013] | +0.0009 [-0.0008, +0.0026] |
| vidstg/confirm | +0.0004 [+0.0001, +0.0008] | +0.0005 [-0.0002, +0.0011] | +0.0009 [+0.0001, +0.0018] |
| hc2/search | +0.0005 [+0.0001, +0.0010] | -0.0026 [-0.0079, +0.0011] | -0.0021 [-0.0070, +0.0014] |
| hc2/confirm | +0.0392 [-0.0003, +0.1167] | -0.0255 [-0.0874, +0.0099] | +0.0137 [+0.0001, +0.0326] |

## 负证据、teacher和实际梯度

逐有效观察帧e_jk=max(IoU(center_j,E_j)−IoU(candidate_kj,E_j),0)。global先逐帧取正部再平均，local每帧建立分布；不把正部平均改为平均后正部。q∝p0 exp(−e)，lambda固定1；KL(p||q)=KL(p||p0)+E_p[e]+logZ，最后项在单步中固定。e=0不是correctness标签；没有观察与全零证据显式零梯度。

teacher中相同e（尤其未决e=0）的候选odds保持，但有限SGD及共享参数可改变实际odds和未观察帧。center的e恒为0，归一化可能保留错误center，不能用该构造宣称GT正确性。误罚仅在GT事件内且有标注的观察帧判断；事件外或缺标注的证据不补成错误标签。full-event候选质量是另一个诊断，不能混作逐帧标注。

| 面板 / arm | steps | observed误罚条目/已知负条目 | observed误罚权重比例 | loss↓ / ↑ / = | mean实际step displacement |
|---|---:|---:|---:|---:|---:|
| vidstg/search/rank_native | 80 | 44/424 | 0.0778 | 74/1/5 | 0.403166 |
| vidstg/search/rank_cross | 605 | 493/3504 | 0.0959 | 597/3/5 | 0.0232469 |
| vidstg/search/negative_global | 80 | 44/424 | 0.0778 | 42/18/20 | 0.000328987 |
| vidstg/search/negative_local | 80 | 44/424 | 0.0778 | 53/7/20 | 0.000367588 |
| vidstg/confirm/rank_native | 40 | 46/212 | 0.1438 | 40/0/0 | 0.244759 |
| vidstg/confirm/rank_cross | 320 | 374/1698 | 0.1617 | 320/0/0 | 0.0135768 |
| vidstg/confirm/negative_global | 40 | 46/212 | 0.1438 | 33/7/0 | 9.67691e-05 |
| vidstg/confirm/negative_local | 40 | 46/212 | 0.1438 | 37/3/0 | 0.000139618 |
| hc2/search/rank_native | 605 | 758/2038 | 0.3001 | 594/6/5 | 0.0205958 |
| hc2/search/rank_cross | 80 | 86/246 | 0.2220 | 74/1/5 | 0.42086 |
| hc2/search/negative_global | 605 | 688/1968 | 0.2212 | 393/207/5 | 0.00010842 |
| hc2/search/negative_local | 605 | 688/1968 | 0.2213 | 486/114/5 | 9.91771e-05 |
| hc2/confirm/rank_native | 320 | 779/2003 | 0.2329 | 316/4/0 | 0.0254851 |
| hc2/confirm/rank_cross | 40 | 109/254 | 0.3582 | 40/0/0 | 0.600023 |
| hc2/confirm/negative_global | 320 | 883/2047 | 0.4058 | 198/122/0 | 0.000167252 |
| hc2/confirm/negative_local | 320 | 872/2034 | 0.3452 | 244/76/0 | 0.000118627 |

这些是重复step/candidate/观察的诊断分母，不能当独立来源样本量。每步用同一冻结q评估before/after；下一步刷新target，不能把第一步loss与最后一步loss视为同一目标的优化曲线。目标上升保留，loss下降但GT下降另计，均不作为选择器或GT门控。EXECUTION_DIAGNOSTICS保留实际参数位移、eta×梯度范数、观察/未观察输出梯度与框移动、GT较好候选概率变化及未决odds的target误差/实际变化。

query residual256与norm1/norm3/norm4共1536 LN参数的四个block-only反事实，在其余块固定pre的条件下重放。块效应不是可加的Jacobian分解；框变化可能经过共享decoder/空间读出，不能按参数名称直接归因为局部/全局。SUMMARY和ROWS保留全部块效果。

## 下一非专家query：隔离的一次写迁移

全部288专家位置的下一到达存在且非专家。无写baseline把当前专家write的共同A prestate带到未来query；rank_native写后状态恰为旧A未来query prestate。每arm只带入当前一次写后的状态；每个probe结束后丢弃，未形成新的长期轨迹。主CI按write-source配对，另给target-source敏感性；write与target复用可能留下两种单向bootstrap都未完全表达的依赖。

| 面板 | arm−Rank native | ΔvIoU write-source CI | target-source CI |
|---|---|---:|---:|
| vidstg/search | rank_cross | +0.1588 [-0.1636, +0.4927] | +0.1612 [-0.1815, +0.5180] |
| vidstg/search | negative_global | +0.2897 [-0.1786, +0.9267] | +0.3247 [-0.1746, +0.9987] |
| vidstg/search | negative_local | +0.2897 [-0.1786, +0.9267] | +0.3248 [-0.1744, +0.9987] |
| vidstg/confirm | rank_cross | +0.3584 [-0.1504, +0.8577] | +0.3724 [-0.1683, +0.9179] |
| vidstg/confirm | negative_global | +0.4812 [-0.5078, +1.4108] | +0.4921 [-0.5880, +1.5337] |
| vidstg/confirm | negative_local | +0.4810 [-0.5077, +1.4103] | +0.4918 [-0.5878, +1.5330] |
| hc2/search | rank_cross | +0.1229 [-0.2231, +0.4284] | +0.0901 [-0.2716, +0.4012] |
| hc2/search | negative_global | -0.5235 [-1.1794, +0.0364] | -0.4636 [-1.1053, +0.1156] |
| hc2/search | negative_local | -0.5257 [-1.1877, +0.0367] | -0.4648 [-1.1088, +0.1160] |
| hc2/confirm | rank_cross | -0.4837 [-1.1041, +0.0203] | -0.5362 [-1.1115, -0.0611] |
| hc2/confirm | negative_global | +0.1283 [+0.0021, +0.2632] | +0.1660 [+0.0270, +0.3127] |
| hc2/confirm | negative_local | +0.1310 [+0.0022, +0.2678] | +0.1685 [+0.0286, +0.3168] |

## reset-u：另一个完整在线删除对照

每query进入前只把256维query residual归零，三组LN继承；每condition/order/split回source状态。原本数据集Rank lr/T/K与Fast读出保持，当前输出先于空间write封存。这个1152到达流只回答query residual生命周期删除的效果，没有捆绑负证据loss。true输出与固定旧A interval空间对照分开。

| 面板 | true reset-u−原A ΔvIoU | fixed-A interval ΔvIoU | >5pp / >20pp true损害 |
|---|---:|---:|---:|
| vidstg/search | -0.6061 [-2.2580, +0.4215] | -0.6061 [-2.2580, +0.4215] | 15/6 |
| vidstg/confirm | +0.0426 [-0.0299, +0.1580] | +0.0426 [-0.0299, +0.1580] | 0/0 |
| hc2/search | -0.8237 [-4.0170, +1.2936] | -0.8237 [-4.0170, +1.2936] | 10/10 |
| hc2/confirm | +0.0772 [-0.0127, +0.2290] | +0.0772 [-0.0127, +0.2290] | 0/0 |

## clean、顺序、来源集中与反例

SUMMARY保留clean/all的CI和尾部，以及两个order各自CI与gross均值；reset另分expert/nonexpert。SOURCE_CONCENTRATION提供逐源值、最大正源份额、前两负源份额、leave-one-source-out范围。CASES在每面板每对照保留最大正例和负例；没有删源追分、事后改lambda、挑GT步数或另加幅度对照。

| 确认clean | global−Rank vIoU | local−Rank vIoU | future local−Rank vIoU | reset-u−A vIoU |
|---|---:|---:|---:|---:|
| vidstg | -1.4147 [-2.7019, -0.1900] | -1.4158 [-2.7039, -0.1905] | +0.5553 [-0.4709, +1.4931] | +0.0329 [-0.1127, +0.2100] |
| hc2 | -0.5608 [-2.1977, +0.7418] | -0.5536 [-2.1890, +0.7619] | +0.2266 [+0.0185, +0.4957] | +0.0586 [-0.0141, +0.1622] |

## 与旧N1和原论文的边界

旧N1在16个历史曝光Vid源上用GT事先筛Useful/Noisy正负信号，目标为方向pairwise softplus，且主正负臂更新次数与梯度量不同；其负结果不能直接否定这里GT-free有限负teacher/同帧loss。这里重开的是不同机制和same-prestate匹配问题，未抹去N1负结果。

[NLNL原论文](https://arxiv.org/pdf/1908.07387)的negative learning用互补类别−log(1−p_c)，并配有选择NL/PL阶段；互补类别仍可能碰到真实类。[U2PL原论文](https://arxiv.org/pdf/2203.03884)把不可靠像素用于类别相对的对比负样本队列，同时保留监督与可靠正伪标签，并非低置信度即错误。[GKD原论文](https://arxiv.org/pdf/2306.13649)在学生生成prefix上用完整teacher token分布蒸馏，不穿过采样反传；这里九个几何probe分布不是原生自回归tube policy，不能写成对GKD完整复现。

已存DTA合法i<j的factorized start/end logit家族里，forward KL joint teacher的梯度只依赖其start/end边缘；相同边缘但不同联合相关性会给相同梯度。这个投影限制不是所有joint/mixture机制无用的证明，本轮也没有运行或晋升DTA。

## 核验与资源

全局seal先于GT评分；独立根审计46,118,535个数值/链/哈希项，公开审计256,847项。ROOT_AUDIT单独声明：未加载模型，因此不能独立复现完整参数Jacobian或重新执行block-only输出；可独立复算输出空间导数、SGD坐标、状态继承、目标、IoU证据、密集指标和误罚统计。

RESOURCES分别保留两集local/reset-u的suffix/backward/block/future重放、wall/VRAM；缓存backbone和专家的新调用为0。CPU评分/审计/报告时间独立，不把此前专家成本、开发时间或缓存生成含入本轮worker时间。

CONFIRMATION_POSITIVE_LOWER_BOUND只记录两集corrupt确认主CI是否均>0，不是生产晋升门。历史曝光、小专家独立源、有效步长变化、严重尾部以及单次迁移边界仍限制结论；CURRENT_METHOD未改变。

公开结果仅匿名标量、teacher概率/证据标量、聚合、CI、正负例和图；raw expert boxes、学生框轨迹、H/features、参数状态、caption、媒体、标注、权重不公开。
