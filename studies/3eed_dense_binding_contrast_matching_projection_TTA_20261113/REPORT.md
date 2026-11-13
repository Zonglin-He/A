# R94 — 用真实对应的反事实差分修复密集证据与匹配投影TTA

北京时间 2026-11-13 15:53。全部DEV336为历史暴露开发数据，两个Drone/三个Quad场景；没有独立确认，不访问FINAL。

预先固定主方法 binding_bt1=22.984912%。相对R91零步-0.047133pp，相对同信息直接目标-0.175686pp，相对同一优化目标的精确零更新最优混合-0.064460pp。固定开发增量检查通过：False。

## 保持与实质改变

R91_R76_full_disc架构、Waymo-only源准备权重、原A256框与RGB/LiDAR/全文不变。没有新增视觉专家、源训练、meta训练、回归或新目标输入。R93最多4候选与同源同向接收，使127条可由现有框挽救的Acc25失败没有得到一条直接纠错偏好；R94把新增对应证据扩展至全部256候选，并允许反对原第一名。此变化不是放宽旧阈值或把局部偏好升级为唯一目标标签。

保留R91模型前向双向结构，但本轮TTA仅更新已有匹配MLP最后64×288线性权重与64偏置，共18496参数。原102498低层融合theta、本轮所有其他matching层、lateRGB投影、原3EED、源判别接口、RGB编码器均冻结。停止物理恢复主loss，直接检验新的证据是否能指导匹配。旧物理配方保留为R93参考，不宣称已修好三维实例载体或语义角色绑定。

## 密集对应证据：不是身份真值

在完整观测和原互补遮挡76001/2，重用同权重R91 factual特征，并对两种固定错误绑定实际重新运行双向融合和原生语言/decoder。对可见点作确定性循环错位，p2i行及i2p逆映射同时变化，保持每点可见资格、同张RGB、XYZ、深度目标、A256与query空间锚点。它破坏对应而不删除候选、改描述或创造新视角；并非真实可能的另一标定，也不保证是语义上无关的负观测。

delta_i=mean_v b_real(v,i)-mean_{c,v}b_wrong(c,v,i)，仅减去候选均值去掉softmax无效共同偏移。q=softmax(b_full+r0+delta)，强度1固定，不按source概率裁掉候选、不要求偏好与source或残差同向。r0为当前输入原R91残差，停止梯度。对比信号也可能反映错误绑定导致的分布外响应，不等于目标身份证据；保留按query打乱delta、反转delta及不用差分的正确视图均值。未给不可见候选伪造实例证据，native上下文仍可影响全部候选；没有将投影支持当完整实例mask。

## 更新、接收与强零更新对照

每输入恢复原source。目标q和p0都停止梯度，L=KL(q||pW)+KL(p0||pW)。只有候选分布，不把q的Top1作为唯一GT。此概率空间目标的精确最优解为(p0+q)/2；因此同时保留q直接读出和该精确混合，不选择较弱零更新对照。特征缓存仅因可更新参数全在末端投影之后才成立；独立原生后缀重放验证所有trial。

Adam lr0.0005/clip1，固定主1步、辅3步。每步保留原9档半尺度，c1e-4/tol1e-10，以clip后的提案梯度和实际FP32位移计算下降界；不是grounding正确性认证。先保存完整Adam提案，再回溯；没有接受则同时恢复参数与optimizer状态。接受缩放提案时保留该次Adam梯度矩。始终同一初始监督、不使用GT挑步长或输出。3步/raw/控制不会根据开发分数替换锁定主方法。

## 全DEV336任务结果

| 方法 | Drone mIoU % | Quad mIoU % | 等权mIoU % | 等权Acc25 % | 等权Acc50 % |
|---|---:|---:|---:|---:|---:|
| source | 15.539892 | 28.120770 | 21.830331 | 40.773810 | 21.130952 |
| ABC | 15.736332 | 28.868588 | 22.302460 | 42.261905 | 21.428571 |
| R59 | 16.385568 | 28.836664 | 22.611116 | 43.154762 | 21.726190 |
| R91_frozen | 17.346189 | 28.717901 | 23.032045 | 41.964286 | 21.726190 |
| R91_mean3 | 17.145428 | 28.802929 | 22.974178 | 41.964286 | 21.726190 |
| R93_bt | 17.159730 | 28.670921 | 22.915325 | 41.964286 | 21.726190 |
| R93_raw | 16.941057 | 29.466507 | 23.203782 | 42.559524 | 21.428571 |
| binding_direct | 17.327045 | 28.994151 | 23.160598 | 42.857143 | 21.428571 |
| binding_optimal_mixture | 17.394220 | 28.704524 | 23.049372 | 42.559524 | 21.428571 |
| reverse_direct | 16.326920 | 28.689693 | 22.508307 | 40.178571 | 21.726190 |
| shuffle_direct | 16.728711 | 28.644783 | 22.686747 | 41.666667 | 21.428571 |
| viewmean_direct | 17.162099 | 28.802929 | 22.982514 | 42.261905 | 21.726190 |
| viewmean_optimal_mixture | 17.145428 | 28.774588 | 22.960008 | 41.964286 | 21.726190 |
| binding_raw1 | 17.302161 | 28.609339 | 22.955750 | 42.261905 | 21.726190 |
| binding_bt1 | 17.251923 | 28.717901 | 22.984912 | 42.261905 | 21.726190 |
| binding_bt3 | 17.196556 | 28.609339 | 22.902947 | 42.261905 | 21.726190 |
| shuffle_bt1 | 17.329698 | 28.709369 | 23.019534 | 41.964286 | 21.726190 |
| reverse_bt1 | 17.346189 | 28.741204 | 23.043696 | 42.261905 | 21.726190 |
| viewmean_bt1 | 17.249148 | 28.741204 | 22.995176 | 42.261905 | 21.726190 |

## 组件与全部配对负结果

| 类型 | 方法 | 对照 | 平台 | Δpp | 改善/恶化/不变 | 换query | Acc25救回/破坏 |
|---|---|---|---|---:|---:|---:|---:|
| primary | binding_bt1 | R91_frozen | balanced | -0.047133 | 2/3/331 | 10 | 2/1 |
| primary | binding_bt1 | R91_frozen | drone | -0.094266 | 2/3/163 | 9 | 2/1 |
| primary | binding_bt1 | R91_frozen | quad | +0.000000 | 0/0/168 | 1 | 0/0 |
| primary | binding_bt1 | R91_mean3 | balanced | +0.010734 | 3/5/328 | 22 | 2/1 |
| primary | binding_bt1 | R91_mean3 | drone | +0.106495 | 2/2/164 | 16 | 2/0 |
| primary | binding_bt1 | R91_mean3 | quad | -0.085027 | 1/3/164 | 6 | 0/1 |
| primary | binding_bt1 | R59 | balanced | +0.373797 | 44/55/237 | 180 | 14/17 |
| primary | binding_bt1 | R59 | drone | +0.866355 | 25/35/108 | 112 | 9/10 |
| primary | binding_bt1 | R59 | quad | -0.118762 | 19/20/129 | 68 | 5/7 |
| primary | binding_bt1 | R93_bt | balanced | +0.069587 | 4/4/328 | 17 | 2/1 |
| primary | binding_bt1 | R93_bt | drone | +0.092193 | 3/3/162 | 14 | 2/0 |
| primary | binding_bt1 | R93_bt | quad | +0.046980 | 1/1/166 | 3 | 0/1 |
| primary | binding_bt1 | binding_direct | balanced | -0.175686 | 12/13/311 | 57 | 1/3 |
| primary | binding_bt1 | binding_direct | drone | -0.075122 | 8/9/151 | 39 | 0/2 |
| primary | binding_bt1 | binding_direct | quad | -0.276249 | 4/4/160 | 18 | 1/1 |
| primary | binding_bt1 | binding_optimal_mixture | balanced | -0.064460 | 7/7/322 | 32 | 0/1 |
| primary | binding_bt1 | binding_optimal_mixture | drone | -0.142297 | 5/4/159 | 23 | 0/1 |
| primary | binding_bt1 | binding_optimal_mixture | quad | +0.013377 | 2/3/163 | 9 | 0/0 |
| primary | binding_bt1 | shuffle_bt1 | balanced | -0.034621 | 3/3/330 | 13 | 2/1 |
| primary | binding_bt1 | shuffle_bt1 | drone | -0.077775 | 2/3/163 | 11 | 2/1 |
| primary | binding_bt1 | shuffle_bt1 | quad | +0.008532 | 1/0/167 | 2 | 0/0 |
| primary | binding_bt1 | reverse_bt1 | balanced | -0.058784 | 2/4/330 | 18 | 2/2 |
| primary | binding_bt1 | reverse_bt1 | drone | -0.094266 | 2/3/163 | 14 | 2/1 |
| primary | binding_bt1 | reverse_bt1 | quad | -0.023302 | 0/1/167 | 4 | 0/1 |
| primary | binding_bt1 | viewmean_bt1 | balanced | -0.010264 | 1/2/333 | 11 | 1/1 |
| primary | binding_bt1 | viewmean_bt1 | drone | +0.002775 | 1/1/166 | 9 | 1/0 |
| primary | binding_bt1 | viewmean_bt1 | quad | -0.023302 | 0/1/167 | 2 | 0/1 |
| primary | binding_bt1 | binding_raw1 | balanced | +0.029163 | 4/2/330 | 13 | 0/0 |
| primary | binding_bt1 | binding_raw1 | drone | -0.050238 | 2/2/164 | 10 | 0/0 |
| primary | binding_bt1 | binding_raw1 | quad | +0.108563 | 2/0/166 | 3 | 0/0 |
| primary | binding_bt1 | binding_bt3 | balanced | +0.081965 | 5/1/330 | 14 | 0/0 |
| primary | binding_bt1 | binding_bt3 | drone | +0.055368 | 3/1/164 | 11 | 0/0 |
| primary | binding_bt1 | binding_bt3 | quad | +0.108563 | 2/0/166 | 3 | 0/0 |
| evidence | binding_direct | R91_frozen | balanced | +0.128552 | 14/14/308 | 60 | 4/1 |
| evidence | binding_direct | R91_frozen | drone | -0.019144 | 10/10/148 | 42 | 3/0 |
| evidence | binding_direct | R91_frozen | quad | +0.276249 | 4/4/160 | 18 | 1/1 |
| evidence | binding_direct | shuffle_direct | balanced | +0.473851 | 25/22/289 | 104 | 7/3 |
| evidence | binding_direct | shuffle_direct | drone | +0.598334 | 18/11/139 | 74 | 4/1 |
| evidence | binding_direct | shuffle_direct | quad | +0.349368 | 7/11/150 | 30 | 3/2 |
| evidence | binding_direct | reverse_direct | balanced | +0.652291 | 30/24/282 | 114 | 10/1 |
| evidence | binding_direct | reverse_direct | drone | +1.000125 | 23/16/129 | 81 | 9/0 |
| evidence | binding_direct | reverse_direct | quad | +0.304458 | 7/8/153 | 33 | 1/1 |
| evidence | binding_direct | viewmean_direct | balanced | +0.178084 | 15/17/304 | 63 | 4/2 |
| evidence | binding_direct | viewmean_direct | drone | +0.164946 | 10/10/148 | 44 | 3/0 |
| evidence | binding_direct | viewmean_direct | quad | +0.191222 | 5/7/156 | 19 | 1/2 |
| exact_zero_update_optimum | binding_optimal_mixture | R91_frozen | balanced | +0.017327 | 8/9/319 | 36 | 2/0 |
| exact_zero_update_optimum | binding_optimal_mixture | R91_frozen | drone | +0.048031 | 5/7/156 | 26 | 2/0 |
| exact_zero_update_optimum | binding_optimal_mixture | R91_frozen | quad | -0.013377 | 3/2/163 | 10 | 0/0 |

## 纠错覆盖：按实际任务用途统计

GT仅在封存、原生审计后作以下定位效用诊断，不用于构造delta、优化、接收或筛样本。IoU合格不等于独立认证物体身份。稠密256覆盖也不等于已经测量出可靠目标语义。

| 平台 | n | 可恢复Acc25失败 | q中有合格候选超过错误第一名 | 直接换Top1 | 直接Acc25救回/破坏 | 两错配delta相关均值 |
|---|---:|---:|---:|---:|---:|---:|
| pooled | 336 | 127 | 4 | 60 | 4/1 | 0.54358225 |
| drone | 168 | 72 | 3 | 42 | 3/0 | 0.61385743 |
| quad | 168 | 55 | 1 | 18 | 1/1 | 0.47330707 |

## 实际拟合与回溯

| 目标/步 | n | 接受数 | 平均scale | 初始/端点loss | 平均参数L2 |
|---|---:|---:|---:|---:|---:|
| binding/1 | 336 | 336 | 0.40271577 | 0.0069838782/0.0062406535 | 0.02342856 |
| binding/2 | 336 | 302 | 0.42840867 | 0.0062406535/0.0059993429 | 0.03420847 |
| binding/3 | 336 | 289 | 0.52669271 | 0.0059993429/0.0057646666 | 0.04470769 |
| shuffle/1 | 336 | 336 | 0.32114955 | 0.0093476878/0.0089609051 | 0.01863108 |
| reverse/1 | 336 | 336 | 0.39095052 | 0.0068506711/0.0061292471 | 0.02273762 |
| viewmean/1 | 336 | 336 | 0.20214844 | 0.0016583583/0.0012951433 | 0.01174714 |

## 五个历史场景

| 平台/场景 | n | 方法 | mIoU % |
|---|---:|---|---:|
| drone/D_S1 | 106 | R59 | 11.761612 |
| drone/D_S1 | 106 | R91_frozen | 13.989152 |
| drone/D_S1 | 106 | R93_bt | 13.693631 |
| drone/D_S1 | 106 | binding_direct | 14.120446 |
| drone/D_S1 | 106 | binding_optimal_mixture | 14.210958 |
| drone/D_S1 | 106 | binding_bt1 | 13.840276 |
| drone/D_S1 | 106 | shuffle_bt1 | 13.989963 |
| drone/D_S1 | 106 | reverse_bt1 | 13.989152 |
| drone/D_S2 | 62 | R59 | 24.291041 |
| drone/D_S2 | 62 | R91_frozen | 23.085641 |
| drone/D_S2 | 62 | R93_bt | 23.085641 |
| drone/D_S2 | 62 | binding_direct | 22.809294 |
| drone/D_S2 | 62 | binding_optimal_mixture | 22.836572 |
| drone/D_S2 | 62 | binding_bt1 | 23.084740 |
| drone/D_S2 | 62 | shuffle_bt1 | 23.039568 |
| drone/D_S2 | 62 | reverse_bt1 | 23.085641 |
| quad/Q_S1 | 62 | R59 | 9.632116 |
| quad/Q_S1 | 62 | R91_frozen | 10.708624 |
| quad/Q_S1 | 62 | R93_bt | 10.771765 |
| quad/Q_S1 | 62 | binding_direct | 10.693794 |
| quad/Q_S1 | 62 | binding_optimal_mixture | 10.776104 |
| quad/Q_S1 | 62 | binding_bt1 | 10.708624 |
| quad/Q_S1 | 62 | shuffle_bt1 | 10.708624 |
| quad/Q_S1 | 62 | reverse_bt1 | 10.771765 |
| quad/Q_S2 | 49 | R59 | 36.067581 |
| quad/Q_S2 | 49 | R91_frozen | 35.770023 |
| quad/Q_S2 | 49 | R93_bt | 35.529054 |
| quad/Q_S2 | 49 | binding_direct | 35.638776 |
| quad/Q_S2 | 49 | binding_optimal_mixture | 35.638776 |
| quad/Q_S2 | 49 | binding_bt1 | 35.770023 |
| quad/Q_S2 | 49 | shuffle_bt1 | 35.740770 |
| quad/Q_S2 | 49 | reverse_bt1 | 35.770023 |
| quad/Q_S3 | 57 | R59 | 43.509770 |
| quad/Q_S3 | 57 | R91_frozen | 42.244590 |
| quad/Q_S3 | 57 | R93_bt | 42.244590 |
| quad/Q_S3 | 57 | binding_direct | 43.187756 |
| quad/Q_S3 | 57 | binding_optimal_mixture | 42.244590 |
| quad/Q_S3 | 57 | binding_bt1 | 42.244590 |
| quad/Q_S3 | 57 | shuffle_bt1 | 42.244590 |
| quad/Q_S3 | 57 | reverse_bt1 | 42.244590 |

## 固定回访与总体正负尾部

原R93四个固定回访再加主方法/精确混合/直接证据相对零步的总体最大正负贡献，各至多2例。保持全体分母，不能按案例改规则。

| 案例 | 平台 | 原因 | 零步 | q直接 | 精确混合 | TTA1 | TTA3 | shuffle1 |
|---|---|---|---:|---:|---:|---:|---:|
| CASE001 | drone | fixed_R93_CASE001;harm_binding_bt1_vs_R91_frozen;harm_binding_bt1_vs_binding_optimal_mixture | 0.452964 | 0.452964 | 0.452964 | 0.000000 | 0.000000 | 0.452964 |
| CASE002 | quad | fixed_R93_CASE002;gain_binding_direct_vs_R91_frozen | 0.000000 | 0.537604 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| CASE003 | quad | fixed_R93_CASE003 | 0.233223 | 0.233223 | 0.233223 | 0.233223 | 0.233223 | 0.233223 |
| CASE004 | quad | fixed_R93_CASE004 | 0.446062 | 0.446062 | 0.446062 | 0.446062 | 0.327987 | 0.446062 |
| CASE005 | drone | gain_binding_bt1_vs_R91_frozen;gain_binding_direct_vs_R91_frozen | 0.000000 | 0.318500 | 0.318500 | 0.318500 | 0.318500 | 0.000000 |
| CASE006 | drone | gain_binding_bt1_vs_R91_frozen | 0.246800 | 0.274807 | 0.274807 | 0.274807 | 0.274807 | 0.246800 |
| CASE007 | drone | harm_binding_bt1_vs_R91_frozen | 0.278824 | 0.250259 | 0.250259 | 0.250259 | 0.250259 | 0.250259 |
| CASE008 | drone | gain_binding_bt1_vs_binding_optimal_mixture;harm_binding_direct_vs_R91_frozen | 0.511573 | 0.372176 | 0.372176 | 0.511573 | 0.511573 | 0.511573 |
| CASE009 | quad | gain_binding_bt1_vs_binding_optimal_mixture | 0.666919 | 0.602609 | 0.602609 | 0.666919 | 0.602609 | 0.666919 |
| CASE010 | drone | harm_binding_bt1_vs_binding_optimal_mixture | 0.284073 | 0.359581 | 0.359581 | 0.284073 | 0.284073 | 0.284073 |
| CASE011 | drone | harm_binding_direct_vs_R91_frozen | 0.227512 | 0.000000 | 0.227512 | 0.227512 | 0.227512 | 0.227512 |

## 数值审计、成本、范围

独立新生成factual/counterfactual观测，重建全部目标，并独立Adam回滚及原生后缀重放4032端点、6237trial。最大证据/概率误差0，参数0，loss2.40259201423e-16，界0，Top1不一致0。容差score/loss1e-6、参数1e-7、界1e-10，Top1必须一致。

生产62.652s，新增counterfactual原生前向2016，匹配头trial6237，峰值GPU694139904bytes。初始factual视图与点前缀复用历史同权重缓存，独立审计重新生成，因此不是冷启动部署时延。native_zero_parity_calls计数为0：该生产计数未包含新原生zero前向，初始parity由缓存与独立native审计建立，不将0写成未检查。

全DEV336长期暴露、仅五个场景，不能作为全新确认或构造独立场景CI。source训练、FINAL、额外专家、框坐标更新均无。全部UNKNOWN/回退/拒绝trial/控制和负比较保存，支持记录不能当实例身份真值。

## 原论文与实现条件

[VCD论文方法](https://arxiv.org/html/2311.16922v1)及[官方实现](https://raw.githubusercontent.com/DAMO-NLP-SG/VCD/master/vcd_utils/vcd_sample.py)使用原视觉与扰动视觉的logit对比并带原预测plausibility cutoff。R94只借鉴对比证据，改变任务、干预和更新，不称复现；原代码alpha默认.5，R94固定1不是声称原默认。R94特意不使用source cutoff，避免再次排除低排名可恢复候选。

[NeurIPS2025作者项目的反例分析](https://github.com/ustc-hyin/cd_rethink)提醒对比分数收益可能源于输出偏置或筛选。这里以打乱/反向证据、强无更新最优混合及完整任务结果检验，不能把敏感性解释为因果身份认证。R66已做预融合区域消融，本轮改变的是校准对应、稠密候选监督和末端投影更新，不重复类别增量硬门或区域伪正标签。

## 本轮裁决

预设开发检查=False，独立确认=False，稳健采纳=False。结果仅支持或削弱该固定组合，不能外推到所有对应对比/所有TTA。保留R91架构和原零步参照；不按最高开发分数将raw/3步或错配控制提升为主方法，不追加系数/位置/阈值网格。

## 修复后的证据与执行必须分开裁决

密集对应q直接读出 **23.160598%**，相对零步+0.128552pp，14改善/14恶化；Acc25救回4/破坏1。比打乱delta直接+0.473851pp，比反向delta直接+0.652291pp。正常对应修正与候选匹配有关，值得保留开发信号；不称已可靠解决全部身份绑定。

限制：Drone直接mIoU略降、Quad升；Acc50略降；最大单例正贡献+0.160001pp，大于净增益。当前R93_raw参考23.203782%也高于新直接读出。因此不把此结果包装成新的总体最优/稳定跨域TTA。

最关键固定对照：新q保住旧近处车辆并改正旧行人，说明R93的两个难例不再都卡在反馈构造；初版匹配投影更新却仍破坏前者、漏掉后者。原始KL最优混合在行人例本就选错，因此源输出保护会阻碍部分正确证据；但即使不按混合拟合，后续数值修复也未解决全部执行。

## 唯一一次公开披露的后开发求解器修复

以下在初版DEV成绩已暴露后锁定，属于同一开发集上的修复诊断，不是独立重复。原主方法binding_bt1与负结论保留，不以secondary替换。详见protocol/SOLVER_REPAIR_PROTOCOL.md。新证据、图像、候选、权重初值与层位不变；以中心化候选logit平方误差为目标，用阻尼Gauss–Newton一次求解同一末端层的局部变化。lambda=.01trace(JcJcT)/256，最大参数L2为source层范数5%，九档回溯，没有GT选步/网格。普通梯度控制具有同目标、同GN提案L2；混合目标GN隔离源保护，打乱目标GN检验证据对应。

| 方法 | Drone mIoU % | Quad mIoU % | 等权mIoU % | Acc25 % | Acc50 % |
|---|---:|---:|---:|---:|---:|
| R91_frozen | 17.346189 | 28.717901 | 23.032045 | 41.964286 | 21.726190 |
| R59 | 16.385568 | 28.836664 | 22.611116 | 43.154762 | 21.726190 |
| binding_direct | 17.327045 | 28.994151 | 23.160598 | 42.857143 | 21.428571 |
| binding_optimal_mixture | 17.394220 | 28.704524 | 23.049372 | 42.559524 | 21.428571 |
| binding_bt1 | 17.251923 | 28.717901 | 22.984912 | 42.261905 | 21.726190 |
| gn_raw | 17.246702 | 28.717904 | 22.982303 | 42.261905 | 21.726190 |
| gn_bt | 17.246190 | 28.717904 | 22.982047 | 42.261905 | 21.726190 |
| gradient_raw | 16.351556 | 28.641382 | 22.496469 | 41.071429 | 21.130952 |
| gradient_bt | 17.147474 | 28.741204 | 22.944339 | 41.964286 | 21.726190 |
| gn_mixture_raw | 17.246702 | 28.717904 | 22.982303 | 42.261905 | 21.726190 |
| gn_mixture_bt | 17.246190 | 28.717904 | 22.982047 | 42.261905 | 21.726190 |
| gn_shuffle_raw | 17.368335 | 28.717904 | 23.043120 | 41.964286 | 21.726190 |
| gn_shuffle_bt | 17.368335 | 28.717904 | 23.043120 | 41.964286 | 21.726190 |

| 方法 | 对照 | 平台 | Δpp | 改善/恶化/不变 | 换query |
|---|---|---|---:|---:|---:|
| gn_bt | R91_frozen | balanced | -0.049998 | 3/3/330 | 15 |
| gn_bt | R91_frozen | drone | -0.099999 | 2/3/163 | 11 |
| gn_bt | R91_frozen | quad | +0.000003 | 1/0/167 | 4 |
| gn_bt | binding_direct | balanced | -0.178551 | 12/13/311 | 51 |
| gn_bt | binding_direct | drone | -0.080855 | 8/9/151 | 35 |
| gn_bt | binding_direct | quad | -0.276247 | 4/4/160 | 16 |
| gn_bt | binding_optimal_mixture | balanced | -0.067325 | 8/7/321 | 27 |
| gn_bt | binding_optimal_mixture | drone | -0.148030 | 5/4/159 | 19 |
| gn_bt | binding_optimal_mixture | quad | +0.013380 | 3/3/162 | 8 |
| gn_bt | binding_bt1 | balanced | -0.002865 | 2/1/333 | 15 |
| gn_bt | binding_bt1 | drone | -0.005733 | 1/1/166 | 12 |
| gn_bt | binding_bt1 | quad | +0.000003 | 1/0/167 | 3 |
| gn_bt | gradient_bt | balanced | +0.037709 | 3/4/329 | 18 |
| gn_bt | gradient_bt | drone | +0.098717 | 2/3/163 | 13 |
| gn_bt | gradient_bt | quad | -0.023299 | 1/1/166 | 5 |
| gn_bt | gn_mixture_bt | balanced | +0.000000 | 0/0/336 | 1 |
| gn_bt | gn_mixture_bt | drone | +0.000000 | 0/0/168 | 1 |
| gn_bt | gn_mixture_bt | quad | +0.000000 | 0/0/168 | 0 |
| gn_bt | gn_shuffle_bt | balanced | -0.061073 | 3/5/328 | 20 |
| gn_bt | gn_shuffle_bt | drone | -0.122145 | 3/5/160 | 18 |
| gn_bt | gn_shuffle_bt | quad | +0.000000 | 0/0/168 | 2 |
| gn_bt | gn_raw | balanced | -0.000256 | 0/1/335 | 2 |
| gn_bt | gn_raw | drone | -0.000512 | 0/1/167 | 2 |
| gn_bt | gn_raw | quad | +0.000000 | 0/0/168 | 0 |

| 更新 | 60条q要求换第一名中实际执行数 | 全336与q一致数 |
|---|---:|---:|
| binding_bt1 | 6 | 279 |
| gn_bt | 11 | 285 |
| gradient_bt | 1 | 273 |
| gn_mixture_bt | 10 | 284 |
| gn_shuffle_bt | 4 | 271 |

GN提案336/336触及固定5%幅度上限；平均中心化分数误差只完成15.901548%修正。元数据unconstrained_L2字段实际存放限幅后的提案L2，按其与source_relative_radius_limit相等识别触限，不把它当真实未限幅范数。当前信赖域/阻尼限制仍是欠拟合的竞争解释；这份单步配置不能证明末层表达能力不可能。没有追加放大幅度或减阻尼网格。

独立构造显式256×18496Jacobian验证因子Gram方法，并逐trial完整原生重放2946次；参数最大差7.45058059692e-09、概率0、loss0、Top1不一致0。没有新增证据调用或GT优化。

| 固定/总体案例 | 原图观察与二次求解端点 |
|---|---|
| CASE001 | 新q和精确混合均保留近处车辆；初版Adam和修复GN仍转向远处区域。证据未要求该误改，说明共享末层更新未忠实执行这份正确保护。 |
| CASE002 | 新q直接读出改到左侧主体，0到0.5376；混合仍选另一行人，Adam与GN也没有改正。源分布保护会阻碍这份有益变化，不能再只归因于G没信号。 |
| CASE003 | 同一SUV框保持，旧物理更新中的重复框改善本轮未复现。普通梯度次对照可换到较高IoU重复框，不等同实例身份监督。 |
| CASE004 | 直接证据和一步更新保留同一左侧行人框；三步Adam才改到较差重复框。更充分拟合当前目标没有带来更好的定位。 |
| CASE005 | 从远处区域回到左侧边缘车辆附近，q/Adam/GN均0到0.3185，shuffleAdam未实现。目标边缘裁切，精确身份细节仍部分未知；全体总分中正常计入。 |
| CASE006 | 同一SUV附近由0.2468换到0.2748，q/Adam/GN均执行并跨Acc25；不是找回另一物体。 |
| CASE007 | 同一SUV附近改到较差重复框0.2788到0.2503，q与更新共同继承此负例，仍计总体。 |
| CASE008 | q与混合转到同一白色车辆的较差重复框0.5116到0.3722；Adam/GN保留原框。更新相对直接读出变好只是避免伪目标损伤，没有超过零步。 |
| CASE009 | 画面右边缘同一人物的重复框，q/混合略差；Adam/GN一步保持原框，三步Adam继承较差框。不能把保持原框称新实例纠错。 |
| CASE010 | 同一白色车辆已有更好重复框，q/混合从0.2841到0.3596，Adam/GN都未执行这份提升。 |
| CASE011 | q从SUV附近转到远处上沿区域并降到0；混合、Adam和GN保留原低IoU框。该例证明新证据仍可误导，不因总体净收益而删除。 |

最终保留：R91双向架构，密集对应差分作为有有限开发价值的新证据组件；全部负例和旧参照不变。未修好：参数更新尚未超过自身零步、q直接或同信息最优混合，GN也未完成关键两例的同时修复，不采纳这两份TTA配方。角色/实例支持并未获得独立身份认证，127条可恢复Acc25失败仅4条在新目标中获得明确合格候选优势，覆盖仍低；固定A256的几何上限也未改变。阶段A/B共用同336观测，不重复计算独立样本数。终止本轮有限修复，不启动新训练/专家/层位搜索。

二次修复不只是把Adam换成矩阵求解：同时由原KL＋source保持的混合目标，改为直接teacher中心化分数拟合。GN与同目标同幅度普通梯度的比较隔离数值求解；GN_mixture用于区分源保护/目标变化。不能将它与原Adam主方法的差分全部归因于Gauss–Newton。
