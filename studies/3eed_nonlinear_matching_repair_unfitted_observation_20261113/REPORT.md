# R95 — 非线性匹配更新修复与未拟合遮挡迁移

## 结果边界

锁定主方法 lm_FC=22.769706%，相对R91零步-0.262340pp，相对同信息direct_FC-0.006161pp，相对四观测direct_four+0.032823pp。预设开发增量检查=False；历史DEV结果不构成独立确认。F或C单独结果不会根据GT升级为主结果。

保留R91 R76_full_disc原架构、源参数、原A256框；仅修改既有matching MLP最后64×288线性投影和64偏置，18496参数。每输入从source重置，不重新训练源域，不加载新专家，不访问FINAL。全体336条、两个平台各168条、五个历史场景、所有对照与负例均保留。

## 修复内容与未拟合观测

R94二阶求解全部撞上人为5%源参数范数上限，平均只执行约16%的logit目标修正。R95保持同一封存R94 dense binding contrast分布q，仅改执行：对归一化匹配层的centered log-probability误差重新线性化，LM最多10轮、每轮最多6个damping试探，实际下降/线性预测下降>0.1才接受；>0.75时damping/3，<0.25时×2，拒绝×10。固定数值终止为原误差1e-3或1e-12，不读取GT。主方法无固定参数百分比上限。lm_capped控制每次提案仍限source范数5%，这是每步上限，不是总位移上限。

目标来自R94完整观测和遮挡76001/2的真实绑定相对两种固定错配绑定的差分，保持强度1。它是未认证正确性的内部观测证据；不是目标身份真值。本轮不重新生成或用任务标签筛选q。lm_mixture改为(p0+q)/2目标，lm_shuffle保持优化预算但打乱目标候选对应。

每输入在全部四种head拟合结束之后，才生成76003遮挡C。它不进入更新、接收、damping或停止规则。随后将同一真实权重改动用于F及C，锁定主读出FC是两者概率等权平均。C是同一物理输入的另一遮挡，不是新相机视角，也不是未见场景。C仍使用静态当前观测残差r0，不能宣称它是完全没有当前观测信息的独立推理。

direct_C将相同delta直接加到冻结C分数；direct_FC等权合并qF和qC；direct_four使用原三观测及C，四者都使用相同delta。它们不更新任何权重。mixture_FC和shuffle_FC分别与混合目标和错配目标匹配，拒绝以较弱无更新对照制造TTA收益。

## 全DEV336结果

| 方法 | Drone mIoU % | Quad mIoU % | 等权 mIoU % | Acc25 % | Acc50 % |
|---|---:|---:|---:|---:|---:|
| source | 15.539892 | 28.120770 | 21.830331 | 40.773810 | 21.130952 |
| ABC | 15.736332 | 28.868588 | 22.302460 | 42.261905 | 21.428571 |
| R59 | 16.385568 | 28.836664 | 22.611116 | 43.154762 | 21.726190 |
| R91_frozen | 17.346189 | 28.717901 | 23.032045 | 41.964286 | 21.726190 |
| R94_direct | 17.327045 | 28.994151 | 23.160598 | 42.857143 | 21.428571 |
| R94_mixture | 17.394220 | 28.704524 | 23.049372 | 42.559524 | 21.428571 |
| R94_Adam | 17.251923 | 28.717901 | 22.984912 | 42.261905 | 21.726190 |
| R94_GN | 17.246190 | 28.717904 | 22.982047 | 42.261905 | 21.726190 |
| frozen_C | 17.096483 | 29.137319 | 23.116901 | 42.559524 | 22.023810 |
| direct_C | 16.543141 | 28.732288 | 22.637715 | 42.261905 | 21.130952 |
| direct_FC | 16.673385 | 28.878349 | 22.775867 | 42.559524 | 21.130952 |
| direct_four | 16.433171 | 29.040595 | 22.736883 | 41.964286 | 21.130952 |
| mixture_FC | 17.054572 | 29.082627 | 23.068600 | 42.559524 | 21.726190 |
| shuffle_FC | 16.976914 | 28.661255 | 22.819084 | 41.964286 | 21.428571 |
| lm_F | 17.032005 | 28.839525 | 22.935765 | 42.261905 | 21.726190 |
| lm_C | 16.643061 | 28.889531 | 22.766296 | 41.964286 | 21.726190 |
| lm_FC | 16.643200 | 28.896211 | 22.769706 | 41.964286 | 21.428571 |
| lm_mixture_F | 17.027085 | 28.704059 | 22.865572 | 41.964286 | 21.428571 |
| lm_mixture_C | 16.655030 | 29.114102 | 22.884566 | 42.857143 | 21.726190 |
| lm_mixture_FC | 17.213545 | 29.082627 | 23.148086 | 42.857143 | 22.023810 |
| lm_shuffle_F | 16.745435 | 28.485851 | 22.615643 | 41.666667 | 21.428571 |
| lm_shuffle_C | 17.040758 | 28.774955 | 22.907857 | 42.559524 | 21.428571 |
| lm_shuffle_FC | 16.685487 | 28.456111 | 22.570799 | 42.261905 | 21.428571 |
| lm_capped_F | 17.081460 | 28.995564 | 23.038512 | 42.261905 | 21.726190 |
| lm_capped_C | 16.665015 | 29.065580 | 22.865297 | 42.261905 | 21.726190 |
| lm_capped_FC | 16.721152 | 29.052251 | 22.886701 | 41.964286 | 21.428571 |

## 固定配对的任务效应

| 方法 | 对照 | 平台 | Δpp | 改善/恶化/不变 | 换query | Acc25救回/破坏 |
|---|---|---|---:|---:|---:|---:|
| lm_FC | R91_frozen | balanced | -0.262340 | 13/15/308 | 54 | 4/4 |
| lm_FC | R91_frozen | drone | -0.702989 | 7/11/150 | 36 | 2/3 |
| lm_FC | R91_frozen | quad | +0.178310 | 6/4/158 | 18 | 2/1 |
| lm_FC | ABC | balanced | +0.467246 | 47/55/234 | 184 | 15/16 |
| lm_FC | ABC | drone | +0.906868 | 25/32/111 | 111 | 8/8 |
| lm_FC | ABC | quad | +0.027623 | 22/23/123 | 73 | 7/8 |
| lm_FC | R59 | balanced | +0.158590 | 47/60/229 | 189 | 14/18 |
| lm_FC | R59 | drone | +0.257632 | 23/38/107 | 115 | 7/10 |
| lm_FC | R59 | quad | +0.059548 | 24/22/122 | 74 | 7/8 |
| lm_FC | R94_direct | balanced | -0.390892 | 10/10/316 | 43 | 2/5 |
| lm_FC | R94_direct | drone | -0.683845 | 6/8/154 | 32 | 0/4 |
| lm_FC | R94_direct | quad | -0.097940 | 4/2/162 | 11 | 2/1 |
| lm_FC | R94_Adam | balanced | -0.215207 | 12/12/312 | 50 | 2/3 |
| lm_FC | R94_Adam | drone | -0.608723 | 6/8/154 | 33 | 0/2 |
| lm_FC | R94_Adam | quad | +0.178310 | 6/4/158 | 17 | 2/1 |
| lm_FC | R94_GN | balanced | -0.212341 | 12/12/312 | 41 | 2/3 |
| lm_FC | R94_GN | drone | -0.602990 | 6/8/154 | 26 | 0/2 |
| lm_FC | R94_GN | quad | +0.178307 | 6/4/158 | 15 | 2/1 |
| lm_FC | direct_FC | balanced | -0.006161 | 9/8/319 | 39 | 1/3 |
| lm_FC | direct_FC | drone | -0.030185 | 7/6/155 | 30 | 0/3 |
| lm_FC | direct_FC | quad | +0.017862 | 2/2/164 | 9 | 1/0 |
| lm_FC | direct_four | balanced | +0.032823 | 11/11/314 | 45 | 2/2 |
| lm_FC | direct_four | drone | +0.210029 | 9/6/153 | 32 | 1/2 |
| lm_FC | direct_four | quad | -0.144383 | 2/5/161 | 13 | 1/0 |
| lm_FC | mixture_FC | balanced | -0.298894 | 7/5/324 | 30 | 1/3 |
| lm_FC | mixture_FC | drone | -0.411372 | 7/3/158 | 22 | 1/2 |
| lm_FC | mixture_FC | quad | -0.186416 | 0/2/166 | 8 | 0/1 |
| lm_FC | lm_F | balanced | -0.166059 | 4/3/329 | 16 | 1/2 |
| lm_FC | lm_F | drone | -0.388805 | 2/2/164 | 11 | 0/2 |
| lm_FC | lm_F | quad | +0.056687 | 2/1/165 | 5 | 1/0 |
| lm_FC | lm_C | balanced | +0.003410 | 5/2/329 | 27 | 1/1 |
| lm_FC | lm_C | drone | +0.000139 | 3/1/164 | 17 | 0/0 |
| lm_FC | lm_C | quad | +0.006680 | 2/1/165 | 10 | 1/1 |
| lm_FC | lm_mixture_FC | balanced | -0.378380 | 4/6/326 | 27 | 0/3 |
| lm_FC | lm_mixture_FC | drone | -0.570345 | 4/4/160 | 20 | 0/2 |
| lm_FC | lm_mixture_FC | quad | -0.186416 | 0/2/166 | 7 | 0/1 |
| lm_FC | lm_shuffle_FC | balanced | +0.198907 | 24/17/295 | 88 | 4/5 |
| lm_FC | lm_shuffle_FC | drone | -0.042287 | 15/12/141 | 64 | 2/4 |
| lm_FC | lm_shuffle_FC | quad | +0.440101 | 9/5/154 | 24 | 2/1 |
| lm_FC | lm_capped_FC | balanced | -0.116996 | 3/5/328 | 20 | 1/1 |
| lm_FC | lm_capped_FC | drone | -0.077952 | 2/4/162 | 12 | 0/0 |
| lm_FC | lm_capped_FC | quad | -0.156040 | 1/1/166 | 8 | 1/1 |
| lm_F | R91_frozen | balanced | -0.096281 | 12/13/311 | 54 | 4/3 |
| lm_F | R91_frozen | drone | -0.314184 | 8/10/150 | 39 | 3/2 |
| lm_F | R91_frozen | quad | +0.121623 | 4/3/161 | 15 | 1/1 |
| lm_F | R94_direct | balanced | -0.224833 | 6/7/323 | 31 | 1/3 |
| lm_F | R94_direct | drone | -0.295040 | 4/6/158 | 25 | 0/2 |
| lm_F | R94_direct | quad | -0.154626 | 2/1/165 | 6 | 1/1 |
| lm_F | R94_mixture | balanced | -0.113607 | 8/6/322 | 34 | 2/3 |
| lm_F | R94_mixture | drone | -0.362215 | 6/4/158 | 26 | 1/2 |
| lm_F | R94_mixture | quad | +0.135000 | 2/2/164 | 8 | 1/1 |
| lm_F | R94_Adam | balanced | -0.049148 | 10/10/316 | 48 | 2/2 |
| lm_F | R94_Adam | drone | -0.219918 | 6/7/155 | 34 | 1/1 |
| lm_F | R94_Adam | quad | +0.121623 | 4/3/161 | 14 | 1/1 |
| lm_F | R94_GN | balanced | -0.046282 | 10/10/316 | 40 | 2/2 |
| lm_F | R94_GN | drone | -0.214185 | 6/7/155 | 28 | 1/1 |
| lm_F | R94_GN | quad | +0.121620 | 4/3/161 | 12 | 1/1 |
| lm_F | lm_shuffle_F | balanced | +0.320122 | 19/17/300 | 83 | 5/3 |
| lm_F | lm_shuffle_F | drone | +0.286570 | 11/11/146 | 58 | 2/1 |
| lm_F | lm_shuffle_F | quad | +0.353674 | 8/6/154 | 25 | 3/2 |
| lm_F | lm_capped_F | balanced | -0.102747 | 1/3/332 | 13 | 1/1 |
| lm_F | lm_capped_F | drone | -0.049455 | 0/2/166 | 10 | 0/0 |
| lm_F | lm_capped_F | quad | -0.156040 | 1/1/166 | 3 | 1/1 |
| lm_C | frozen_C | balanced | -0.350605 | 10/15/311 | 52 | 2/4 |
| lm_C | frozen_C | drone | -0.453422 | 7/9/152 | 34 | 2/2 |
| lm_C | frozen_C | quad | -0.247788 | 3/6/159 | 18 | 0/2 |
| lm_C | direct_C | balanced | +0.128581 | 9/13/314 | 41 | 2/3 |
| lm_C | direct_C | drone | +0.099920 | 6/7/155 | 30 | 1/2 |
| lm_C | direct_C | quad | +0.157243 | 3/6/159 | 11 | 1/1 |
| lm_C | lm_shuffle_C | balanced | -0.141560 | 21/17/298 | 82 | 2/4 |
| lm_C | lm_shuffle_C | drone | -0.397697 | 13/13/142 | 62 | 2/4 |
| lm_C | lm_shuffle_C | quad | +0.114576 | 8/4/156 | 20 | 0/0 |
| direct_FC | R91_frozen | balanced | -0.256178 | 17/19/300 | 70 | 6/4 |
| direct_FC | R91_frozen | drone | -0.672804 | 10/13/145 | 48 | 4/2 |
| direct_FC | R91_frozen | quad | +0.160448 | 7/6/155 | 22 | 2/2 |
| direct_FC | R94_direct | balanced | -0.384731 | 6/7/323 | 22 | 2/3 |
| direct_FC | R94_direct | drone | -0.653660 | 2/5/161 | 15 | 1/2 |
| direct_FC | R94_direct | quad | -0.115802 | 4/2/162 | 7 | 1/1 |
| direct_FC | direct_four | balanced | +0.038984 | 3/4/329 | 16 | 2/0 |
| direct_FC | direct_four | drone | +0.240214 | 3/1/164 | 12 | 2/0 |
| direct_FC | direct_four | quad | -0.162246 | 0/3/165 | 4 | 0/0 |
| direct_FC | shuffle_FC | balanced | -0.043217 | 24/22/290 | 101 | 6/4 |
| direct_FC | shuffle_FC | drone | -0.303529 | 15/13/140 | 68 | 3/2 |
| direct_FC | shuffle_FC | quad | +0.217095 | 9/9/150 | 33 | 3/2 |

## 执行能力与任务收益分开

| 目标 | n | 平均初始/最终误差 | 平均相对误差 | 平均接受步数 | 平均参数L2 |
|---|---:|---:|---:|---:|---:|
| lm | 336 | 0.0098639522/0.0021635386 | 0.24417548 | 10.000000 | 0.722118 |
| lm_mixture | 336 | 0.0023648199/0.0004582527 | 0.21323412 | 10.000000 | 0.626399 |
| lm_shuffle | 336 | 0.0098639522/0.0060687696 | 0.61220069 | 10.000000 | 1.319990 |
| lm_capped | 336 | 0.0098639522/0.0018410236 | 0.21271548 | 10.000000 | 1.136956 |

原q要求改Top1=60条；LM_F实际与q第一名一致=39/60。全体LM_F与q一致=305/336。它说明监督执行程度，不说明这些选择正确。

## 五个场景

| 场景 | n | 方法 | mIoU % |
|---|---:|---|---:|
| D_S1 | 106 | R59 | 11.761612 |
| D_S1 | 106 | R91_frozen | 13.989152 |
| D_S1 | 106 | R94_direct | 14.120446 |
| D_S1 | 106 | direct_FC | 12.972183 |
| D_S1 | 106 | direct_four | 12.721537 |
| D_S1 | 106 | lm_F | 13.504888 |
| D_S1 | 106 | lm_C | 12.873486 |
| D_S1 | 106 | lm_FC | 12.778628 |
| D_S1 | 106 | lm_shuffle_FC | 13.129298 |
| D_S2 | 62 | R59 | 24.291041 |
| D_S2 | 62 | R91_frozen | 23.085641 |
| D_S2 | 62 | R94_direct | 22.809294 |
| D_S2 | 62 | direct_FC | 23.001247 |
| D_S2 | 62 | direct_four | 22.778868 |
| D_S2 | 62 | lm_F | 23.062237 |
| D_S2 | 62 | lm_C | 23.087820 |
| D_S2 | 62 | lm_FC | 23.250372 |
| D_S2 | 62 | lm_shuffle_FC | 22.765424 |
| Q_S1 | 62 | R59 | 9.632116 |
| Q_S1 | 62 | R91_frozen | 10.708624 |
| Q_S1 | 62 | R94_direct | 10.693794 |
| Q_S1 | 62 | direct_FC | 10.570452 |
| Q_S1 | 62 | direct_four | 11.010085 |
| Q_S1 | 62 | lm_F | 10.274807 |
| Q_S1 | 62 | lm_C | 10.600751 |
| Q_S1 | 62 | lm_FC | 10.618853 |
| Q_S1 | 62 | lm_shuffle_FC | 10.426720 |
| Q_S2 | 49 | R59 | 36.067581 |
| Q_S2 | 49 | R91_frozen | 35.770023 |
| Q_S2 | 49 | R94_direct | 35.638776 |
| Q_S2 | 49 | direct_FC | 35.397808 |
| Q_S2 | 49 | direct_four | 35.397808 |
| Q_S2 | 49 | lm_F | 35.638776 |
| Q_S2 | 49 | lm_C | 35.397808 |
| Q_S2 | 49 | lm_FC | 35.397808 |
| Q_S2 | 49 | lm_shuffle_FC | 35.229149 |
| Q_S3 | 57 | R59 | 43.509770 |
| Q_S3 | 57 | R91_frozen | 42.244590 |
| Q_S3 | 57 | R94_direct | 43.187756 |
| Q_S3 | 57 | direct_FC | 43.187756 |
| Q_S3 | 57 | direct_four | 43.187756 |
| Q_S3 | 57 | lm_F | 43.187756 |
| Q_S3 | 57 | lm_C | 43.187756 |
| Q_S3 | 57 | lm_FC | 43.187756 |
| Q_S3 | 57 | lm_shuffle_FC | 42.244590 |

## 固定回访与总体正负尾部

案例仅在封存后按固定回访、总体最大正负贡献抽取，保留原分母；不根据案例挑步幅或输出。以下是数值对照，不将IoU变化自动命名为物体身份改变。

| 案例 | 平台 | 原因 | R91零步 | R94q直接 | LM_F | 冻结C | LM_C | direct_FC | LM_FC |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| CASE001 | drone | fixed_prior_case;harm_lm_FC_vs_R91_frozen;harm_lm_F_vs_R94_direct | 0.452964 | 0.452964 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| CASE002 | quad | fixed_prior_case;gain_lm_FC_vs_R91_frozen | 0.000000 | 0.537604 | 0.537604 | 0.537604 | 0.537604 | 0.537604 | 0.537604 |
| CASE003 | drone | gain_lm_FC_vs_R91_frozen | 0.000000 | 0.318500 | 0.318500 | 0.318500 | 0.318500 | 0.318500 | 0.318500 |
| CASE004 | drone | harm_lm_FC_vs_R91_frozen;harm_lm_C_vs_frozen_C | 0.769836 | 0.769836 | 0.769836 | 0.769836 | 0.000000 | 0.000000 | 0.000000 |
| CASE005 | drone | gain_lm_FC_vs_direct_FC;gain_lm_F_vs_R94_direct | 0.511573 | 0.372176 | 0.511573 | 0.511573 | 0.511573 | 0.372176 | 0.511573 |
| CASE006 | drone | gain_lm_FC_vs_direct_FC;gain_lm_F_vs_R94_direct | 0.394054 | 0.312267 | 0.394054 | 0.394054 | 0.394054 | 0.312267 | 0.394054 |
| CASE007 | drone | harm_lm_FC_vs_direct_FC | 0.410834 | 0.410834 | 0.217958 | 0.410834 | 0.217958 | 0.410834 | 0.217958 |
| CASE008 | drone | harm_lm_FC_vs_direct_FC | 0.229251 | 0.274477 | 0.274477 | 0.229251 | 0.229251 | 0.274477 | 0.229251 |
| CASE009 | drone | gain_lm_C_vs_frozen_C | 0.255694 | 0.255694 | 0.255694 | 0.000000 | 0.255694 | 0.255694 | 0.255694 |
| CASE010 | drone | gain_lm_C_vs_frozen_C | 0.000000 | 0.135771 | 0.135771 | 0.000000 | 0.135771 | 0.135771 | 0.135771 |
| CASE011 | quad | harm_lm_C_vs_frozen_C;harm_lm_F_vs_R94_direct | 0.295762 | 0.295762 | 0.000000 | 0.295762 | 0.000000 | 0.000000 | 0.000000 |

## 审计、成本与不采纳边界

独立重建全部LM试探状态、真实量化位移、gain ratio、接受/回滚和原生后缀。重放18689trial、6048端点，概率最大误差1.11022302463e-16、参数5.82076609135e-11、loss1.38777878078e-16、gain ratio3.12638803734e-13，Top1不一致0。

预测阶段118.174s，新增C原生调用336，head试探18689，GPU峰值683687936bytes。R94 factual/错配证据及源前缀缓存复用，不能当作冷启动端到端时延。

不按平台选择不同方法，不选最好控制替换主方法，不访问FINAL，不为微小开发均值构造独立场景置信结论。R93历史raw开发读出23.203782%仍高于R94q的23.160598%，不得将局部修复宣布为全历史最佳。所有未知、未更新、负比较、接受/拒绝试探均在公开匿名结果中保留。

[Ceres官方非线性最小二乘文档](https://ceres-solver.readthedocs.io/latest/nnls_solving.html)提供实际/预测下降比控制步长的标准依据。本轮是独立实现的LM，不是Ceres复现，也不将优化器本身作为新颖性。

最终预设检查=False，稳健采纳=False，独立确认=False。执行拟合的修复、同信息任务增量和未拟合遮挡迁移必须分别解释。


## 修复裁决与剩余问题

R95 fixes an execution limitation but does not repair the entire grounding pipeline. All following interpretations are descriptive on historically exposed development data.

- Same factual-view target: R94 one-shot GN executed 11/60 requested Top1 changes; R95 executes 39/60, and agrees with the target on 305/336 inputs. Mean remaining centered-logit squared error is 24.4175%. All ten iterations were used, so complete convergence is not established.
- The same dense correction changes factual-view mIoU by +0.128552 pp but changes the unfitted C-view result by -0.479186 pp. This intervention shows that reusing the correction is not safe across masking conditions. It does not by itself identify the precise image-instance identity error.
- Parameter transfer on C is -0.350605 pp. The main FC update is -0.262340 pp versus its own zero and -0.006161 pp versus same-information direct use. It fails the locked task criterion.
- The source-mixture secondary yields 23.148086% versus 23.068600% for its direct counterpart. This is a disclosed secondary positive comparison, not a replacement for the failed primary; it also stays below factual-only R94 direct 23.160598%.
- Same-input fitting is not enough: wrong target choices, candidate-coupling through the shared head, residual fitting error, and observation-dependent corrections all remain possible. None is resolved by proving that parameters changed.
- Stop this fixed R95 recipe. Retain the R91 model and the R94 direct evidence as development references. No robust new TTA adoption, source retraining, architecture replacement, or FINAL evaluation is claimed. A further iteration would need to establish observation-conditioned, task-useful evidence rather than merely push the same target harder.


## 原图复核（封存后描述性分析）

全部11页完整图、局部图、固定框俯视和排名表已查看。主体/参照物身份没有独立实例标注复核，不将这些观察升级为测试标签。

| 案例 | 现象 |
|---|---|
| CASE001 | Direct factual keeps the nearby vehicle; factual head edit and all C-containing readouts select a distant region. |
| CASE002 | Factual head edit follows direct evidence to the left-side person. Frozen C already selects the same person, so this case does not establish additional C adaptation value. |
| CASE003 | Direct and adapted results select the visible border vehicle region; frozen C already does so. |
| CASE004 | Factual head keeps the vehicle box; adapted C and C-containing fusion select a wider neighboring support region. Precise object identity of that region remains unknown. |
| CASE005 | Parameter result restores the original overlapping vehicle box; direct selects an offset overlapping box. This is not evidence of new instance recognition. |
| CASE006 | Parameter result restores source geometry for an overlapping vehicle hypothesis, rather than finding a new target. |
| CASE007 | Adaptation picks a shifted overlapping vehicle box. IoU decline alone must not be described as a confirmed different-object switch. |
| CASE008 | Factual adaptation follows the better overlapping SUV box; C and FC revert to the original duplicate, losing the Acc25 crossing. |
| CASE009 | Adapted C recovers the vehicle hypothesis already selected in the source factual view. This is restoration, not a new target discovery. |
| CASE010 | A tree or elevated background hypothesis gives way to the van region, but final IoU stays below Acc25. |
| CASE011 | Factual and C adaptation select an elevated building or vegetation region instead of the visible vehicle. |
