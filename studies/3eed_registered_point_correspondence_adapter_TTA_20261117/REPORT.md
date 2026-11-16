# R123 — 注册点对应驱动的真实点Adapter TTA

本轮是实际参数适应，不是只改最终分数。仍固定三个epoch100、现有R91双向结构、完整文字、原A256与六层空间锚点；没有新source准备、外部专家、PointNet推理、原始媒体或FINAL访问。采用与否按预先锁定的全部总体对照判定。

## 单一机制与计算路径

R122显示打乱空间特征可显著损害选框，但精确几何中位数未提供稳定收益。本轮不拟合失败的中位数G，也不再更换GN恢复损失。直接使用真实注册扫描A/B的pre-language点特征，适应现有point_adapter.up的4896个参数，再重算原生cross-encoder/decoder的候选和文字表示。

从已验证的native foundation开始：原source准备的pointAdapter输出权重、偏置及早期RGB门置零，固定原source down投影；不是给旧source Adapter继续叠加不明偏移。只编辑输出288×16权重及288偏置。每条输入恢复原source；三个固定归一化梯度步骤，每步半径为原source完整pointAdapter范数的1%。没有按GT/辅助loss做逐样本步长搜索。

A/B有效支持由旧注册确定，位置相距大于3.2米才作负区域，近邻与重复支持不互斥。正对应是近似注册空间支持，不是经认证的同一物体。对称InfoNCE温度0.07加单位权重的归一化source特征保持；不生成文字目标伪标签，不使用颜色/左右规则。

原R74是在同帧采样的postdecoder候选区域和matching MLP做对比；本轮改变实际扫描、监督单位和进入语言推理前的计算路径。PointContrast论文与官方实现是借鉴依据，但原方法长程预训练后有监督微调，不能借其结果保证当前短步TTA有效，见LITERATURE.md。

## 同信息强对照与C边界

全部参数/自由特征更新先固定，随后才解码C数组与旧C上下文提供者。C不参与梯度或停止规则。最终F读出允许像所有基线一样使用已有ABC观测，因此不是严格在线或全新场景验证。C区域检索和C选框作为未用于拟合的观测检查。

对照包含：等权点特征mean、原始headmean、R122稳健聚合；同loss同预算的错误注册；与实际参数改变量等范数的随机参数；同A/B loss直接优化自由特征并逐步匹配实际Adapter引起的特征变化能量。最后一项不改网络，通过A/B平均修正场作用于F/C，是强但非全局最优的无更新对照。错误注册同时破坏正负绑定，不能狭称只测试正样本身份。

固定RGB late取sourceA，竞争残差取sourceMeanF，均与零步及全部新臂相同。保留source双向RGB上下文，但新更新仅针对LiDAR descriptor，不能宣称本轮已经完成联合RGB参数适应。

## 数据、工程与封存

DEV768与EXP1536均历史曝光；154保护别名先检查且不加载特征，使用原prepared_A回退并保留分母。各方向等权平均；另用旧全量场景频数后分层，不按GT加权。EXP缺少4个Waymo场景，只报告共同支持范围。当前headmean不是历史R59的软query传输，历史DEV336 R59仍保留原研究条目，不能在不同清单伪造同条件数值。

24条真实工程验证零步F/C候选和文字表示与旧缓存逐值相同。实际Up写回与等价点特征输入的h/U逐值相同是正常的数学等价，仅证明实现执行，不算额外收益。25臂×2304条共57600任务值，32256新预测、25344复用基线；所有预测先封存，再独立复算CPU梯度、自由特征、更新范数、head、框IoU及总体/场景/未拟合C。

## 总体结果

| 数据 | 方法 | mIoU% | Acc25% | Acc50% |
|---|---|---:|---:|---:|
| DEV | original_source | 16.811383 | 30.989583 | 14.973958 |
| DEV | prepared_A | 16.228652 | 29.947917 | 12.630208 |
| DEV | prepared_F | 16.703230 | 30.989583 | 13.151042 |
| DEV | native_F | 16.974318 | 31.250000 | 14.843750 |
| DEV | native_ABC | 16.734016 | 30.729167 | 14.843750 |
| DEV | lateOnly_F | 17.180150 | 31.901042 | 14.453125 |
| DEV | mean_context | 17.349002 | 31.901042 | 14.453125 |
| DEV | headmean_context | 17.176454 | 31.640625 | 14.583333 |
| DEV | robust_context | 17.383000 | 32.161458 | 14.583333 |
| DEV | weight_mean_context | 17.385491 | 32.031250 | 14.453125 |
| DEV | zero_C | 16.493471 | 30.729167 | 13.671875 |
| DEV | corr1_F | 17.385860 | 32.291667 | 14.322917 |
| DEV | corr1_C | 16.430404 | 30.598958 | 13.541667 |
| DEV | corr3_F | 17.233621 | 31.770833 | 14.322917 |
| DEV | corr3_C | 16.369542 | 30.468750 | 13.541667 |
| DEV | wrong1_F | 17.349002 | 31.901042 | 14.453125 |
| DEV | wrong1_C | 16.415806 | 30.598958 | 13.541667 |
| DEV | wrong3_F | 17.326923 | 31.901042 | 14.453125 |
| DEV | wrong3_C | 16.281354 | 30.208333 | 13.671875 |
| DEV | random1_F | 17.349002 | 31.901042 | 14.453125 |
| DEV | random1_C | 16.493471 | 30.729167 | 13.671875 |
| DEV | random3_F | 17.351293 | 32.031250 | 14.453125 |
| DEV | random3_C | 16.364259 | 30.468750 | 13.541667 |
| DEV | free3_F | 17.199501 | 31.640625 | 14.192708 |
| DEV | free3_C | 16.338490 | 30.338542 | 13.671875 |
| EXP | original_source | 16.144619 | 30.598958 | 12.434896 |
| EXP | prepared_A | 16.251067 | 30.598958 | 11.783854 |
| EXP | prepared_F | 16.238436 | 30.598958 | 12.044271 |
| EXP | native_F | 16.237263 | 30.859375 | 12.434896 |
| EXP | native_ABC | 16.321800 | 31.250000 | 12.434896 |
| EXP | lateOnly_F | 16.439770 | 31.184896 | 12.174479 |
| EXP | mean_context | 16.499566 | 31.119792 | 12.174479 |
| EXP | headmean_context | 16.533556 | 31.315104 | 12.369792 |
| EXP | robust_context | 16.435239 | 31.184896 | 12.239583 |
| EXP | weight_mean_context | 16.434424 | 30.989583 | 12.109375 |
| EXP | zero_C | 16.185584 | 30.794271 | 12.174479 |
| EXP | corr1_F | 16.410656 | 31.119792 | 12.044271 |
| EXP | corr1_C | 16.117860 | 30.729167 | 12.044271 |
| EXP | corr3_F | 16.410475 | 31.184896 | 11.979167 |
| EXP | corr3_C | 16.157884 | 30.924479 | 12.109375 |
| EXP | wrong1_F | 16.445769 | 30.989583 | 12.109375 |
| EXP | wrong1_C | 16.188449 | 30.729167 | 12.174479 |
| EXP | wrong3_F | 16.413569 | 31.054688 | 12.044271 |
| EXP | wrong3_C | 16.177642 | 30.598958 | 12.174479 |
| EXP | random1_F | 16.499566 | 31.119792 | 12.174479 |
| EXP | random1_C | 16.199495 | 30.794271 | 12.174479 |
| EXP | random3_F | 16.474116 | 31.054688 | 12.174479 |
| EXP | random3_C | 16.152830 | 30.664062 | 12.174479 |
| EXP | free3_F | 16.396969 | 30.989583 | 12.109375 |
| EXP | free3_C | 16.169274 | 30.859375 | 12.044271 |

## 各目标平台

| 数据 | 目标 | 方法 | mIoU% | Acc25% | Acc50% |
|---|---|---|---:|---:|---:|
| DEV | waymo | original_source | 11.473631 | 23.437500 | 7.421875 |
| DEV | waymo | prepared_A | 12.615516 | 26.171875 | 5.078125 |
| DEV | waymo | mean_context | 12.247576 | 24.609375 | 7.812500 |
| DEV | waymo | headmean_context | 12.097959 | 24.218750 | 7.812500 |
| DEV | waymo | zero_C | 12.008417 | 23.828125 | 7.812500 |
| DEV | waymo | corr1_F | 12.299002 | 25.000000 | 7.812500 |
| DEV | waymo | corr3_F | 12.222110 | 25.000000 | 7.812500 |
| DEV | waymo | corr3_C | 12.055980 | 24.218750 | 7.812500 |
| DEV | waymo | wrong3_F | 12.405996 | 25.000000 | 7.812500 |
| DEV | waymo | random3_F | 12.262906 | 25.000000 | 7.812500 |
| DEV | waymo | free3_F | 12.249959 | 24.609375 | 7.812500 |
| DEV | drone | original_source | 13.867693 | 24.609375 | 12.500000 |
| DEV | drone | prepared_A | 14.071678 | 24.609375 | 11.718750 |
| DEV | drone | mean_context | 14.805341 | 26.953125 | 11.718750 |
| DEV | drone | headmean_context | 14.514873 | 26.171875 | 11.718750 |
| DEV | drone | zero_C | 14.545494 | 26.953125 | 11.328125 |
| DEV | drone | corr1_F | 14.930947 | 27.343750 | 11.718750 |
| DEV | drone | corr3_F | 14.732353 | 26.171875 | 11.718750 |
| DEV | drone | corr3_C | 14.478907 | 26.562500 | 11.328125 |
| DEV | drone | wrong3_F | 14.711111 | 26.953125 | 11.718750 |
| DEV | drone | random3_F | 14.796885 | 26.953125 | 11.718750 |
| DEV | drone | free3_F | 14.921881 | 27.343750 | 11.718750 |
| DEV | quad | original_source | 25.092825 | 44.921875 | 25.000000 |
| DEV | quad | prepared_A | 21.998762 | 39.062500 | 21.093750 |
| DEV | quad | mean_context | 24.994087 | 44.140625 | 23.828125 |
| DEV | quad | headmean_context | 24.916529 | 44.531250 | 24.218750 |
| DEV | quad | zero_C | 22.926502 | 41.406250 | 21.875000 |
| DEV | quad | corr1_F | 24.927630 | 44.531250 | 23.437500 |
| DEV | quad | corr3_F | 24.746399 | 44.140625 | 23.437500 |
| DEV | quad | corr3_C | 22.573739 | 40.625000 | 21.484375 |
| DEV | quad | wrong3_F | 24.863664 | 43.750000 | 23.828125 |
| DEV | quad | random3_F | 24.994087 | 44.140625 | 23.828125 |
| DEV | quad | free3_F | 24.426663 | 42.968750 | 23.046875 |
| EXP | waymo | original_source | 13.450978 | 28.906250 | 6.835938 |
| EXP | waymo | prepared_A | 13.214120 | 28.125000 | 6.250000 |
| EXP | waymo | mean_context | 12.855988 | 27.343750 | 6.054688 |
| EXP | waymo | headmean_context | 12.988950 | 27.734375 | 6.054688 |
| EXP | waymo | zero_C | 12.799592 | 27.343750 | 6.054688 |
| EXP | waymo | corr1_F | 12.702878 | 27.148438 | 5.859375 |
| EXP | waymo | corr3_F | 12.764602 | 27.343750 | 5.859375 |
| EXP | waymo | corr3_C | 12.654569 | 26.953125 | 6.054688 |
| EXP | waymo | wrong3_F | 12.866941 | 27.734375 | 6.054688 |
| EXP | waymo | random3_F | 12.790343 | 27.148438 | 6.054688 |
| EXP | waymo | free3_F | 12.600801 | 27.148438 | 5.859375 |
| EXP | drone | original_source | 11.525665 | 20.898438 | 9.375000 |
| EXP | drone | prepared_A | 12.009923 | 20.312500 | 9.179688 |
| EXP | drone | mean_context | 12.279426 | 22.265625 | 9.570312 |
| EXP | drone | headmean_context | 12.240413 | 22.070312 | 9.765625 |
| EXP | drone | zero_C | 11.958571 | 21.289062 | 9.375000 |
| EXP | drone | corr1_F | 12.302482 | 22.265625 | 9.570312 |
| EXP | drone | corr3_F | 12.233867 | 22.265625 | 9.375000 |
| EXP | drone | corr3_C | 11.761259 | 20.898438 | 9.179688 |
| EXP | drone | wrong3_F | 12.336122 | 22.460938 | 9.570312 |
| EXP | drone | random3_F | 12.279663 | 22.265625 | 9.570312 |
| EXP | drone | free3_F | 12.318410 | 22.265625 | 9.570312 |
| EXP | quad | original_source | 23.457215 | 41.992188 | 21.093750 |
| EXP | quad | prepared_A | 23.529157 | 43.359375 | 19.921875 |
| EXP | quad | mean_context | 24.363284 | 43.750000 | 20.898438 |
| EXP | quad | headmean_context | 24.371304 | 44.140625 | 21.289062 |
| EXP | quad | zero_C | 23.798588 | 43.750000 | 21.093750 |
| EXP | quad | corr1_F | 24.226608 | 43.945312 | 20.703125 |
| EXP | quad | corr3_F | 24.232957 | 43.945312 | 20.703125 |
| EXP | quad | corr3_C | 24.057824 | 44.921875 | 21.093750 |
| EXP | quad | wrong3_F | 24.037643 | 42.968750 | 20.507812 |
| EXP | quad | random3_F | 24.352342 | 43.750000 | 20.898438 |
| EXP | quad | free3_F | 24.271695 | 43.554688 | 20.898438 |

## 场景频数与未拟合C

| 数据 | 方法 | 最小场景支持覆盖 | 后分层mIoU% |
|---|---|---:|---:|
| DEV | original_source | 1.000000 | 12.186315 |
| DEV | prepared_A | 1.000000 | 11.764418 |
| DEV | prepared_F | 1.000000 | 11.980683 |
| DEV | native_F | 1.000000 | 12.237360 |
| DEV | native_ABC | 1.000000 | 11.834713 |
| DEV | lateOnly_F | 1.000000 | 12.479188 |
| DEV | mean_context | 1.000000 | 12.673432 |
| DEV | headmean_context | 1.000000 | 12.326706 |
| DEV | robust_context | 1.000000 | 12.646340 |
| DEV | weight_mean_context | 1.000000 | 12.698587 |
| DEV | zero_C | 1.000000 | 12.169366 |
| DEV | corr1_F | 1.000000 | 12.685743 |
| DEV | corr1_C | 1.000000 | 12.143411 |
| DEV | corr3_F | 1.000000 | 12.362621 |
| DEV | corr3_C | 1.000000 | 12.075644 |
| DEV | wrong1_F | 1.000000 | 12.673432 |
| DEV | wrong1_C | 1.000000 | 12.131342 |
| DEV | wrong3_F | 1.000000 | 12.778678 |
| DEV | wrong3_C | 1.000000 | 12.104260 |
| DEV | random1_F | 1.000000 | 12.673432 |
| DEV | random1_C | 1.000000 | 12.169366 |
| DEV | random3_F | 1.000000 | 12.671792 |
| DEV | random3_C | 1.000000 | 12.124658 |
| DEV | free3_F | 1.000000 | 12.462132 |
| DEV | free3_C | 1.000000 | 12.141788 |
| EXP | original_source | 0.998149 | 12.339915 |
| EXP | prepared_A | 0.998149 | 13.127455 |
| EXP | prepared_F | 0.998149 | 12.899456 |
| EXP | native_F | 0.998149 | 12.670824 |
| EXP | native_ABC | 0.998149 | 12.817131 |
| EXP | lateOnly_F | 0.998149 | 12.880574 |
| EXP | mean_context | 0.998149 | 12.951703 |
| EXP | headmean_context | 0.998149 | 12.951158 |
| EXP | robust_context | 0.998149 | 12.772886 |
| EXP | weight_mean_context | 0.998149 | 12.817846 |
| EXP | zero_C | 0.998149 | 12.599336 |
| EXP | corr1_F | 0.998149 | 12.822955 |
| EXP | corr1_C | 0.998149 | 12.557673 |
| EXP | corr3_F | 0.998149 | 12.847571 |
| EXP | corr3_C | 0.998149 | 12.538529 |
| EXP | wrong1_F | 0.998149 | 12.889409 |
| EXP | wrong1_C | 0.998149 | 12.661100 |
| EXP | wrong3_F | 0.998149 | 12.972140 |
| EXP | wrong3_C | 0.998149 | 12.686938 |
| EXP | random1_F | 0.998149 | 12.951703 |
| EXP | random1_C | 0.998149 | 12.629566 |
| EXP | random3_F | 0.998149 | 12.881096 |
| EXP | random3_C | 0.998149 | 12.592157 |
| EXP | free3_F | 0.998149 | 12.842364 |
| EXP | free3_C | 0.998149 | 12.744221 |

## HELDOUT_C_TASK_ROWS

```json
[
  {
    "cohort": "DEV",
    "source": "drone",
    "target": "quad",
    "n": 62,
    "gain_pp": 0.4197791166395688,
    "better": 3,
    "worse": 2
  },
  {
    "cohort": "DEV",
    "source": "drone",
    "target": "waymo",
    "n": 17,
    "gain_pp": 0.2308453037329696,
    "better": 1,
    "worse": 0
  },
  {
    "cohort": "DEV",
    "source": "quad",
    "target": "drone",
    "n": 10,
    "gain_pp": 0.0,
    "better": 0,
    "worse": 0
  },
  {
    "cohort": "DEV",
    "source": "quad",
    "target": "waymo",
    "n": 17,
    "gain_pp": 0.48540869110279977,
    "better": 1,
    "worse": 0
  },
  {
    "cohort": "DEV",
    "source": "waymo",
    "target": "drone",
    "n": 10,
    "gain_pp": -3.4885222890665846,
    "better": 0,
    "worse": 1
  },
  {
    "cohort": "DEV",
    "source": "waymo",
    "target": "quad",
    "n": 62,
    "gain_pp": -1.8763486045610924,
    "better": 0,
    "worse": 2
  },
  {
    "cohort": "EXP",
    "source": "drone",
    "target": "quad",
    "n": 109,
    "gain_pp": 0.4018080878791758,
    "better": 3,
    "worse": 2
  },
  {
    "cohort": "EXP",
    "source": "drone",
    "target": "waymo",
    "n": 40,
    "gain_pp": -0.27717950484816817,
    "better": 1,
    "worse": 1
  },
  {
    "cohort": "EXP",
    "source": "quad",
    "target": "drone",
    "n": 26,
    "gain_pp": 0.0,
    "better": 0,
    "worse": 0
  },
  {
    "cohort": "EXP",
    "source": "quad",
    "target": "waymo",
    "n": 40,
    "gain_pp": 0.0694086846458244,
    "better": 1,
    "worse": 0
  },
  {
    "cohort": "EXP",
    "source": "waymo",
    "target": "drone",
    "n": 26,
    "gain_pp": 0.0,
    "better": 0,
    "worse": 0
  },
  {
    "cohort": "EXP",
    "source": "waymo",
    "target": "quad",
    "n": 109,
    "gain_pp": 0.26609775544946607,
    "better": 2,
    "worse": 0
  }
]
```

## CORRESPONDENCE_COMPONENT_SUMMARY

```json
[
  {
    "cohort": "DEV",
    "source": "waymo",
    "n_nonalias": 234,
    "active_AB": 108,
    "mean_pairs": 326.9102564102564,
    "mean_actual_correct_wrong_first_step_cosine": -0.7206411511784859,
    "corr1_fit_objective_n": 108,
    "corr1_fit_objective_mean_change": -0.05162822317194055,
    "corr1_fit_objective_improved": 108,
    "corr1_heldout_corr_n": 72,
    "corr1_heldout_corr_mean_change": -0.05159657862451342,
    "corr1_heldout_corr_improved": 72,
    "corr1_heldout_retrieval_n": 72,
    "corr1_heldout_retrieval_mean_change": 0.0004375461075041029,
    "corr1_heldout_retrieval_improved": 25,
    "corr3_fit_objective_n": 108,
    "corr3_fit_objective_mean_change": -0.15228201504106875,
    "corr3_fit_objective_improved": 108,
    "corr3_heldout_corr_n": 72,
    "corr3_heldout_corr_mean_change": -0.15329431494077048,
    "corr3_heldout_corr_improved": 72,
    "corr3_heldout_retrieval_n": 72,
    "corr3_heldout_retrieval_mean_change": 0.0019500342508157094,
    "corr3_heldout_retrieval_improved": 48,
    "wrong3_fit_objective_n": 108,
    "wrong3_fit_objective_mean_change": 0.12573283248477513,
    "wrong3_fit_objective_improved": 0,
    "wrong3_heldout_corr_n": 72,
    "wrong3_heldout_corr_mean_change": 0.12570587131712171,
    "wrong3_heldout_corr_improved": 0,
    "wrong3_heldout_retrieval_n": 72,
    "wrong3_heldout_retrieval_mean_change": 0.0015797122485107845,
    "wrong3_heldout_retrieval_improved": 47,
    "random3_fit_objective_n": 108,
    "random3_fit_objective_mean_change": 0.0007140548140914352,
    "random3_fit_objective_improved": 40,
    "random3_heldout_corr_n": 72,
    "random3_heldout_corr_mean_change": 0.0004037883546617296,
    "random3_heldout_corr_improved": 28,
    "random3_heldout_retrieval_n": 72,
    "random3_heldout_retrieval_mean_change": -0.00019599621494611105,
    "random3_heldout_retrieval_improved": 9,
    "free3_fit_objective_n": 108,
    "free3_fit_objective_mean_change": -0.6447578834162818,
    "free3_fit_objective_improved": 108,
    "free3_heldout_corr_n": 72,
    "free3_heldout_corr_mean_change": -0.44194180766741437,
    "free3_heldout_corr_improved": 72,
    "free3_heldout_retrieval_n": 72,
    "free3_heldout_retrieval_mean_change": 0.09536530822515488,
    "free3_heldout_retrieval_improved": 72
  },
  {
    "cohort": "DEV",
    "source": "drone",
    "n_nonalias": 244,
    "active_AB": 101,
    "mean_pairs": 294.12704918032784,
    "mean_actual_correct_wrong_first_step_cosine": -0.9040490420969014,
    "corr1_fit_objective_n": 101,
    "corr1_fit_objective_mean_change": -0.042052063611474365,
    "corr1_fit_objective_improved": 101,
    "corr1_heldout_corr_n": 79,
    "corr1_heldout_corr_mean_change": -0.04215903825397733,
    "corr1_heldout_corr_improved": 79,
    "corr1_heldout_retrieval_n": 79,
    "corr1_heldout_retrieval_mean_change": 0.00012452202507212193,
    "corr1_heldout_retrieval_improved": 16,
    "corr3_fit_objective_n": 101,
    "corr3_fit_objective_mean_change": -0.12072038886570695,
    "corr3_fit_objective_improved": 101,
    "corr3_heldout_corr_n": 79,
    "corr3_heldout_corr_mean_change": -0.12285070781466327,
    "corr3_heldout_corr_improved": 79,
    "corr3_heldout_retrieval_n": 79,
    "corr3_heldout_retrieval_mean_change": 0.000542727452290209,
    "corr3_heldout_retrieval_improved": 34,
    "wrong3_fit_objective_n": 101,
    "wrong3_fit_objective_mean_change": 0.1275253697197036,
    "wrong3_fit_objective_improved": 0,
    "wrong3_heldout_corr_n": 79,
    "wrong3_heldout_corr_mean_change": 0.12736105617088608,
    "wrong3_heldout_corr_improved": 0,
    "wrong3_heldout_retrieval_n": 79,
    "wrong3_heldout_retrieval_mean_change": -0.00026000262815741043,
    "wrong3_heldout_retrieval_improved": 14,
    "random3_fit_objective_n": 101,
    "random3_fit_objective_mean_change": 0.000864852773080958,
    "random3_fit_objective_improved": 30,
    "random3_heldout_corr_n": 79,
    "random3_heldout_corr_mean_change": 0.0007291624817667128,
    "random3_heldout_corr_improved": 24,
    "random3_heldout_retrieval_n": 79,
    "random3_heldout_retrieval_mean_change": -1.8070015726210196e-06,
    "random3_heldout_retrieval_improved": 10,
    "free3_fit_objective_n": 101,
    "free3_fit_objective_mean_change": -0.6424612585860904,
    "free3_fit_objective_improved": 101,
    "free3_heldout_corr_n": 79,
    "free3_heldout_corr_mean_change": -0.460589276084417,
    "free3_heldout_corr_improved": 79,
    "free3_heldout_retrieval_n": 79,
    "free3_heldout_retrieval_mean_change": 0.12028552931320818,
    "free3_heldout_retrieval_improved": 79
  },
  {
    "cohort": "DEV",
    "source": "quad",
    "n_nonalias": 246,
    "active_AB": 49,
    "mean_pairs": 151.21138211382114,
    "mean_actual_correct_wrong_first_step_cosine": -0.7290941291170573,
    "corr1_fit_objective_n": 49,
    "corr1_fit_objective_mean_change": -0.039733964569714605,
    "corr1_fit_objective_improved": 49,
    "corr1_heldout_corr_n": 27,
    "corr1_heldout_corr_mean_change": -0.03720165182043005,
    "corr1_heldout_corr_improved": 27,
    "corr1_heldout_retrieval_n": 27,
    "corr1_heldout_retrieval_mean_change": 3.661601631729691e-05,
    "corr1_heldout_retrieval_improved": 7,
    "corr3_fit_objective_n": 49,
    "corr3_fit_objective_mean_change": -0.11582988135668697,
    "corr3_fit_objective_improved": 49,
    "corr3_heldout_corr_n": 27,
    "corr3_heldout_corr_mean_change": -0.10954145148948387,
    "corr3_heldout_corr_improved": 27,
    "corr3_heldout_retrieval_n": 27,
    "corr3_heldout_retrieval_mean_change": 0.0006330609321594238,
    "corr3_heldout_retrieval_improved": 13,
    "wrong3_fit_objective_n": 49,
    "wrong3_fit_objective_mean_change": 0.09452829069020796,
    "wrong3_fit_objective_improved": 3,
    "wrong3_heldout_corr_n": 27,
    "wrong3_heldout_corr_mean_change": 0.07909197277492946,
    "wrong3_heldout_corr_improved": 2,
    "wrong3_heldout_retrieval_n": 27,
    "wrong3_heldout_retrieval_mean_change": 0.00038593234839262786,
    "wrong3_heldout_retrieval_improved": 7,
    "random3_fit_objective_n": 49,
    "random3_fit_objective_mean_change": 0.0003378585893280652,
    "random3_fit_objective_improved": 21,
    "random3_heldout_corr_n": 27,
    "random3_heldout_corr_mean_change": 0.000507460700141059,
    "random3_heldout_corr_improved": 8,
    "random3_heldout_retrieval_n": 27,
    "random3_heldout_retrieval_mean_change": -7.776584890153672e-05,
    "random3_heldout_retrieval_improved": 1,
    "free3_fit_objective_n": 49,
    "free3_fit_objective_mean_change": -0.5478303335150894,
    "free3_fit_objective_improved": 49,
    "free3_heldout_corr_n": 27,
    "free3_heldout_corr_mean_change": -0.3883963779166893,
    "free3_heldout_corr_improved": 27,
    "free3_heldout_retrieval_n": 27,
    "free3_heldout_retrieval_mean_change": 0.10200459206545795,
    "free3_heldout_retrieval_improved": 27
  },
  {
    "cohort": "EXP",
    "source": "waymo",
    "n_nonalias": 457,
    "active_AB": 211,
    "mean_pairs": 316.2691466083151,
    "mean_actual_correct_wrong_first_step_cosine": -0.7142671841778119,
    "corr1_fit_objective_n": 211,
    "corr1_fit_objective_mean_change": -0.05187198788068871,
    "corr1_fit_objective_improved": 211,
    "corr1_heldout_corr_n": 135,
    "corr1_heldout_corr_mean_change": -0.05085290802849664,
    "corr1_heldout_corr_improved": 135,
    "corr1_heldout_retrieval_n": 135,
    "corr1_heldout_retrieval_mean_change": 0.0006399359967973497,
    "corr1_heldout_retrieval_improved": 54,
    "corr3_fit_objective_n": 211,
    "corr3_fit_objective_mean_change": -0.15334476678857306,
    "corr3_fit_objective_improved": 211,
    "corr3_heldout_corr_n": 135,
    "corr3_heldout_corr_mean_change": -0.15086986400462962,
    "corr3_heldout_corr_improved": 135,
    "corr3_heldout_retrieval_n": 135,
    "corr3_heldout_retrieval_mean_change": 0.0018672192538226092,
    "corr3_heldout_retrieval_improved": 83,
    "wrong3_fit_objective_n": 211,
    "wrong3_fit_objective_mean_change": 0.12506799336293298,
    "wrong3_fit_objective_improved": 0,
    "wrong3_heldout_corr_n": 135,
    "wrong3_heldout_corr_mean_change": 0.12454188311541523,
    "wrong3_heldout_corr_improved": 0,
    "wrong3_heldout_retrieval_n": 135,
    "wrong3_heldout_retrieval_mean_change": 0.001254976678777624,
    "wrong3_heldout_retrieval_improved": 87,
    "random3_fit_objective_n": 211,
    "random3_fit_objective_mean_change": 0.0012576455753561444,
    "random3_fit_objective_improved": 58,
    "random3_heldout_corr_n": 135,
    "random3_heldout_corr_mean_change": 0.0008591369346336082,
    "random3_heldout_corr_improved": 51,
    "random3_heldout_retrieval_n": 135,
    "random3_heldout_retrieval_mean_change": 9.266513365286367e-05,
    "random3_heldout_retrieval_improved": 20,
    "free3_fit_objective_n": 211,
    "free3_fit_objective_mean_change": -0.6628617025664633,
    "free3_fit_objective_improved": 211,
    "free3_heldout_corr_n": 135,
    "free3_heldout_corr_mean_change": -0.43439718352423773,
    "free3_heldout_corr_improved": 135,
    "free3_heldout_retrieval_n": 135,
    "free3_heldout_retrieval_mean_change": 0.09817407396104601,
    "free3_heldout_retrieval_improved": 135
  },
  {
    "cohort": "EXP",
    "source": "drone",
    "n_nonalias": 483,
    "active_AB": 200,
    "mean_pairs": 293.7577639751553,
    "mean_actual_correct_wrong_first_step_cosine": -0.9057217709288244,
    "corr1_fit_objective_n": 200,
    "corr1_fit_objective_mean_change": -0.0430649471282959,
    "corr1_fit_objective_improved": 200,
    "corr1_heldout_corr_n": 149,
    "corr1_heldout_corr_mean_change": -0.04339193337715712,
    "corr1_heldout_corr_improved": 149,
    "corr1_heldout_retrieval_n": 149,
    "corr1_heldout_retrieval_mean_change": 0.00010886788368225098,
    "corr1_heldout_retrieval_improved": 36,
    "corr3_fit_objective_n": 200,
    "corr3_fit_objective_mean_change": -0.12374044060707093,
    "corr3_fit_objective_improved": 200,
    "corr3_heldout_corr_n": 149,
    "corr3_heldout_corr_mean_change": -0.1266568739142194,
    "corr3_heldout_corr_improved": 149,
    "corr3_heldout_retrieval_n": 149,
    "corr3_heldout_retrieval_mean_change": 0.0003848722917121529,
    "corr3_heldout_retrieval_improved": 61,
    "wrong3_fit_objective_n": 200,
    "wrong3_fit_objective_mean_change": 0.13020254611968995,
    "wrong3_fit_objective_improved": 0,
    "wrong3_heldout_corr_n": 149,
    "wrong3_heldout_corr_mean_change": 0.13269985122168623,
    "wrong3_heldout_corr_improved": 0,
    "wrong3_heldout_retrieval_n": 149,
    "wrong3_heldout_retrieval_mean_change": -0.00033451026718088445,
    "wrong3_heldout_retrieval_improved": 34,
    "random3_fit_objective_n": 200,
    "random3_fit_objective_mean_change": 0.0007436525821685791,
    "random3_fit_objective_improved": 73,
    "random3_heldout_corr_n": 149,
    "random3_heldout_corr_mean_change": 0.0003552068799934131,
    "random3_heldout_corr_improved": 68,
    "random3_heldout_retrieval_n": 149,
    "random3_heldout_retrieval_mean_change": -1.0971854997161251e-05,
    "random3_heldout_retrieval_improved": 11,
    "free3_fit_objective_n": 200,
    "free3_fit_objective_mean_change": -0.643919118642807,
    "free3_fit_objective_improved": 200,
    "free3_heldout_corr_n": 149,
    "free3_heldout_corr_mean_change": -0.4564311216341569,
    "free3_heldout_corr_improved": 149,
    "free3_heldout_retrieval_n": 149,
    "free3_heldout_retrieval_mean_change": 0.12225087267040406,
    "free3_heldout_retrieval_improved": 149
  },
  {
    "cohort": "EXP",
    "source": "quad",
    "n_nonalias": 486,
    "active_AB": 117,
    "mean_pairs": 178.6008230452675,
    "mean_actual_correct_wrong_first_step_cosine": -0.7221021156194344,
    "corr1_fit_objective_n": 117,
    "corr1_fit_objective_mean_change": -0.03928342843667055,
    "corr1_fit_objective_improved": 117,
    "corr1_heldout_corr_n": 66,
    "corr1_heldout_corr_mean_change": -0.03649878863132361,
    "corr1_heldout_corr_improved": 66,
    "corr1_heldout_retrieval_n": 66,
    "corr1_heldout_retrieval_mean_change": 0.00010063905607570301,
    "corr1_heldout_retrieval_improved": 13,
    "corr3_fit_objective_n": 117,
    "corr3_fit_objective_mean_change": -0.11430515387119392,
    "corr3_fit_objective_improved": 117,
    "corr3_heldout_corr_n": 66,
    "corr3_heldout_corr_mean_change": -0.10740846576112689,
    "corr3_heldout_corr_improved": 66,
    "corr3_heldout_retrieval_n": 66,
    "corr3_heldout_retrieval_mean_change": 0.00044497757247000027,
    "corr3_heldout_retrieval_improved": 29,
    "wrong3_fit_objective_n": 117,
    "wrong3_fit_objective_mean_change": 0.09360660039461575,
    "wrong3_fit_objective_improved": 12,
    "wrong3_heldout_corr_n": 66,
    "wrong3_heldout_corr_mean_change": 0.09257604859092018,
    "wrong3_heldout_corr_improved": 3,
    "wrong3_heldout_retrieval_n": 66,
    "wrong3_heldout_retrieval_mean_change": -5.1520313277389064e-05,
    "wrong3_heldout_retrieval_improved": 16,
    "random3_fit_objective_n": 117,
    "random3_fit_objective_mean_change": 0.0004435400677542401,
    "random3_fit_objective_improved": 43,
    "random3_heldout_corr_n": 66,
    "random3_heldout_corr_mean_change": 9.899067156242602e-05,
    "random3_heldout_corr_improved": 32,
    "random3_heldout_retrieval_n": 66,
    "random3_heldout_retrieval_mean_change": 9.399358973358616e-05,
    "random3_heldout_retrieval_improved": 8,
    "free3_fit_objective_n": 117,
    "free3_fit_objective_mean_change": -0.5593536144647843,
    "free3_fit_objective_improved": 117,
    "free3_heldout_corr_n": 66,
    "free3_heldout_corr_mean_change": -0.3848896062735355,
    "free3_heldout_corr_improved": 66,
    "free3_heldout_retrieval_n": 66,
    "free3_heldout_retrieval_mean_change": 0.1089479641029329,
    "free3_heldout_retrieval_improved": 66
  }
]
```

## CONCENTRATION

```json
[
  {
    "cohort": "DEV",
    "method": "corr3_F",
    "control": "mean_context",
    "n": 768,
    "gain_pp": -0.11538090988860215,
    "better": 11,
    "worse": 10,
    "same": 747,
    "query_changes": 29,
    "largest_positive_single_input_pp": 0.04186852011328779,
    "largest_negative_single_input_pp": -0.07265321711525753
  },
  {
    "cohort": "DEV",
    "method": "corr3_F",
    "control": "headmean_context",
    "n": 768,
    "gain_pp": 0.0571668128049902,
    "better": 21,
    "worse": 15,
    "same": 732,
    "query_changes": 57,
    "largest_positive_single_input_pp": 0.08265650351809108,
    "largest_negative_single_input_pp": -0.0893579070700335
  },
  {
    "cohort": "DEV",
    "method": "corr3_F",
    "control": "robust_context",
    "n": 768,
    "gain_pp": -0.1493796418481824,
    "better": 15,
    "worse": 12,
    "same": 741,
    "query_changes": 38,
    "largest_positive_single_input_pp": 0.055367509359264044,
    "largest_negative_single_input_pp": -0.07381118895591267
  },
  {
    "cohort": "DEV",
    "method": "corr3_F",
    "control": "free3_F",
    "n": 768,
    "gain_pp": 0.03411960421656429,
    "better": 10,
    "worse": 8,
    "same": 750,
    "query_changes": 26,
    "largest_positive_single_input_pp": 0.054098254938435335,
    "largest_negative_single_input_pp": -0.04378231070238453
  },
  {
    "cohort": "DEV",
    "method": "corr3_F",
    "control": "wrong3_F",
    "n": 768,
    "gain_pp": -0.09330264314555844,
    "better": 14,
    "worse": 12,
    "same": 742,
    "query_changes": 37,
    "largest_positive_single_input_pp": 0.04186852011328779,
    "largest_negative_single_input_pp": -0.07265321711525753
  },
  {
    "cohort": "DEV",
    "method": "corr3_F",
    "control": "random3_F",
    "n": 768,
    "gain_pp": -0.11767191827915847,
    "better": 11,
    "worse": 10,
    "same": 747,
    "query_changes": 27,
    "largest_positive_single_input_pp": 0.04186852011328779,
    "largest_negative_single_input_pp": -0.07265321711525753
  },
  {
    "cohort": "EXP",
    "method": "corr3_F",
    "control": "mean_context",
    "n": 1536,
    "gain_pp": -0.08909071853915634,
    "better": 11,
    "worse": 10,
    "same": 1515,
    "query_changes": 42,
    "largest_positive_single_input_pp": 0.020871559304455223,
    "largest_negative_single_input_pp": -0.051036878261435326
  },
  {
    "cohort": "EXP",
    "method": "corr3_F",
    "control": "headmean_context",
    "n": 1536,
    "gain_pp": -0.12308053532766663,
    "better": 28,
    "worse": 24,
    "same": 1484,
    "query_changes": 103,
    "largest_positive_single_input_pp": 0.04503449860016842,
    "largest_negative_single_input_pp": -0.051036878261435326
  },
  {
    "cohort": "EXP",
    "method": "corr3_F",
    "control": "robust_context",
    "n": 1536,
    "gain_pp": -0.024763397636114598,
    "better": 16,
    "worse": 13,
    "same": 1507,
    "query_changes": 57,
    "largest_positive_single_input_pp": 0.03171037520579941,
    "largest_negative_single_input_pp": -0.04729911980131544
  },
  {
    "cohort": "EXP",
    "method": "corr3_F",
    "control": "free3_F",
    "n": 1536,
    "gain_pp": 0.013506419375361166,
    "better": 19,
    "worse": 14,
    "same": 1503,
    "query_changes": 62,
    "largest_positive_single_input_pp": 0.03171037520579941,
    "largest_negative_single_input_pp": -0.04729911980131544
  },
  {
    "cohort": "EXP",
    "method": "corr3_F",
    "control": "wrong3_F",
    "n": 1536,
    "gain_pp": -0.0030935496926485944,
    "better": 19,
    "worse": 19,
    "same": 1498,
    "query_changes": 68,
    "largest_positive_single_input_pp": 0.03457252457641046,
    "largest_negative_single_input_pp": -0.051036878261435326
  },
  {
    "cohort": "EXP",
    "method": "corr3_F",
    "control": "random3_F",
    "n": 1536,
    "gain_pp": -0.06364085280218643,
    "better": 11,
    "worse": 10,
    "same": 1515,
    "query_changes": 42,
    "largest_positive_single_input_pp": 0.02188165248061685,
    "largest_negative_single_input_pp": -0.051036878261435326
  }
]
```

## DECISION

```json
{
  "primary": "corr3_F",
  "cohorts": {
    "DEV": {
      "mIoU_percent": 17.233620747365077,
      "gains_pp": {
        "mean_context": -0.11538090988860716,
        "headmean_context": 0.05716681280499358,
        "robust_context": -0.14937964184818497,
        "free3_F": 0.03411960421656346,
        "wrong3_F": -0.09330264314556302,
        "random3_F": -0.11767191827916079
      },
      "poststratified_mIoU_percent": 12.362620571199011,
      "poststratified_gains_pp": {
        "mean_context": -0.3108114439951848,
        "headmean_context": 0.035914612770307386,
        "robust_context": -0.2837195841564686,
        "free3_F": -0.09951130448416379,
        "wrong3_F": -0.41605775610395845,
        "random3_F": -0.3091709824084923
      },
      "passes_all_prelocked_controls": false,
      "unused_C_available_macro_gain_pp": -0.7048062970253898,
      "unused_C_gate": false
    },
    "EXP": {
      "mIoU_percent": 16.410475156647877,
      "gains_pp": {
        "mean_context": -0.08909071853915762,
        "headmean_context": -0.12308053532767005,
        "robust_context": -0.0247633976361179,
        "free3_F": 0.013506419375360679,
        "wrong3_F": -0.0030935496926467465,
        "random3_F": -0.06364085280218745
      },
      "poststratified_mIoU_percent": 12.847571414465035,
      "poststratified_gains_pp": {
        "mean_context": -0.10413146094535009,
        "headmean_context": -0.10358683646999722,
        "robust_context": 0.0746856057133698,
        "free3_F": 0.005206994726183467,
        "wrong3_F": -0.12456846432316127,
        "random3_F": -0.03352409471290585
      },
      "passes_all_prelocked_controls": false,
      "unused_C_available_macro_gain_pp": 0.07668917052104969,
      "unused_C_gate": true
    }
  },
  "FINAL_access": false,
  "posthoc_primary_switch": false,
  "development_gate_pass": false,
  "TTA_robustly_adopted": false
}
```

## NATIVE_AUDIT

```json
{
  "state": "PASS_ALL_POINT_CORRESPONDENCE_SGD_FREE_FEATURE_FIELDS_SCOPE_NORMS_AND_NATIVE_READOUTS",
  "n": 2304,
  "aliases": 154,
  "CPU_heads": 30100,
  "updated_states": 15050,
  "active_AB_inputs": 786,
  "max_probability_error": 1.9517341141028766e-06,
  "max_state_element_error": 1.2516975402832031e-06,
  "max_loss_error": 9.5367431640625e-07,
  "max_actual_norm_error": 2.220446049250313e-16,
  "near_ties": 0,
  "GT_used": false,
  "FINAL_access": false
}
```

## SCALAR_AUDIT

```json
{
  "state": "PASS_ALL_TASK_PLATFORM_SCENE_PAIRS_POSTSTRATIFICATION",
  "n": 2304,
  "arms": 25,
  "task_values": 57600,
  "new_task_values": 32256,
  "reused_task_values": 25344,
  "aliases": 154,
  "FINAL_access": false
}
```

## GEOMETRY_AUDIT

```json
{
  "n": 2304,
  "new_readouts": 32256,
  "reused_readouts": 25344,
  "geometry_max_error": 6.661338147750939e-16,
  "serialization_recovery_only": true
}
```

## 执行裁决

预设开发门未通过，停止这一固定注册点InfoNCE短步Adapter配方。不将一步、某个平台或某个场景的正次项事后升级主方法，不按本轮结果继续扫温度、半径、步数或层位置。

区域对应检索改善、辅助loss下降和实际参数非零都不足以替代选框收益。只有正确对应在总体/场景频数控制下超过同信息直接使用、错误/随机，并在未拟合C仍有收益，才能支持这一参数适应机制。所有good/failure均指相对IoU变化，未访问私有图像认证的实例身份保留UNKNOWN。全量匿名逐例和场景负例公开，不删除回退、缺B/C或没有负区域的样本。

本轮每条seconds是多臂共享缓存和多次后缀研究开销，不能当单方法全新端到端延迟。原始观测、描述、坐标、query索引、特征、权重和梯度不公开。
