# R122 — 注册多观测的稳健seed表示聚合

本轮是新的观测证据准入实验，不是网络参数TTA，也没有替换生产方法。保留三个epoch100/R91双向结构、原A256与全部六层decoder位置、当前完整文字/RGB/标定。R121的GN物理恢复家族已停止，本轮没有换一种恢复损失、更新层或source训练。

## 为什么测试这一个改变

已有配准为seed给出二值有效性，通过后对A/B/C特征等权平均。对应失败与邻物污染可能使均值受单份偏离特征影响，但这只是待检验解释。这里将每个三份有效观测的288D pre-language seed特征，改成欧氏几何中位数；两个向量时中位数不唯一，固定取原均值，单向量保持A。没有把高一致性直接认证为正确实例。

最多三向量处于一个三角平面，可用Fermat点的重心坐标或120度顶点条件求精确解；重复向量单独处理。因此不是靠选Weiszfeld步数、放大更新或降低某个恢复loss来寻找涨分。最初缓存资格检查的4次Weiszfeld仅用于确认三份有效特征及非均匀权重的机会，尚未调用模型或任务GT；正式模型前改成精确解，原资格代码与hash保留。独立凸KKT检查用于验证真实向量上的最优性。

RFA原论文/官方实现及三角形公式的适用边界见LITERATURE.md。借鉴稳健聚合统计机制，不是联邦学习复现或grounding有效性保证：如果两份相关误差压过一份真实新信息，中位数也可能更差。

## 固定读出与必要对照

原生点—文字路径接受新的聚合特征并重算h/U；RGB late仍取sourceA，竞争残差仍取原sourceMeanF。它沿用R118已暴露的次项读出基座，不将其改写成已采用方法。保持这两个接口固定，是为了仅测试进入原生grounding的点观测聚合差异；native-only读出另列，避免把接口影响混成聚合收益。

比较三类权重：空间自适应中位数权重、在三份有效seed间打乱同一权重集合、当前输入各view平均权重。后两项控制空间对应和全局view偏置。另将B/C向量各自在三份有效seed间错配，保留支持和向量集合，比较错配中位数/均值。所有条件沿用相同A/B/C，无新增传感器或PointNet前向。

额外强对照headmean在相同canonical空间锚点上平均A/B/C原始matching投影，文字token表示平均后归一化，再给出native和相同固定RGB上下文读出。这不是历史R59的软query传输算法。R59原DEV336的22.611116%历史结果继续保留于原study；这里的三source/注册观测/清单不同，不能伪造同条件R59数值。headmean在任何任务评分前单独锁定、预测封存和独立审计，不改变原主项或削弱准入要求。

## 数据与完整性

DEV768/EXP1536已历史曝光，不能称独立确认；分别包含44/110保护别名，先检查再加载任何特征，全部在分母。528/2304定向输入有A/B/C三份同时有效seed，其余按原均值；不是筛选有利案例。不能把两个incoming source的同一物理观测当独立新场景。主指标各方向等权平均，另按全量既有场景频数后分层；EXP缺少4个Waymo场景，明确仅共同支持范围。

24条工程输入覆盖三份有效和不足三份两种情形，原均值完整后缀重放h/U与旧缓存逐值一致。全部2304原始预测及附加强直接读出先封存，再做CPU读出/KKT/权重/错配/geometry审计、评分。所有负例、无效观测与UNKNOWN保留。

## 总体结果

| 数据 | 方法 | mIoU% | Acc25% | Acc50% |
|---|---|---:|---:|---:|
| DEV | original_source | 16.811383 | 30.989583 | 14.973958 |
| DEV | prepared_A | 16.228652 | 29.947917 | 12.630208 |
| DEV | prepared_F | 16.703230 | 30.989583 | 13.151042 |
| DEV | native_F | 16.974318 | 31.250000 | 14.843750 |
| DEV | repaired_F | 17.265658 | 31.901042 | 14.322917 |
| DEV | lateOnly_F | 17.180150 | 31.901042 | 14.453125 |
| DEV | mean_context | 17.349002 | 31.901042 | 14.453125 |
| DEV | native_ABC | 16.734016 | 30.729167 | 14.843750 |
| DEV | context_ABC | 17.077734 | 31.380208 | 14.453125 |
| DEV | context_noR_ABC | 17.033323 | 31.770833 | 14.453125 |
| DEV | robust_native | 17.175226 | 31.640625 | 15.104167 |
| DEV | robust_context | 17.383000 | 32.161458 | 14.583333 |
| DEV | weight_shuffle_native | 16.731426 | 30.989583 | 14.713542 |
| DEV | weight_shuffle_context | 16.957835 | 31.380208 | 14.062500 |
| DEV | weight_mean_native | 16.977449 | 31.250000 | 14.843750 |
| DEV | weight_mean_context | 17.385491 | 32.031250 | 14.453125 |
| DEV | wrong_robust_native | 14.296423 | 26.302083 | 12.239583 |
| DEV | wrong_robust_context | 14.452836 | 26.692708 | 11.848958 |
| DEV | wrong_mean_native | 15.076561 | 27.604167 | 13.281250 |
| DEV | wrong_mean_context | 14.959436 | 27.213542 | 12.630208 |
| DEV | headmean_native | 16.516213 | 30.338542 | 14.453125 |
| DEV | headmean_context | 17.176454 | 31.640625 | 14.583333 |
| EXP | original_source | 16.144619 | 30.598958 | 12.434896 |
| EXP | prepared_A | 16.251067 | 30.598958 | 11.783854 |
| EXP | prepared_F | 16.238436 | 30.598958 | 12.044271 |
| EXP | native_F | 16.237263 | 30.859375 | 12.434896 |
| EXP | repaired_F | 16.314985 | 30.924479 | 12.109375 |
| EXP | lateOnly_F | 16.439770 | 31.184896 | 12.174479 |
| EXP | mean_context | 16.499566 | 31.119792 | 12.174479 |
| EXP | native_ABC | 16.321800 | 31.250000 | 12.434896 |
| EXP | context_ABC | 16.280616 | 30.924479 | 11.914062 |
| EXP | context_noR_ABC | 16.359192 | 30.989583 | 12.174479 |
| EXP | robust_native | 16.283769 | 31.184896 | 12.434896 |
| EXP | robust_context | 16.435239 | 31.184896 | 12.239583 |
| EXP | weight_shuffle_native | 16.383383 | 31.380208 | 12.434896 |
| EXP | weight_shuffle_context | 16.452075 | 31.315104 | 12.044271 |
| EXP | weight_mean_native | 16.188953 | 30.664062 | 12.369792 |
| EXP | weight_mean_context | 16.434424 | 30.989583 | 12.109375 |
| EXP | wrong_robust_native | 14.034856 | 27.083333 | 9.895833 |
| EXP | wrong_robust_context | 14.067076 | 26.888021 | 9.765625 |
| EXP | wrong_mean_native | 14.702186 | 28.255208 | 10.807292 |
| EXP | wrong_mean_context | 14.768151 | 28.059896 | 10.742188 |
| EXP | headmean_native | 16.222712 | 30.989583 | 12.434896 |
| EXP | headmean_context | 16.533556 | 31.315104 | 12.369792 |

## 平台结果

| 数据 | 目标 | 方法 | mIoU% | Acc25% | Acc50% |
|---|---|---|---:|---:|---:|
| DEV | waymo | original_source | 11.473631 | 23.437500 | 7.421875 |
| DEV | waymo | prepared_F | 12.569007 | 26.171875 | 5.078125 |
| DEV | waymo | native_F | 11.535583 | 23.437500 | 7.421875 |
| DEV | waymo | lateOnly_F | 12.181059 | 25.390625 | 7.421875 |
| DEV | waymo | mean_context | 12.247576 | 24.609375 | 7.812500 |
| DEV | waymo | native_ABC | 11.215742 | 22.656250 | 7.421875 |
| DEV | waymo | robust_context | 12.278621 | 24.609375 | 7.812500 |
| DEV | waymo | weight_shuffle_context | 12.242181 | 24.609375 | 7.812500 |
| DEV | waymo | weight_mean_context | 12.247576 | 24.609375 | 7.812500 |
| DEV | waymo | wrong_robust_context | 11.771195 | 23.828125 | 7.812500 |
| DEV | waymo | headmean_context | 12.097959 | 24.218750 | 7.812500 |
| DEV | drone | original_source | 13.867693 | 24.609375 | 12.500000 |
| DEV | drone | prepared_F | 14.029708 | 25.000000 | 11.718750 |
| DEV | drone | native_F | 14.710633 | 26.171875 | 12.890625 |
| DEV | drone | lateOnly_F | 14.345223 | 25.781250 | 11.718750 |
| DEV | drone | mean_context | 14.805341 | 26.953125 | 11.718750 |
| DEV | drone | native_ABC | 14.188880 | 24.609375 | 12.500000 |
| DEV | drone | robust_context | 14.770480 | 26.953125 | 11.718750 |
| DEV | drone | weight_shuffle_context | 14.533285 | 26.171875 | 11.718750 |
| DEV | drone | weight_mean_context | 14.805341 | 26.953125 | 11.718750 |
| DEV | drone | wrong_robust_context | 14.607291 | 26.171875 | 11.718750 |
| DEV | drone | headmean_context | 14.514873 | 26.171875 | 11.718750 |
| DEV | quad | original_source | 25.092825 | 44.921875 | 25.000000 |
| DEV | quad | prepared_F | 23.510976 | 41.796875 | 22.656250 |
| DEV | quad | native_F | 24.676737 | 44.140625 | 24.218750 |
| DEV | quad | lateOnly_F | 25.014169 | 44.531250 | 24.218750 |
| DEV | quad | mean_context | 24.994087 | 44.140625 | 23.828125 |
| DEV | quad | native_ABC | 24.797425 | 44.921875 | 24.609375 |
| DEV | quad | robust_context | 25.099900 | 44.921875 | 24.218750 |
| DEV | quad | weight_shuffle_context | 24.098040 | 43.359375 | 22.656250 |
| DEV | quad | weight_mean_context | 25.103555 | 44.531250 | 23.828125 |
| DEV | quad | wrong_robust_context | 16.980023 | 30.078125 | 16.015625 |
| DEV | quad | headmean_context | 24.916529 | 44.531250 | 24.218750 |
| EXP | waymo | original_source | 13.450978 | 28.906250 | 6.835938 |
| EXP | waymo | prepared_F | 13.107853 | 27.734375 | 5.859375 |
| EXP | waymo | native_F | 13.085533 | 27.929688 | 6.445312 |
| EXP | waymo | lateOnly_F | 12.839589 | 27.539062 | 6.054688 |
| EXP | waymo | mean_context | 12.855988 | 27.343750 | 6.054688 |
| EXP | waymo | native_ABC | 13.399290 | 28.710938 | 6.640625 |
| EXP | waymo | robust_context | 12.629787 | 26.953125 | 5.859375 |
| EXP | waymo | weight_shuffle_context | 12.846817 | 27.539062 | 5.859375 |
| EXP | waymo | weight_mean_context | 12.639387 | 26.953125 | 5.859375 |
| EXP | waymo | wrong_robust_context | 12.281524 | 26.171875 | 5.468750 |
| EXP | waymo | headmean_context | 12.988950 | 27.734375 | 6.054688 |
| EXP | drone | original_source | 11.525665 | 20.898438 | 9.375000 |
| EXP | drone | prepared_F | 11.874628 | 20.312500 | 9.375000 |
| EXP | drone | native_F | 12.067931 | 22.265625 | 9.765625 |
| EXP | drone | lateOnly_F | 12.383083 | 22.656250 | 9.570312 |
| EXP | drone | mean_context | 12.279426 | 22.265625 | 9.570312 |
| EXP | drone | native_ABC | 11.933850 | 22.265625 | 9.570312 |
| EXP | drone | robust_context | 12.288861 | 22.265625 | 9.570312 |
| EXP | drone | weight_shuffle_context | 12.194211 | 22.070312 | 9.570312 |
| EXP | drone | weight_mean_context | 12.324685 | 22.265625 | 9.570312 |
| EXP | drone | wrong_robust_context | 11.926337 | 21.484375 | 9.375000 |
| EXP | drone | headmean_context | 12.240413 | 22.070312 | 9.765625 |
| EXP | quad | original_source | 23.457215 | 41.992188 | 21.093750 |
| EXP | quad | prepared_F | 23.732826 | 43.750000 | 20.898438 |
| EXP | quad | native_F | 23.558326 | 42.382812 | 21.093750 |
| EXP | quad | lateOnly_F | 24.096638 | 43.359375 | 20.898438 |
| EXP | quad | mean_context | 24.363284 | 43.750000 | 20.898438 |
| EXP | quad | native_ABC | 23.632260 | 42.773438 | 21.093750 |
| EXP | quad | robust_context | 24.387068 | 44.335938 | 21.289062 |
| EXP | quad | weight_shuffle_context | 24.315196 | 44.335938 | 20.703125 |
| EXP | quad | weight_mean_context | 24.339202 | 43.750000 | 20.898438 |
| EXP | quad | wrong_robust_context | 17.993368 | 33.007812 | 14.453125 |
| EXP | quad | headmean_context | 24.371304 | 44.140625 | 21.289062 |

## 场景频数控制

| 数据 | 方法 | 完整场景最小覆盖 | 后分层mIoU% |
|---|---|---:|---:|
| DEV | original_source | 1.000000 | 12.186315 |
| DEV | prepared_A | 1.000000 | 11.764418 |
| DEV | prepared_F | 1.000000 | 11.980683 |
| DEV | native_F | 1.000000 | 12.237360 |
| DEV | repaired_F | 1.000000 | 12.590408 |
| DEV | lateOnly_F | 1.000000 | 12.479188 |
| DEV | mean_context | 1.000000 | 12.673432 |
| DEV | native_ABC | 1.000000 | 11.834713 |
| DEV | context_ABC | 1.000000 | 12.379495 |
| DEV | context_noR_ABC | 1.000000 | 12.343518 |
| DEV | robust_native | 1.000000 | 12.397200 |
| DEV | robust_context | 1.000000 | 12.646340 |
| DEV | weight_shuffle_native | 1.000000 | 11.893913 |
| DEV | weight_shuffle_context | 1.000000 | 12.334013 |
| DEV | weight_mean_native | 1.000000 | 12.243217 |
| DEV | weight_mean_context | 1.000000 | 12.698587 |
| DEV | wrong_robust_native | 1.000000 | 10.399732 |
| DEV | wrong_robust_context | 1.000000 | 10.927399 |
| DEV | wrong_mean_native | 1.000000 | 10.467619 |
| DEV | wrong_mean_context | 1.000000 | 10.823693 |
| DEV | headmean_native | 1.000000 | 11.676996 |
| DEV | headmean_context | 1.000000 | 12.326706 |
| EXP | original_source | 0.998149 | 12.339915 |
| EXP | prepared_A | 0.998149 | 13.127455 |
| EXP | prepared_F | 0.998149 | 12.899456 |
| EXP | native_F | 0.998149 | 12.670824 |
| EXP | repaired_F | 0.998149 | 12.706617 |
| EXP | lateOnly_F | 0.998149 | 12.880574 |
| EXP | mean_context | 0.998149 | 12.951703 |
| EXP | native_ABC | 0.998149 | 12.817131 |
| EXP | context_ABC | 0.998149 | 12.838787 |
| EXP | context_noR_ABC | 0.998149 | 12.909649 |
| EXP | robust_native | 0.998149 | 12.691617 |
| EXP | robust_context | 0.998149 | 12.772886 |
| EXP | weight_shuffle_native | 0.998149 | 12.821686 |
| EXP | weight_shuffle_context | 0.998149 | 12.964488 |
| EXP | weight_mean_native | 0.998149 | 12.626428 |
| EXP | weight_mean_context | 0.998149 | 12.817846 |
| EXP | wrong_robust_native | 0.998149 | 11.340605 |
| EXP | wrong_robust_context | 0.998149 | 11.321427 |
| EXP | wrong_mean_native | 0.998149 | 11.613230 |
| EXP | wrong_mean_context | 0.998149 | 11.539866 |
| EXP | headmean_native | 0.998149 | 12.701281 |
| EXP | headmean_context | 0.998149 | 12.951158 |

## 预设裁决

```json
{
  "primary": "robust_context",
  "cohorts": {
    "DEV": {
      "mIoU_percent": 17.38300038921326,
      "gains_pp": {
        "mean_context": 0.03399873195957781,
        "weight_shuffle_context": 0.42516524472813444,
        "weight_mean_context": -0.0024906086885678747,
        "wrong_robust_context": 2.930163906699965,
        "wrong_mean_context": 2.4235642574044327,
        "native_ABC": 0.6489846945504851,
        "lateOnly_F": 0.2028500914238549,
        "headmean_native": 0.8667872964356178,
        "headmean_context": 0.20654645465317856
      },
      "poststratified_mIoU_percent": 12.64634015535548,
      "poststratified_gains_pp": {
        "mean_context": -0.027091859838714427,
        "weight_shuffle_context": 0.3123266974058865,
        "weight_mean_context": -0.05224668249705289,
        "wrong_robust_context": 1.7189407849740999,
        "wrong_mean_context": 1.8226475641687045,
        "native_ABC": 0.8116268761142749,
        "lateOnly_F": 0.167151839421404,
        "headmean_native": 0.9693440569310923,
        "headmean_context": 0.31963419692677597
      },
      "passes_all_prelocked_controls": false
    },
    "EXP": {
      "mIoU_percent": 16.435238554283995,
      "gains_pp": {
        "mean_context": -0.06432732090303972,
        "weight_shuffle_context": -0.016836116632386933,
        "weight_mean_context": 0.0008140563362601938,
        "wrong_robust_context": 2.3681621751536266,
        "wrong_mean_context": 1.667087936687663,
        "native_ABC": 0.11343834191607982,
        "lateOnly_F": -0.004531370019186909,
        "headmean_native": 0.21252632528573656,
        "headmean_context": -0.09831713769155215
      },
      "poststratified_mIoU_percent": 12.772885808751665,
      "poststratified_gains_pp": {
        "mean_context": -0.17881706665871988,
        "weight_shuffle_context": -0.19160235167865558,
        "weight_mean_context": -0.044960688893436185,
        "wrong_robust_context": 1.451458593315511,
        "wrong_mean_context": 1.233019544208977,
        "native_ABC": -0.044245602124123096,
        "lateOnly_F": -0.1076878431535313,
        "headmean_native": 0.07160497637215002,
        "headmean_context": -0.17827244218336702
      },
      "passes_all_prelocked_controls": false
    }
  },
  "FINAL_access": false,
  "posthoc_primary_switch": false,
  "development_gate_pass": false,
  "TTA_robustly_adopted": false
}
```

只有主项在两个已暴露范围的总体及场景后分层均超过均值、空间打乱/全局权重/错配/ABC/lateOnly和headmean强对照，才准入后续匹配参数适应。通过也不自动声称TTA有效；未通过不继续扫描中位数强度或补损失。native-only或某平台的有利次项不事后升级主项。

## 组件与正负案例

```json
[
  {
    "cohort": "DEV",
    "source": "waymo",
    "n": 234,
    "inputs_with_triple": 72,
    "triple_seeds_total": 44986,
    "mean_geometric_objective_before": 2.3399253288768866,
    "mean_geometric_objective_after": 2.2664316764816155
  },
  {
    "cohort": "DEV",
    "source": "drone",
    "n": 244,
    "inputs_with_triple": 79,
    "triple_seeds_total": 50689,
    "mean_geometric_objective_before": 2.1873060216262683,
    "mean_geometric_objective_after": 2.1076609690054604
  },
  {
    "cohort": "DEV",
    "source": "quad",
    "n": 246,
    "inputs_with_triple": 27,
    "triple_seeds_total": 18171,
    "mean_geometric_objective_before": 2.3143111136430936,
    "mean_geometric_objective_after": 2.2060463197532716
  },
  {
    "cohort": "EXP",
    "source": "waymo",
    "n": 457,
    "inputs_with_triple": 135,
    "triple_seeds_total": 81151,
    "mean_geometric_objective_before": 2.3506377918110006,
    "mean_geometric_objective_after": 2.2771927858729617
  },
  {
    "cohort": "EXP",
    "source": "drone",
    "n": 483,
    "inputs_with_triple": 149,
    "triple_seeds_total": 94813,
    "mean_geometric_objective_before": 2.142920114991539,
    "mean_geometric_objective_after": 2.0646848457796896
  },
  {
    "cohort": "EXP",
    "source": "quad",
    "n": 486,
    "inputs_with_triple": 66,
    "triple_seeds_total": 44076,
    "mean_geometric_objective_before": 2.294295570839323,
    "mean_geometric_objective_after": 2.1954627643004274
  }
]
[
  {
    "cohort": "DEV",
    "control": "mean_context",
    "n": 768,
    "gain_pp": 0.0339987319595802,
    "better": 6,
    "worse": 7,
    "same": 755,
    "query_changes": 18,
    "largest_positive_single_input_pp": 0.07381118895591267,
    "largest_negative_single_input_pp": -0.04164013147968911
  },
  {
    "cohort": "DEV",
    "control": "headmean_context",
    "n": 768,
    "gain_pp": 0.20654645465317265,
    "better": 18,
    "worse": 11,
    "same": 739,
    "query_changes": 50,
    "largest_positive_single_input_pp": 0.08265650351809108,
    "largest_negative_single_input_pp": -0.0893579070700335
  },
  {
    "cohort": "DEV",
    "control": "headmean_native",
    "n": 768,
    "gain_pp": 0.8667872964356201,
    "better": 61,
    "worse": 40,
    "same": 667,
    "query_changes": 184,
    "largest_positive_single_input_pp": 0.08699899059864145,
    "largest_negative_single_input_pp": -0.11153539380202347
  },
  {
    "cohort": "DEV",
    "control": "weight_shuffle_context",
    "n": 768,
    "gain_pp": 0.4251652447281343,
    "better": 10,
    "worse": 7,
    "same": 751,
    "query_changes": 31,
    "largest_positive_single_input_pp": 0.08057482712365807,
    "largest_negative_single_input_pp": -0.01413549760339333
  },
  {
    "cohort": "DEV",
    "control": "weight_mean_context",
    "n": 768,
    "gain_pp": -0.002490608688564989,
    "better": 4,
    "worse": 8,
    "same": 756,
    "query_changes": 17,
    "largest_positive_single_input_pp": 0.07381118895591267,
    "largest_negative_single_input_pp": -0.04164013147968911
  },
  {
    "cohort": "DEV",
    "control": "wrong_robust_context",
    "n": 768,
    "gain_pp": 2.9301639066999647,
    "better": 75,
    "worse": 17,
    "same": 676,
    "query_changes": 127,
    "largest_positive_single_input_pp": 0.11620391437363997,
    "largest_negative_single_input_pp": -0.08073022972152126
  },
  {
    "cohort": "DEV",
    "control": "wrong_mean_context",
    "n": 768,
    "gain_pp": 2.4235642574044323,
    "better": 65,
    "worse": 15,
    "same": 688,
    "query_changes": 117,
    "largest_positive_single_input_pp": 0.088787215861668,
    "largest_negative_single_input_pp": -0.08420280778933663
  },
  {
    "cohort": "DEV",
    "control": "native_ABC",
    "n": 768,
    "gain_pp": 0.648984694550487,
    "better": 60,
    "worse": 45,
    "same": 663,
    "query_changes": 193,
    "largest_positive_single_input_pp": 0.08699899059864145,
    "largest_negative_single_input_pp": -0.11153539380202347
  },
  {
    "cohort": "DEV",
    "control": "lateOnly_F",
    "n": 768,
    "gain_pp": 0.20285009142385207,
    "better": 16,
    "worse": 19,
    "same": 733,
    "query_changes": 62,
    "largest_positive_single_input_pp": 0.08699899059864145,
    "largest_negative_single_input_pp": -0.04164013147968911
  },
  {
    "cohort": "EXP",
    "control": "mean_context",
    "n": 1536,
    "gain_pp": -0.06432732090304172,
    "better": 12,
    "worse": 16,
    "same": 1508,
    "query_changes": 42,
    "largest_positive_single_input_pp": 0.038206062815087545,
    "largest_negative_single_input_pp": -0.051036878261435326
  },
  {
    "cohort": "EXP",
    "control": "headmean_context",
    "n": 1536,
    "gain_pp": -0.09831713769155204,
    "better": 27,
    "worse": 29,
    "same": 1480,
    "query_changes": 103,
    "largest_positive_single_input_pp": 0.04503449860016842,
    "largest_negative_single_input_pp": -0.051036878261435326
  },
  {
    "cohort": "EXP",
    "control": "headmean_native",
    "n": 1536,
    "gain_pp": 0.21252632528573653,
    "better": 107,
    "worse": 90,
    "same": 1339,
    "query_changes": 387,
    "largest_positive_single_input_pp": 0.05015647696241018,
    "largest_negative_single_input_pp": -0.051036878261435326
  },
  {
    "cohort": "EXP",
    "control": "weight_shuffle_context",
    "n": 1536,
    "gain_pp": -0.01683611663238991,
    "better": 19,
    "worse": 21,
    "same": 1496,
    "query_changes": 64,
    "largest_positive_single_input_pp": 0.038206062815087545,
    "largest_negative_single_input_pp": -0.03826176004744391
  },
  {
    "cohort": "EXP",
    "control": "weight_mean_context",
    "n": 1536,
    "gain_pp": 0.0008140563362548609,
    "better": 11,
    "worse": 14,
    "same": 1511,
    "query_changes": 39,
    "largest_positive_single_input_pp": 0.038206062815087545,
    "largest_negative_single_input_pp": -0.03803720941588019
  },
  {
    "cohort": "EXP",
    "control": "wrong_robust_context",
    "n": 1536,
    "gain_pp": 2.368162175153625,
    "better": 113,
    "worse": 45,
    "same": 1378,
    "query_changes": 242,
    "largest_positive_single_input_pp": 0.05510489088930068,
    "largest_negative_single_input_pp": -0.03892608322974319
  },
  {
    "cohort": "EXP",
    "control": "wrong_mean_context",
    "n": 1536,
    "gain_pp": 1.6670879366876616,
    "better": 95,
    "worse": 40,
    "same": 1401,
    "query_changes": 215,
    "largest_positive_single_input_pp": 0.0541219419982191,
    "largest_negative_single_input_pp": -0.04046528122671463
  },
  {
    "cohort": "EXP",
    "control": "native_ABC",
    "n": 1536,
    "gain_pp": 0.11343834191607736,
    "better": 110,
    "worse": 94,
    "same": 1332,
    "query_changes": 394,
    "largest_positive_single_input_pp": 0.05015647696241018,
    "largest_negative_single_input_pp": -0.051036878261435326
  },
  {
    "cohort": "EXP",
    "control": "lateOnly_F",
    "n": 1536,
    "gain_pp": -0.004531370019190202,
    "better": 30,
    "worse": 35,
    "same": 1471,
    "query_changes": 134,
    "largest_positive_single_input_pp": 0.04272420444360952,
    "largest_negative_single_input_pp": -0.051036878261435326
  }
]
```

全部匿名逐例、场景、负比较及凸优化诊断公开。所列good/failure是IoU变化，不是经人工认证的实例身份；未经图像复核的换query保持UNKNOWN。几何聚合目标下降只证明聚合按定义执行，不能代替语言目标的候选排序改善。

## 独立审计与成本边界

```json
{
  "state": "PASS_ALL_FROZEN_HEADS_EXACT_MEDIAN_CONVEX_KKT_WEIGHT_AND_CORRESPONDENCE_CONTROLS",
  "n": 2304,
  "aliases": 154,
  "CPU_heads": 21500,
  "triple_supported_inputs": 528,
  "point_triplet_KKT_checks": 667772,
  "max_KKT_violation": 4.569146926686581e-10,
  "max_probability_error": 1.9592923089506442e-06,
  "CPU_near_ties": 0,
  "GT_used": false,
  "FINAL_access": false
}
{
  "state": "PASS_SAME_OBSERVATION_RAW_PROJECTED_CONSENSUS_DIRECT_CONTROLS",
  "n": 2304,
  "readouts": 4608,
  "aliases": 154,
  "max_probability_error": 2.9853331662277327e-07,
  "near_ties": 0,
  "GT_used": false
}
{
  "state": "PASS_ALL_TASK_PLATFORM_SCENE_PAIRS_POSTSTRATIFICATION",
  "n": 2304,
  "arms": 22,
  "task_values": 50688,
  "new_task_values": 27648,
  "reused_task_values": 23040,
  "aliases": 154,
  "FINAL_access": false
}
{
  "n": 2304,
  "new_readouts": 27648,
  "reused_readouts": 23040,
  "geometry_max_error": 6.661338147750939e-16
}
```

没有梯度、参数更新、source准备、外部专家、原始媒体或FINAL访问。各条seconds覆盖多臂共享缓存/后缀研究开销，不是一个方法的新鲜端到端延迟。原始像素/文字/ID/坐标/特征/权重不公开。保留每个方向与每个场景的负结果，不用少数案例替代总体裁决。


## 最终解释与停止决定

主项稳健聚合DEV17.383000%，相对均值17.349002%仅+0.033999pp（6条改善、7条恶化）；EXP16.435239%，低于均值16.499566%，-0.064327pp（12条改善、16条恶化）。按完整场景频数后分层后，DEV和EXP相对均值分别为-0.027092pp和-0.178817pp。它未通过准入，不将中位数输出当新伪标签启动匹配TTA。

空间自适应权重没有建立稳定增量：DEV比当前输入的全局平均权重还低0.002491pp；EXP与全局平均权重几乎相同，且比空间打乱权重低0.016836pp。因此不能把DEV相对打乱的正差单独解释为“识别到了当前seed的坏观测”。几何聚合的凸KKT已经核验，无需通过增加迭代次数解释或挽救它。

附加强直接headmean也不稳定。它在EXP为16.533556%，略高于原均值上下文，但在DEV只有17.176454%，低于原均值。不能按范围选择均值、中位数或headmean，拼出一个虚假的统一方法。

保留的机制证据是：破坏空间对应会显著损害任务。真实中位数相对错配中位数DEV/EXP分别高2.930164/2.368162pp；原均值相对错配均值也有明显差异。这说明原生特征进入空间位置的方式确实影响选框，并非这些表示完全无效；却不认证每个已通过几何阈值的对应都是同一实例，也不证明无标签欧氏中心性可以判断哪一份表示更符合文字。

全部good/bad按总体对照留存。很多没有三份有效观测的输入完全复用均值表示；全部仍计分。对换query的案例，未看原图不宣布换了物体。当前的结论是停止这一固定稳健聚合配方，不展开中位数倍率、观测数或鲁棒核网格，也不拟合这份未合格的G。

若继续研究参数适应，下一机制必须从“中心性估计”转向可以直接约束已有表示的观测对应，并明确与过去query区域InfoNCE/随机采样一致性的差别。空间破坏控制只提供设计依据，不是新TTA已有效的结果；仍需同信息直接读出、错配/随机参数方向和未参与更新观测对照。源域训练、外挂专家和FINAL保持禁止。
