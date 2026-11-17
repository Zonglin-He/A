# R125 — 注册观测的任务原生语义一致性pointAdapter TTA

R123的点描述子对比改善辅助任务却损害选框；R124的源域GT受控诊断显示相同pointUp4896/三步预算可提高源域选框4.2226个百分点，但不是无标签TTA。本轮不使用这些标签、梯度或oracle权重，固定接口，只检验真实注册观测上的原生grounding语义一致性。

## 本轮实际改变

教师由同一冻结3EED在注册A/B完整文字上的候选分布等权平均，停止梯度。学生重算A/B原生cross-encoder与全部decoder，最小化教师到两份学生分布的平均KL，加单位权重的归一化source点特征保持。没有硬正标签、参照物规则、颜色规则、外部专家或新source准备。

仍从已有native foundation开始：pointAdapter输出权重/偏置与early RGB门置零，source-trained down固定；仅写输出288×16权重及288偏置4896参数。最多三步，每步归一化梯度半径上限沿用原source整个pointAdapter范数的1%，固定9级二分Armijo，仅按无标签目标接受或回退。一步是次项，不按GT选最佳端点。

A/B使用真实配准扫描，固定原A256空间锚点与六层位置，所以不把随机采样间变化的query编号硬作同物体；但近似注册仍不认证实例身份。资格与R123相同，无充分B支持时跳过更新，零梯度记录表示程序跳过，不宣称物理导数为零。未知与所有负例计入总体。

## 与旧实验及强对照的区别

R57是在同一观测的采样间做近似候选传输，并更新完整匹配MLP；R75在遮挡上下文的query/token空间做恢复，也只更新完整匹配MLP。本轮并不把KL或回溯说成新原理，区别是实际注册证据、固定候选位置及进入语言推理前的真实pointAdapter更新。旧方法失败不保证新设置成功，必须直接检验。

固定RGB late为sourceA、竞争残差为sourceMeanF，与所有新臂相同。适应本身仍是LiDAR—文字原生分布一致性；保留RGB上下文不等于本轮学到了三模态共享语义。

除冻结点均值、head均值、R122中位数等对照外，新增完全同信息的直接语义读出：log教师概率 + 零步context logits − 零步native logits，再softmax，无网络更新。它是冻结AB语义共识加相同RGB/竞争修正，系数固定一，不另扫分数融合。

错配对照置换教师的候选位置，保留概率集合与熵；随机对照与实际正确更新的累计参数范数相同。更强无网络更新对照直接优化A/B原始点特征变量，使用相同KL、保持与Armijo，步能量上限匹配真实Adapter各步诱发的A/B特征改变量，最后将平均修正场作用于F/C。它需要反向传播到自由变量，但不修改网络；不是零计算，也不是全局最优直接解。

全部提案、自由特征与随机端点先确定，随后才解码C及C相关context provider。C不参与梯度或验收；F和最终RGB上下文本来允许读取C，故不是因果在线协议或新场景确认。额外报告未参与拟合C的KL与真实选框，两者不互相替代。C仅更换已注册点特征，任务仍选择A的固定256框并沿用A标注；它不是独立新帧GT或新的场景确认，动态目标的实例对应仍可能未知。

## 数据与完整性

DEV768/EXP1536均历史曝光，所有2304方向输入、154保护别名在分母。别名先检查、不加载特征，沿用prepared_A回退。三source100/R91均不重训，原A256坐标/全文视角/所有decoder位置不变，FINAL不动。场景后分层只用原全量场景频数；EXP缺少4个Waymo场景，仅共同支持。headmean参照不是历史R59软query传输，原DEV336 R59研究结果保留原study，不伪造跨清单同条件值。

首次未完成运行的private序列化让同名梯度覆盖参数端点，预测计算在内存中正确但审计端点不可靠。发现时尚未读取任务GT，已中止并保留全部partial/engineering产物与原代码/锁/日志。分离_gradient键后重新完成24工程与6CPU schema探针，再正式重做；没有根据任务成绩改配置。失败尝试保留，不能删除或计作方法负结果。

## 总体结果

| 范围 | 方法 | mIoU% | Acc25% | Acc50% |
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
| DEV | corr1_F | 17.263535 | 31.901042 | 14.192708 |
| DEV | corr1_C | 16.453460 | 30.468750 | 13.671875 |
| DEV | corr3_F | 17.183657 | 31.901042 | 14.192708 |
| DEV | corr3_C | 16.395135 | 30.338542 | 13.541667 |
| DEV | wrong1_F | 17.073532 | 31.640625 | 14.192708 |
| DEV | wrong1_C | 16.269577 | 30.468750 | 13.281250 |
| DEV | wrong3_F | 16.753181 | 30.859375 | 14.062500 |
| DEV | wrong3_C | 16.083455 | 30.208333 | 13.411458 |
| DEV | random1_F | 17.349002 | 31.901042 | 14.453125 |
| DEV | random1_C | 16.493471 | 30.729167 | 13.671875 |
| DEV | random3_F | 17.353267 | 32.031250 | 14.453125 |
| DEV | random3_C | 16.504216 | 30.729167 | 13.671875 |
| DEV | free3_F | 17.049592 | 31.119792 | 14.062500 |
| DEV | free3_C | 16.281418 | 29.817708 | 13.671875 |
| DEV | direct_F | 17.004480 | 31.640625 | 13.802083 |
| DEV | direct_C | 16.750924 | 31.250000 | 13.671875 |
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
| EXP | corr1_F | 16.506737 | 31.250000 | 12.109375 |
| EXP | corr1_C | 16.202944 | 30.794271 | 12.109375 |
| EXP | corr3_F | 16.435119 | 31.119792 | 12.044271 |
| EXP | corr3_C | 16.215379 | 30.794271 | 12.174479 |
| EXP | wrong1_F | 16.334969 | 30.859375 | 12.109375 |
| EXP | wrong1_C | 16.178852 | 30.924479 | 12.044271 |
| EXP | wrong3_F | 16.077595 | 30.533854 | 11.848958 |
| EXP | wrong3_C | 15.961698 | 30.403646 | 12.044271 |
| EXP | random1_F | 16.499566 | 31.119792 | 12.174479 |
| EXP | random1_C | 16.199495 | 30.794271 | 12.174479 |
| EXP | random3_F | 16.520437 | 31.184896 | 12.174479 |
| EXP | random3_C | 16.184336 | 30.729167 | 12.174479 |
| EXP | free3_F | 16.379069 | 31.054688 | 12.044271 |
| EXP | free3_C | 16.139837 | 30.729167 | 12.174479 |
| EXP | direct_F | 16.243921 | 30.729167 | 11.914062 |
| EXP | direct_C | 16.176727 | 30.664062 | 11.588542 |

## 各目标平台

| 范围 | 目标 | 方法 | mIoU% | Acc25% | Acc50% |
|---|---|---|---:|---:|---:|
| DEV | waymo | original_source | 11.473631 | 23.437500 | 7.421875 |
| DEV | waymo | prepared_A | 12.615516 | 26.171875 | 5.078125 |
| DEV | waymo | mean_context | 12.247576 | 24.609375 | 7.812500 |
| DEV | waymo | headmean_context | 12.097959 | 24.218750 | 7.812500 |
| DEV | waymo | zero_C | 12.008417 | 23.828125 | 7.812500 |
| DEV | waymo | corr1_F | 12.247576 | 24.609375 | 7.812500 |
| DEV | waymo | corr3_F | 12.247576 | 24.609375 | 7.812500 |
| DEV | waymo | corr3_C | 12.046046 | 23.828125 | 7.812500 |
| DEV | waymo | wrong3_F | 12.092461 | 24.218750 | 7.812500 |
| DEV | waymo | random3_F | 12.247576 | 24.609375 | 7.812500 |
| DEV | waymo | free3_F | 12.199003 | 24.609375 | 7.812500 |
| DEV | waymo | direct_F | 12.284501 | 24.609375 | 7.812500 |
| DEV | drone | original_source | 13.867693 | 24.609375 | 12.500000 |
| DEV | drone | prepared_A | 14.071678 | 24.609375 | 11.718750 |
| DEV | drone | mean_context | 14.805341 | 26.953125 | 11.718750 |
| DEV | drone | headmean_context | 14.514873 | 26.171875 | 11.718750 |
| DEV | drone | zero_C | 14.545494 | 26.953125 | 11.328125 |
| DEV | drone | corr1_F | 14.917599 | 27.343750 | 11.718750 |
| DEV | drone | corr3_F | 14.932339 | 27.734375 | 11.718750 |
| DEV | drone | corr3_C | 14.614340 | 26.953125 | 11.328125 |
| DEV | drone | wrong3_F | 14.376792 | 25.781250 | 11.718750 |
| DEV | drone | random3_F | 14.805341 | 26.953125 | 11.718750 |
| DEV | drone | free3_F | 14.635430 | 26.562500 | 11.328125 |
| DEV | drone | direct_F | 14.809839 | 26.953125 | 11.718750 |
| DEV | quad | original_source | 25.092825 | 44.921875 | 25.000000 |
| DEV | quad | prepared_A | 21.998762 | 39.062500 | 21.093750 |
| DEV | quad | mean_context | 24.994087 | 44.140625 | 23.828125 |
| DEV | quad | headmean_context | 24.916529 | 44.531250 | 24.218750 |
| DEV | quad | zero_C | 22.926502 | 41.406250 | 21.875000 |
| DEV | quad | corr1_F | 24.625429 | 43.750000 | 23.046875 |
| DEV | quad | corr3_F | 24.371055 | 43.359375 | 23.046875 |
| DEV | quad | corr3_C | 22.525019 | 40.234375 | 21.484375 |
| DEV | quad | wrong3_F | 23.790292 | 42.578125 | 22.656250 |
| DEV | quad | random3_F | 25.006882 | 44.531250 | 23.828125 |
| DEV | quad | free3_F | 24.314343 | 42.187500 | 23.046875 |
| DEV | quad | direct_F | 23.919102 | 43.359375 | 21.875000 |
| EXP | waymo | original_source | 13.450978 | 28.906250 | 6.835938 |
| EXP | waymo | prepared_A | 13.214120 | 28.125000 | 6.250000 |
| EXP | waymo | mean_context | 12.855988 | 27.343750 | 6.054688 |
| EXP | waymo | headmean_context | 12.988950 | 27.734375 | 6.054688 |
| EXP | waymo | zero_C | 12.799592 | 27.343750 | 6.054688 |
| EXP | waymo | corr1_F | 12.814917 | 27.343750 | 6.054688 |
| EXP | waymo | corr3_F | 12.744220 | 27.148438 | 6.054688 |
| EXP | waymo | corr3_C | 12.675991 | 26.953125 | 6.054688 |
| EXP | waymo | wrong3_F | 12.430416 | 26.562500 | 5.664062 |
| EXP | waymo | random3_F | 12.855988 | 27.343750 | 6.054688 |
| EXP | waymo | free3_F | 12.567778 | 27.148438 | 6.054688 |
| EXP | waymo | direct_F | 12.805879 | 27.539062 | 6.054688 |
| EXP | drone | original_source | 11.525665 | 20.898438 | 9.375000 |
| EXP | drone | prepared_A | 12.009923 | 20.312500 | 9.179688 |
| EXP | drone | mean_context | 12.279426 | 22.265625 | 9.570312 |
| EXP | drone | headmean_context | 12.240413 | 22.070312 | 9.765625 |
| EXP | drone | zero_C | 11.958571 | 21.289062 | 9.375000 |
| EXP | drone | corr1_F | 12.287886 | 22.265625 | 9.570312 |
| EXP | drone | corr3_F | 12.290231 | 22.265625 | 9.570312 |
| EXP | drone | corr3_C | 11.937016 | 21.289062 | 9.375000 |
| EXP | drone | wrong3_F | 12.064147 | 21.875000 | 9.375000 |
| EXP | drone | random3_F | 12.279426 | 22.265625 | 9.570312 |
| EXP | drone | free3_F | 12.289046 | 22.070312 | 9.570312 |
| EXP | drone | direct_F | 11.992226 | 21.679688 | 9.179688 |
| EXP | quad | original_source | 23.457215 | 41.992188 | 21.093750 |
| EXP | quad | prepared_A | 23.529157 | 43.359375 | 19.921875 |
| EXP | quad | mean_context | 24.363284 | 43.750000 | 20.898438 |
| EXP | quad | headmean_context | 24.371304 | 44.140625 | 21.289062 |
| EXP | quad | zero_C | 23.798588 | 43.750000 | 21.093750 |
| EXP | quad | corr1_F | 24.417408 | 44.140625 | 20.703125 |
| EXP | quad | corr3_F | 24.270907 | 43.945312 | 20.507812 |
| EXP | quad | corr3_C | 24.033129 | 44.140625 | 21.093750 |
| EXP | quad | wrong3_F | 23.738221 | 43.164062 | 20.507812 |
| EXP | quad | random3_F | 24.425898 | 43.945312 | 20.898438 |
| EXP | quad | free3_F | 24.280383 | 43.945312 | 20.507812 |
| EXP | quad | direct_F | 23.933658 | 42.968750 | 20.507812 |

## 场景频数控制

| 范围 | 方法 | 最小支持覆盖 | 后分层mIoU% |
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
| DEV | corr1_F | 1.000000 | 12.550540 |
| DEV | corr1_C | 1.000000 | 12.152160 |
| DEV | corr3_F | 1.000000 | 12.467380 |
| DEV | corr3_C | 1.000000 | 12.120706 |
| DEV | wrong1_F | 1.000000 | 12.444584 |
| DEV | wrong1_C | 1.000000 | 12.074738 |
| DEV | wrong3_F | 1.000000 | 12.097912 |
| DEV | wrong3_C | 1.000000 | 11.862898 |
| DEV | random1_F | 1.000000 | 12.673432 |
| DEV | random1_C | 1.000000 | 12.169366 |
| DEV | random3_F | 1.000000 | 12.682810 |
| DEV | random3_C | 1.000000 | 12.179548 |
| DEV | free3_F | 1.000000 | 12.334511 |
| DEV | free3_C | 1.000000 | 12.102099 |
| DEV | direct_F | 1.000000 | 12.454153 |
| DEV | direct_C | 1.000000 | 12.404671 |
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
| EXP | corr1_F | 0.998149 | 12.957582 |
| EXP | corr1_C | 0.998149 | 12.712740 |
| EXP | corr3_F | 0.998149 | 12.879481 |
| EXP | corr3_C | 0.998149 | 12.638783 |
| EXP | wrong1_F | 0.998149 | 12.757652 |
| EXP | wrong1_C | 0.998149 | 12.807323 |
| EXP | wrong3_F | 0.998149 | 12.599756 |
| EXP | wrong3_C | 0.998149 | 12.503689 |
| EXP | random1_F | 0.998149 | 12.951703 |
| EXP | random1_C | 0.998149 | 12.629566 |
| EXP | random3_F | 0.998149 | 12.958788 |
| EXP | random3_C | 0.998149 | 12.596626 |
| EXP | free3_F | 0.998149 | 12.780594 |
| EXP | free3_C | 0.998149 | 12.681485 |
| EXP | direct_F | 0.998149 | 12.659956 |
| EXP | direct_C | 0.998149 | 12.788927 |

## OBSERVATION_INFORMATION_SUMMARY

```json
{
  "state": "POSTSCORE_DESCRIPTIVE_ONLY_NO_GATE_OR_SELECTOR",
  "groups": [
    {
      "cohort": "DEV",
      "all_denominator": 768,
      "nonalias": 724,
      "active_AB": 258,
      "claim_boundary": "Query seed-anchor validity and source-distribution changes are approximate support diagnostics, NOT object visibility/identity or correctness; no cutoff chosen",
      "B_seed_valid_fraction_mean": 0.7019970324612403,
      "B_seed_valid_fraction_quantiles_0_25_50_75_100": [
        0.2509765625,
        0.62890625,
        0.7109375,
        0.7998046875,
        0.9365234375
      ],
      "B_native_probability_weighted_anchor_support_mean": 0.7726519137602744,
      "B_native_probability_weighted_anchor_support_quantiles_0_25_50_75_100": [
        5.180307018113337e-05,
        0.7016928315901902,
        0.9340577845441231,
        0.9997435090291951,
        0.999999198535
      ],
      "C_native_probability_weighted_anchor_support_mean": 0.5436007539668293,
      "C_native_probability_weighted_anchor_support_quantiles_0_25_50_75_100": [
        0.0,
        0.0,
        0.7375088953933624,
        0.9996705074568215,
        0.9999993652831315
      ],
      "native_AB_JS_mean": 0.11153689171814925,
      "native_AB_JS_quantiles_0_25_50_75_100": [
        1.618510427559501e-05,
        0.009585766678370088,
        0.02982706476380277,
        0.13213704770885593,
        0.68782915728349
      ],
      "native_AB_probability_L1_mean": 0.5205153484213003,
      "native_AB_probability_L1_quantiles_0_25_50_75_100": [
        0.0001614359961515076,
        0.14424668009751437,
        0.2878874219790285,
        0.7018571682134997,
        1.9930162437567929
      ],
      "active_native_AB_same_Top1": 148,
      "by_zero_winner_anchor_B_support": [
        {
          "B_anchor_supported": false,
          "n": 68,
          "task_gain_pp_within_subgroup": 0.4649182445906906,
          "contribution_pp_original_full_denominator": 0.04116463623980073,
          "better": 1,
          "worse": 1
        },
        {
          "B_anchor_supported": true,
          "n": 190,
          "task_gain_pp_within_subgroup": -0.8347335072370552,
          "contribution_pp_original_full_denominator": -0.20650959163416727,
          "better": 5,
          "worse": 9
        }
      ],
      "changed_query_geometry": [
        {
          "geometric_group_NOT_identity_label": "high_overlap_ge0.7",
          "n": 4,
          "contribution_pp_original_full_denominator": -0.003114523850567904,
          "better": 1,
          "worse": 2
        },
        {
          "geometric_group_NOT_identity_label": "low_overlap_le0.1",
          "n": 12,
          "contribution_pp_original_full_denominator": -0.16473445701052497,
          "better": 3,
          "worse": 7
        },
        {
          "geometric_group_NOT_identity_label": "intermediate_overlap",
          "n": 8,
          "contribution_pp_original_full_denominator": 0.0025040254667263195,
          "better": 2,
          "worse": 1
        }
      ]
    },
    {
      "cohort": "EXP",
      "all_denominator": 1536,
      "nonalias": 1426,
      "active_AB": 528,
      "claim_boundary": "Query seed-anchor validity and source-distribution changes are approximate support diagnostics, NOT object visibility/identity or correctness; no cutoff chosen",
      "B_seed_valid_fraction_mean": 0.6902891216856061,
      "B_seed_valid_fraction_quantiles_0_25_50_75_100": [
        0.244140625,
        0.62060546875,
        0.6953125,
        0.798095703125,
        0.9765625
      ],
      "B_native_probability_weighted_anchor_support_mean": 0.7854159717037249,
      "B_native_probability_weighted_anchor_support_quantiles_0_25_50_75_100": [
        6.349639911937492e-05,
        0.6707311387117165,
        0.9520050804624646,
        0.9997789939246433,
        0.999997698776316
      ],
      "C_native_probability_weighted_anchor_support_mean": 0.525695271871252,
      "C_native_probability_weighted_anchor_support_quantiles_0_25_50_75_100": [
        0.0,
        0.0,
        0.7025844472025082,
        0.9992854504168573,
        0.9999973795055249
      ],
      "native_AB_JS_mean": 0.11920851175136836,
      "native_AB_JS_quantiles_0_25_50_75_100": [
        4.927291702417992e-06,
        0.009124540807163392,
        0.03749728626877376,
        0.1483118824307184,
        0.6914738630558133
      ],
      "native_AB_probability_L1_mean": 0.5478365820038139,
      "native_AB_probability_L1_quantiles_0_25_50_75_100": [
        7.659349945139675e-05,
        0.12556478911953267,
        0.3211221495096458,
        0.7870724006568455,
        1.9986143566267187
      ],
      "active_native_AB_same_Top1": 267,
      "by_zero_winner_anchor_B_support": [
        {
          "B_anchor_supported": false,
          "n": 122,
          "task_gain_pp_within_subgroup": -1.3628367078126116,
          "contribution_pp_original_full_denominator": -0.10824614476115796,
          "better": 1,
          "worse": 5
        },
        {
          "B_anchor_supported": true,
          "n": 406,
          "task_gain_pp_within_subgroup": 0.1657052129025043,
          "contribution_pp_original_full_denominator": 0.04379968518126091,
          "better": 9,
          "worse": 7
        }
      ],
      "changed_query_geometry": [
        {
          "geometric_group_NOT_identity_label": "high_overlap_ge0.7",
          "n": 6,
          "contribution_pp_original_full_denominator": 0.0050432023811988755,
          "better": 3,
          "worse": 2
        },
        {
          "geometric_group_NOT_identity_label": "low_overlap_le0.1",
          "n": 25,
          "contribution_pp_original_full_denominator": -0.05387634133522228,
          "better": 6,
          "worse": 8
        },
        {
          "geometric_group_NOT_identity_label": "intermediate_overlap",
          "n": 8,
          "contribution_pp_original_full_denominator": -0.015613320625873645,
          "better": 1,
          "worse": 2
        }
      ]
    }
  ],
  "source_GT_used": false,
  "used_for_method_selection": false,
  "FINAL_access": false
}
```

## DECISION

```json
{
  "primary": "corr3_F",
  "cohorts": {
    "DEV": {
      "mIoU_percent": 17.183656701859316,
      "gains_pp": {
        "mean_context": -0.16534495539436733,
        "headmean_context": 0.007202767299233415,
        "robust_context": -0.19934368735394514,
        "direct_F": 0.17917623108110803,
        "free3_F": 0.13406459044373875,
        "wrong3_F": 0.4304752581287339,
        "random3_F": -0.16960989688652361
      },
      "poststratified_mIoU_percent": 12.467380004185516,
      "poststratified_gains_pp": {
        "mean_context": -0.20605201100867987,
        "headmean_context": 0.1406740457568123,
        "robust_context": -0.17896015116996367,
        "direct_F": 0.013227072181168253,
        "free3_F": 0.13286886418742228,
        "wrong3_F": 0.36946817386072794,
        "random3_F": -0.21542989405576485
      },
      "passes_all_prelocked_controls": false,
      "unused_C_available_macro_gain_pp": -0.18184563140164337,
      "unused_C_gate": false
    },
    "EXP": {
      "mIoU_percent": 16.435119415607137,
      "gains_pp": {
        "mean_context": -0.06444645957989792,
        "headmean_context": -0.09843627636841035,
        "robust_context": -0.00011913867685819923,
        "direct_F": 0.191198503230531,
        "free3_F": 0.056050618115005335,
        "wrong3_F": 0.3575247713012857,
        "random3_F": -0.08531801888435098
      },
      "poststratified_mIoU_percent": 12.879481104770235,
      "poststratified_gains_pp": {
        "mean_context": -0.07222177064014978,
        "headmean_context": -0.07167714616479337,
        "robust_context": 0.1065952960185701,
        "direct_F": 0.21952529728775794,
        "free3_F": 0.09888748662911873,
        "wrong3_F": 0.2797250612834059,
        "random3_F": -0.0793066899960504
      },
      "passes_all_prelocked_controls": false,
      "unused_C_available_macro_gain_pp": -0.054295825275164224,
      "unused_C_gate": false
    }
  },
  "FINAL_access": false,
  "posthoc_primary_switch": false,
  "development_gate_pass": false,
  "TTA_robustly_adopted": false
}
```

## HELDOUT_C_TASK_ROWS

```json
[
  {
    "cohort": "DEV",
    "source": "drone",
    "target": "quad",
    "n": 62,
    "gain_pp": 0.16741116432578865,
    "better": 2,
    "worse": 3
  },
  {
    "cohort": "DEV",
    "source": "drone",
    "target": "waymo",
    "n": 17,
    "gain_pp": 0.08125417317613264,
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
    "gain_pp": 0.0,
    "better": 0,
    "worse": 0
  },
  {
    "cohort": "DEV",
    "source": "waymo",
    "target": "quad",
    "n": 62,
    "gain_pp": -1.8251478170145812,
    "better": 1,
    "worse": 2
  },
  {
    "cohort": "EXP",
    "source": "drone",
    "target": "quad",
    "n": 109,
    "gain_pp": 0.6737924679498973,
    "better": 5,
    "worse": 3
  },
  {
    "cohort": "EXP",
    "source": "drone",
    "target": "waymo",
    "n": 40,
    "gain_pp": 0.015494983208688695,
    "better": 1,
    "worse": 0
  },
  {
    "cohort": "EXP",
    "source": "quad",
    "target": "drone",
    "n": 26,
    "gain_pp": 0.27622587352972183,
    "better": 1,
    "worse": 0
  },
  {
    "cohort": "EXP",
    "source": "quad",
    "target": "waymo",
    "n": 40,
    "gain_pp": -0.7705307931511775,
    "better": 0,
    "worse": 1
  },
  {
    "cohort": "EXP",
    "source": "waymo",
    "target": "drone",
    "n": 26,
    "gain_pp": -0.7604341903696776,
    "better": 1,
    "worse": 2
  },
  {
    "cohort": "EXP",
    "source": "waymo",
    "target": "quad",
    "n": 109,
    "gain_pp": 0.23967670718156184,
    "better": 1,
    "worse": 1
  }
]
```

## SEMANTIC_COMPONENT_SUMMARY

```json
[
  {
    "cohort": "DEV",
    "source": "waymo",
    "n_nonalias": 234,
    "active_AB": 108,
    "accepted_first_steps": 108,
    "accepted_steps_total": 324,
    "mean_first_accepted_fraction": 0.7217881944444444,
    "mean_actual_parameter_L2_at3": 0.05402417331638428,
    "corr1_fit_objective_n": 108,
    "corr1_fit_objective_mean_change": -0.0407620170569294,
    "corr1_fit_objective_decreases": 108,
    "corr1_fit_KL_n": 108,
    "corr1_fit_KL_mean_change": -0.04101041100839673,
    "corr1_fit_KL_decreases": 108,
    "corr1_heldout_C_KL_n": 72,
    "corr1_heldout_C_KL_mean_change": -0.05793315541415392,
    "corr1_heldout_C_KL_decreases": 49,
    "corr3_fit_objective_n": 108,
    "corr3_fit_objective_mean_change": -0.09169551795117378,
    "corr3_fit_objective_decreases": 108,
    "corr3_fit_KL_n": 108,
    "corr3_fit_KL_mean_change": -0.0926261913706637,
    "corr3_fit_KL_decreases": 108,
    "corr3_heldout_C_KL_n": 72,
    "corr3_heldout_C_KL_mean_change": -0.13337839887322725,
    "corr3_heldout_C_KL_decreases": 55,
    "wrong3_fit_objective_n": 108,
    "wrong3_fit_objective_mean_change": 0.07616288190240514,
    "wrong3_fit_objective_decreases": 16,
    "wrong3_fit_KL_n": 108,
    "wrong3_fit_KL_mean_change": 0.07226090892587471,
    "wrong3_fit_KL_decreases": 17,
    "wrong3_heldout_C_KL_n": 72,
    "wrong3_heldout_C_KL_mean_change": 0.003996057030258148,
    "wrong3_heldout_C_KL_decreases": 25,
    "random3_fit_objective_n": 108,
    "random3_fit_objective_mean_change": -0.00016858408416321028,
    "random3_fit_objective_decreases": 33,
    "random3_fit_KL_n": 108,
    "random3_fit_KL_mean_change": -0.0002888763654512353,
    "random3_fit_KL_decreases": 57,
    "random3_heldout_C_KL_n": 72,
    "random3_heldout_C_KL_mean_change": 0.0014231090079840022,
    "random3_heldout_C_KL_decreases": 36,
    "free3_fit_objective_n": 108,
    "free3_fit_objective_mean_change": -0.13336387209194442,
    "free3_fit_objective_decreases": 108,
    "free3_fit_KL_n": 108,
    "free3_fit_KL_mean_change": -0.13337489588696047,
    "free3_fit_KL_decreases": 108,
    "free3_heldout_C_KL_n": 72,
    "free3_heldout_C_KL_mean_change": 0.01606166462753871,
    "free3_heldout_C_KL_decreases": 41
  },
  {
    "cohort": "DEV",
    "source": "drone",
    "n_nonalias": 244,
    "active_AB": 101,
    "accepted_first_steps": 101,
    "accepted_steps_total": 303,
    "mean_first_accepted_fraction": 0.807549504950495,
    "mean_actual_parameter_L2_at3": 0.05084670142698097,
    "corr1_fit_objective_n": 101,
    "corr1_fit_objective_mean_change": -0.04941313905544556,
    "corr1_fit_objective_decreases": 101,
    "corr1_fit_KL_n": 101,
    "corr1_fit_KL_mean_change": -0.04981160661024825,
    "corr1_fit_KL_decreases": 101,
    "corr1_heldout_C_KL_n": 79,
    "corr1_heldout_C_KL_mean_change": -0.036225320667618875,
    "corr1_heldout_C_KL_decreases": 56,
    "corr3_fit_objective_n": 101,
    "corr3_fit_objective_mean_change": -0.10107002213111851,
    "corr3_fit_objective_decreases": 101,
    "corr3_fit_KL_n": 101,
    "corr3_fit_KL_mean_change": -0.10285953122559124,
    "corr3_fit_KL_decreases": 101,
    "corr3_heldout_C_KL_n": 79,
    "corr3_heldout_C_KL_mean_change": -0.11476880481702886,
    "corr3_heldout_C_KL_decreases": 63,
    "wrong3_fit_objective_n": 101,
    "wrong3_fit_objective_mean_change": 0.08692447491931773,
    "wrong3_fit_objective_decreases": 22,
    "wrong3_fit_KL_n": 101,
    "wrong3_fit_KL_mean_change": 0.08209455813900114,
    "wrong3_fit_KL_decreases": 28,
    "wrong3_heldout_C_KL_n": 79,
    "wrong3_heldout_C_KL_mean_change": 0.18085654624304456,
    "wrong3_heldout_C_KL_decreases": 28,
    "random3_fit_objective_n": 101,
    "random3_fit_objective_mean_change": 0.0006118787202053549,
    "random3_fit_objective_decreases": 35,
    "random3_fit_KL_n": 101,
    "random3_fit_KL_mean_change": 0.0004994437138124655,
    "random3_fit_KL_decreases": 42,
    "random3_heldout_C_KL_n": 79,
    "random3_heldout_C_KL_mean_change": 0.000887323738514412,
    "random3_heldout_C_KL_decreases": 40,
    "free3_fit_objective_n": 101,
    "free3_fit_objective_mean_change": -0.19343357997418228,
    "free3_fit_objective_decreases": 101,
    "free3_fit_KL_n": 101,
    "free3_fit_KL_mean_change": -0.1934440667340673,
    "free3_fit_KL_decreases": 101,
    "free3_heldout_C_KL_n": 79,
    "free3_heldout_C_KL_mean_change": -0.07951018691559764,
    "free3_heldout_C_KL_decreases": 49
  },
  {
    "cohort": "DEV",
    "source": "quad",
    "n_nonalias": 246,
    "active_AB": 49,
    "accepted_first_steps": 49,
    "accepted_steps_total": 147,
    "mean_first_accepted_fraction": 0.7681760204081632,
    "mean_actual_parameter_L2_at3": 0.0475751783815296,
    "corr1_fit_objective_n": 49,
    "corr1_fit_objective_mean_change": -0.05166349187546148,
    "corr1_fit_objective_decreases": 49,
    "corr1_fit_KL_n": 49,
    "corr1_fit_KL_mean_change": -0.051942441204242906,
    "corr1_fit_KL_decreases": 49,
    "corr1_heldout_C_KL_n": 27,
    "corr1_heldout_C_KL_mean_change": -0.06492843716484403,
    "corr1_heldout_C_KL_decreases": 17,
    "corr3_fit_objective_n": 49,
    "corr3_fit_objective_mean_change": -0.13643796216495416,
    "corr3_fit_objective_decreases": 49,
    "corr3_fit_KL_n": 49,
    "corr3_fit_KL_mean_change": -0.13750823853654764,
    "corr3_fit_KL_decreases": 49,
    "corr3_heldout_C_KL_n": 27,
    "corr3_heldout_C_KL_mean_change": -0.18023973952989786,
    "corr3_heldout_C_KL_decreases": 21,
    "wrong3_fit_objective_n": 49,
    "wrong3_fit_objective_mean_change": 0.24884542860516196,
    "wrong3_fit_objective_decreases": 6,
    "wrong3_fit_KL_n": 49,
    "wrong3_fit_KL_mean_change": 0.24551551589616574,
    "wrong3_fit_KL_decreases": 6,
    "wrong3_heldout_C_KL_n": 27,
    "wrong3_heldout_C_KL_mean_change": 0.12259185856759339,
    "wrong3_heldout_C_KL_decreases": 7,
    "random3_fit_objective_n": 49,
    "random3_fit_objective_mean_change": 0.0004096778828827111,
    "random3_fit_objective_decreases": 23,
    "random3_fit_KL_n": 49,
    "random3_fit_KL_mean_change": 0.0003278114859854901,
    "random3_fit_KL_decreases": 26,
    "random3_heldout_C_KL_n": 27,
    "random3_heldout_C_KL_mean_change": -0.009179852846902687,
    "random3_heldout_C_KL_decreases": 13,
    "free3_fit_objective_n": 49,
    "free3_fit_objective_mean_change": -0.3143964987176003,
    "free3_fit_objective_decreases": 49,
    "free3_fit_KL_n": 49,
    "free3_fit_KL_mean_change": -0.31441176910693697,
    "free3_fit_KL_decreases": 49,
    "free3_heldout_C_KL_n": 27,
    "free3_heldout_C_KL_mean_change": -0.12078564680467477,
    "free3_heldout_C_KL_decreases": 18
  },
  {
    "cohort": "EXP",
    "source": "waymo",
    "n_nonalias": 457,
    "active_AB": 211,
    "accepted_first_steps": 211,
    "accepted_steps_total": 633,
    "mean_first_accepted_fraction": 0.7228043542654028,
    "mean_actual_parameter_L2_at3": 0.054916310919681056,
    "corr1_fit_objective_n": 211,
    "corr1_fit_objective_mean_change": -0.05812089978869932,
    "corr1_fit_objective_decreases": 211,
    "corr1_fit_KL_n": 211,
    "corr1_fit_KL_mean_change": -0.05836013952178087,
    "corr1_fit_KL_decreases": 211,
    "corr1_heldout_C_KL_n": 135,
    "corr1_heldout_C_KL_mean_change": -0.045620342821912775,
    "corr1_heldout_C_KL_decreases": 86,
    "corr3_fit_objective_n": 211,
    "corr3_fit_objective_mean_change": -0.1321794387204933,
    "corr3_fit_objective_decreases": 211,
    "corr3_fit_KL_n": 211,
    "corr3_fit_KL_mean_change": -0.13310848834185682,
    "corr3_fit_KL_decreases": 211,
    "corr3_heldout_C_KL_n": 135,
    "corr3_heldout_C_KL_mean_change": -0.09826105188784166,
    "corr3_heldout_C_KL_decreases": 94,
    "wrong3_fit_objective_n": 211,
    "wrong3_fit_objective_mean_change": 0.04683426286082259,
    "wrong3_fit_objective_decreases": 39,
    "wrong3_fit_KL_n": 211,
    "wrong3_fit_KL_mean_change": 0.04296300123200124,
    "wrong3_fit_KL_decreases": 39,
    "wrong3_heldout_C_KL_n": 135,
    "wrong3_heldout_C_KL_mean_change": -0.021295632740909686,
    "wrong3_heldout_C_KL_decreases": 50,
    "random3_fit_objective_n": 211,
    "random3_fit_objective_mean_change": -0.0002809399270817743,
    "random3_fit_objective_decreases": 55,
    "random3_fit_KL_n": 211,
    "random3_fit_KL_mean_change": -0.00040590711190511327,
    "random3_fit_KL_decreases": 106,
    "random3_heldout_C_KL_n": 135,
    "random3_heldout_C_KL_mean_change": -0.0005157048008360895,
    "random3_heldout_C_KL_decreases": 78,
    "free3_fit_objective_n": 211,
    "free3_fit_objective_mean_change": -0.19086814932784066,
    "free3_fit_objective_decreases": 211,
    "free3_fit_KL_n": 211,
    "free3_fit_KL_mean_change": -0.1908760662664149,
    "free3_fit_KL_decreases": 211,
    "free3_heldout_C_KL_n": 135,
    "free3_heldout_C_KL_mean_change": 0.043875153685829296,
    "free3_heldout_C_KL_decreases": 82
  },
  {
    "cohort": "EXP",
    "source": "drone",
    "n_nonalias": 483,
    "active_AB": 200,
    "accepted_first_steps": 200,
    "accepted_steps_total": 598,
    "mean_first_accepted_fraction": 0.82361328125,
    "mean_actual_parameter_L2_at3": 0.05211191076540892,
    "corr1_fit_objective_n": 200,
    "corr1_fit_objective_mean_change": -0.04744304809692487,
    "corr1_fit_objective_decreases": 200,
    "corr1_fit_KL_n": 200,
    "corr1_fit_KL_mean_change": -0.047856347938269546,
    "corr1_fit_KL_decreases": 200,
    "corr1_heldout_C_KL_n": 149,
    "corr1_heldout_C_KL_mean_change": -0.07399811568786666,
    "corr1_heldout_C_KL_decreases": 98,
    "corr3_fit_objective_n": 200,
    "corr3_fit_objective_mean_change": -0.1007110168228827,
    "corr3_fit_objective_decreases": 200,
    "corr3_fit_KL_n": 200,
    "corr3_fit_KL_mean_change": -0.1025502942397583,
    "corr3_fit_KL_decreases": 200,
    "corr3_heldout_C_KL_n": 149,
    "corr3_heldout_C_KL_mean_change": -0.14687463456677055,
    "corr3_heldout_C_KL_decreases": 106,
    "wrong3_fit_objective_n": 200,
    "wrong3_fit_objective_mean_change": 0.08976029769900996,
    "wrong3_fit_objective_decreases": 42,
    "wrong3_fit_KL_n": 200,
    "wrong3_fit_KL_mean_change": 0.08494867358856645,
    "wrong3_fit_KL_decreases": 47,
    "wrong3_heldout_C_KL_n": 149,
    "wrong3_heldout_C_KL_mean_change": 0.13219236032451606,
    "wrong3_heldout_C_KL_decreases": 53,
    "random3_fit_objective_n": 200,
    "random3_fit_objective_mean_change": 5.535167955498629e-05,
    "random3_fit_objective_decreases": 84,
    "random3_fit_KL_n": 200,
    "random3_fit_KL_mean_change": -6.421209400090728e-05,
    "random3_fit_KL_decreases": 100,
    "random3_heldout_C_KL_n": 149,
    "random3_heldout_C_KL_mean_change": 0.0002452005500270821,
    "random3_heldout_C_KL_decreases": 78,
    "free3_fit_objective_n": 200,
    "free3_fit_objective_mean_change": -0.1703367970674014,
    "free3_fit_objective_decreases": 200,
    "free3_fit_KL_n": 200,
    "free3_fit_KL_mean_change": -0.17034533990996323,
    "free3_fit_KL_decreases": 200,
    "free3_heldout_C_KL_n": 149,
    "free3_heldout_C_KL_mean_change": -0.0793041898589697,
    "free3_heldout_C_KL_decreases": 96
  },
  {
    "cohort": "EXP",
    "source": "quad",
    "n_nonalias": 486,
    "active_AB": 117,
    "accepted_first_steps": 117,
    "accepted_steps_total": 351,
    "mean_first_accepted_fraction": 0.8074252136752137,
    "mean_actual_parameter_L2_at3": 0.05056854631757608,
    "corr1_fit_objective_n": 117,
    "corr1_fit_objective_mean_change": -0.059910929901649985,
    "corr1_fit_objective_decreases": 117,
    "corr1_fit_KL_n": 117,
    "corr1_fit_KL_mean_change": -0.06021199081954914,
    "corr1_fit_KL_decreases": 117,
    "corr1_heldout_C_KL_n": 66,
    "corr1_heldout_C_KL_mean_change": -0.041421137651388905,
    "corr1_heldout_C_KL_decreases": 44,
    "corr3_fit_objective_n": 117,
    "corr3_fit_objective_mean_change": -0.1283795469961179,
    "corr3_fit_objective_decreases": 117,
    "corr3_fit_KL_n": 117,
    "corr3_fit_KL_mean_change": -0.12958605943940535,
    "corr3_fit_KL_decreases": 117,
    "corr3_heldout_C_KL_n": 66,
    "corr3_heldout_C_KL_mean_change": -0.1263113674186619,
    "corr3_heldout_C_KL_decreases": 52,
    "wrong3_fit_objective_n": 117,
    "wrong3_fit_objective_mean_change": 0.19260864033989594,
    "wrong3_fit_objective_decreases": 13,
    "wrong3_fit_KL_n": 117,
    "wrong3_fit_KL_mean_change": 0.1890396177686418,
    "wrong3_fit_KL_decreases": 13,
    "wrong3_heldout_C_KL_n": 66,
    "wrong3_heldout_C_KL_mean_change": 0.09684233569197567,
    "wrong3_heldout_C_KL_decreases": 25,
    "random3_fit_objective_n": 117,
    "random3_fit_objective_mean_change": 0.0005628369911859825,
    "random3_fit_objective_decreases": 37,
    "random3_fit_KL_n": 117,
    "random3_fit_KL_mean_change": 0.0004685079924624943,
    "random3_fit_KL_decreases": 48,
    "random3_heldout_C_KL_n": 66,
    "random3_heldout_C_KL_mean_change": -0.0004888859521407871,
    "random3_heldout_C_KL_decreases": 31,
    "free3_fit_objective_n": 117,
    "free3_fit_objective_mean_change": -0.20489425771064754,
    "free3_fit_objective_decreases": 117,
    "free3_fit_KL_n": 117,
    "free3_fit_KL_mean_change": -0.20491512486466568,
    "free3_fit_KL_decreases": 117,
    "free3_heldout_C_KL_n": 66,
    "free3_heldout_C_KL_mean_change": -0.08755186222602375,
    "free3_heldout_C_KL_decreases": 54
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
    "gain_pp": -0.16534495539436658,
    "better": 6,
    "worse": 10,
    "same": 752,
    "query_changes": 24,
    "largest_positive_single_input_pp": 0.04186852011328779,
    "largest_negative_single_input_pp": -0.07486691872986709
  },
  {
    "cohort": "DEV",
    "method": "corr3_F",
    "control": "headmean_context",
    "n": 768,
    "gain_pp": 0.007202767299225854,
    "better": 16,
    "worse": 12,
    "same": 740,
    "query_changes": 49,
    "largest_positive_single_input_pp": 0.08265650351809108,
    "largest_negative_single_input_pp": -0.0893579070700335
  },
  {
    "cohort": "DEV",
    "method": "corr3_F",
    "control": "robust_context",
    "n": 768,
    "gain_pp": -0.1993436873539468,
    "better": 9,
    "worse": 11,
    "same": 748,
    "query_changes": 30,
    "largest_positive_single_input_pp": 0.04164013147968911,
    "largest_negative_single_input_pp": -0.07486691872986709
  },
  {
    "cohort": "DEV",
    "method": "corr3_F",
    "control": "direct_F",
    "n": 768,
    "gain_pp": 0.17917623108110903,
    "better": 20,
    "worse": 21,
    "same": 727,
    "query_changes": 78,
    "largest_positive_single_input_pp": 0.0903288215055313,
    "largest_negative_single_input_pp": -0.0893579070700335
  },
  {
    "cohort": "DEV",
    "method": "corr3_F",
    "control": "free3_F",
    "n": 768,
    "gain_pp": 0.1340645904437346,
    "better": 15,
    "worse": 14,
    "same": 739,
    "query_changes": 45,
    "largest_positive_single_input_pp": 0.06864713716211572,
    "largest_negative_single_input_pp": -0.07486691872986709
  },
  {
    "cohort": "DEV",
    "method": "corr3_F",
    "control": "wrong3_F",
    "n": 768,
    "gain_pp": 0.4304752581287319,
    "better": 20,
    "worse": 11,
    "same": 737,
    "query_changes": 56,
    "largest_positive_single_input_pp": 0.0943659118829029,
    "largest_negative_single_input_pp": -0.08713116465832955
  },
  {
    "cohort": "DEV",
    "method": "corr3_F",
    "control": "random3_F",
    "n": 768,
    "gain_pp": -0.1696098968865245,
    "better": 5,
    "worse": 10,
    "same": 753,
    "query_changes": 23,
    "largest_positive_single_input_pp": 0.04186852011328779,
    "largest_negative_single_input_pp": -0.07486691872986709
  },
  {
    "cohort": "EXP",
    "method": "corr3_F",
    "control": "mean_context",
    "n": 1536,
    "gain_pp": -0.06444645957989703,
    "better": 10,
    "worse": 12,
    "same": 1514,
    "query_changes": 39,
    "largest_positive_single_input_pp": 0.020871559304455223,
    "largest_negative_single_input_pp": -0.03803720941588019
  },
  {
    "cohort": "EXP",
    "method": "corr3_F",
    "control": "headmean_context",
    "n": 1536,
    "gain_pp": -0.09843627636840736,
    "better": 28,
    "worse": 25,
    "same": 1483,
    "query_changes": 101,
    "largest_positive_single_input_pp": 0.04503449860016842,
    "largest_negative_single_input_pp": -0.041491082509298706
  },
  {
    "cohort": "EXP",
    "method": "corr3_F",
    "control": "robust_context",
    "n": 1536,
    "gain_pp": -0.00011913867685528721,
    "better": 17,
    "worse": 15,
    "same": 1504,
    "query_changes": 58,
    "largest_positive_single_input_pp": 0.051036878261435326,
    "largest_negative_single_input_pp": -0.038206062815087545
  },
  {
    "cohort": "EXP",
    "method": "corr3_F",
    "control": "direct_F",
    "n": 1536,
    "gain_pp": 0.19119850323053028,
    "better": 55,
    "worse": 40,
    "same": 1441,
    "query_changes": 181,
    "largest_positive_single_input_pp": 0.05308933266522159,
    "largest_negative_single_input_pp": -0.042297834552076204
  },
  {
    "cohort": "EXP",
    "method": "corr3_F",
    "control": "free3_F",
    "n": 1536,
    "gain_pp": 0.056050618115002705,
    "better": 23,
    "worse": 22,
    "same": 1491,
    "query_changes": 81,
    "largest_positive_single_input_pp": 0.051036878261435326,
    "largest_negative_single_input_pp": -0.041478933278917746
  },
  {
    "cohort": "EXP",
    "method": "corr3_F",
    "control": "wrong3_F",
    "n": 1536,
    "gain_pp": 0.35752477130128973,
    "better": 36,
    "worse": 19,
    "same": 1481,
    "query_changes": 102,
    "largest_positive_single_input_pp": 0.051036878261435326,
    "largest_negative_single_input_pp": -0.03871057962902432
  },
  {
    "cohort": "EXP",
    "method": "corr3_F",
    "control": "random3_F",
    "n": 1536,
    "gain_pp": -0.08531801888435227,
    "better": 9,
    "worse": 12,
    "same": 1515,
    "query_changes": 39,
    "largest_positive_single_input_pp": 0.01919630008067376,
    "largest_negative_single_input_pp": -0.03803720941588019
  }
]
```

## NATIVE_AUDIT

```json
{
  "state": "PASS_ALL_NATIVE_TEACHER_DIRECT_READOUTS_HEADS_PARAMETER_AND_FREE_FIELD_ARITHMETIC",
  "n": 2304,
  "aliases": 154,
  "active_AB_inputs": 786,
  "CPU_head_calls": 82856,
  "max_probability_error": 2.9114893704274003e-06,
  "max_state_arithmetic_error": 1.862645149230957e-09,
  "max_objective_error": 6.9844736452218115e-06,
  "full_decoder_gradients_independently_CPU_recomputed": false,
  "gradient_validation": "engineering finite differences plus all step arithmetic/head/teacher/endpoint objectives; no all-CPU decoder backprop claim",
  "GT_used": false,
  "FINAL_access": false
}
```

## SCALAR_AUDIT

```json
{
  "state": "PASS_ALL_TASK_PLATFORM_SCENE_PAIRS_POSTSTRATIFICATION",
  "n": 2304,
  "arms": 27,
  "task_values": 62208,
  "new_task_values": 36864,
  "reused_task_values": 25344,
  "aliases": 154,
  "FINAL_access": false
}
```

## GEOMETRY_AUDIT

```json
{
  "n": 2304,
  "new_readouts": 36864,
  "reused_readouts": 25344,
  "geometry_max_error": 6.661338147750939e-16
}
```

## CPU_PROBE

```json
{
  "state": "PASS6_REAL_TEACHER_CONSENSUS_STATE_FREE_FIELD_HEAD_AND_OBJECTIVE_SCHEMA",
  "n": 6,
  "checks": [
    {
      "source": "waymo",
      "cohort": "DEV",
      "maxp": 7.457041799607822e-07,
      "maxd": 9.313225746154785e-10,
      "maxloss": 1.437645388915887e-07,
      "heads": 36
    },
    {
      "source": "waymo",
      "cohort": "EXP",
      "maxp": 8.064937326857091e-07,
      "maxd": 9.313225746154785e-10,
      "maxloss": 6.99871627318771e-08,
      "heads": 36
    },
    {
      "source": "drone",
      "cohort": "DEV",
      "maxp": 5.434157333283895e-07,
      "maxd": 4.656612873077393e-10,
      "maxloss": 2.299423448315352e-07,
      "heads": 44
    },
    {
      "source": "drone",
      "cohort": "EXP",
      "maxp": 8.310497623120749e-07,
      "maxd": 9.313225746154785e-10,
      "maxloss": 6.829886644021954e-07,
      "heads": 44
    },
    {
      "source": "quad",
      "cohort": "DEV",
      "maxp": 5.599100491027187e-07,
      "maxd": 4.656612873077393e-10,
      "maxloss": 3.802368505612108e-07,
      "heads": 44
    },
    {
      "source": "quad",
      "cohort": "EXP",
      "maxp": 9.904039802721698e-07,
      "maxd": 9.313225746154785e-10,
      "maxloss": 1.8856481174278628e-06,
      "heads": 44
    }
  ],
  "GT_used": false
}
```

## 本轮裁决

预设开发门失败，停止这份固定注册语义KL短步pointAdapter配方。不扩大步数、温度、层位置或事后选择平台/一步作为主项。

判断以全部方向和同信息对照的选框净收益为主。辅助KL降低与真实写入参数不算任务成功；GT源域能力诊断也不能替无标签信号背书。全部好坏案例、拒绝、无B/C和保护回退均保留，未看私有图像认证的身份保持UNKNOWN。

梯度验证来自工程中心有限差分，以及全量CPU步进/自由特征场/teacher/head/端点目标复算，不声称全量CPU重做decoder反向传播。27臂共62208任务值，36864新读出、25344复用既有对照；全部新预测封存后才读取任务GT，并独立检查框IoU、场景及门槛。

seconds是多臂共享缓存的研究开销，不等于单方法端到端延迟。原始媒体、原句、ID、坐标、query索引、特征、权重及梯度不公开。
