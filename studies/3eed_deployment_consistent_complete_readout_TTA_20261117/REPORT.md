# R127 — 拟合／部署一致的完整读出 TTA：固定四臂因子对照

## 实际执行范围

保留三份epoch100/source-prepared R91模型、原A256框和全部decoder位置。与R124/R125相同，native foundation的pointAdapter Up和early RGB gateP从零开始，训练过的Down固定；只更新Up权重/偏置4896参数，其他网络完全冻结。本轮不新增源域训练或专家。

本轮只有一项有限实验：分别拟合A/B或拟合注册聚合FAB，交叉原生或完整读出监督。主项预先固定fab_full_F，三步共享预算；不按GT挑一步、平台、读出或个案。两列同时改变对应冻结教师与学生读出，不能称为只改一个KL实现位置。

## 适应与最终计算路径

FAB在支持有效处等权平均A/B点特征，不支持处保留A。主要输出始终是FAB，不含C。所有完整读出都采用同一份既有source-A RGB late与source-A竞争残差，数值停止梯度；但RGB向量相加后的归一化、完整文字匹配、竞争残差后的softmax进入真实更新目标。没有使用旧ABC或含C的上下文生成教师／梯度。

原生教师为冻结原生A/B候选概率均值，完整教师为冻结完整A/B概率均值。每项最小化对应教师到学生的KL，加单位权重归一化点特征保持。分别拟合A/B时对两观测平均，拟合FAB时只用FAB。仅三步归一化SGD，半径上限为原source整个点Adapter范数的1%，固定9级二分Armijo，按该臂无标签目标接受或回退。R126只验收已有原生梯度；本轮确实从完整计算图重新生成梯度。

这轮修正优化／部署输入和读出的一致性，仍使用同模型自共识；没有认证候选身份，也没有实现支持条件化监督或独立RGB专家。冻结竞争残差只提供分数偏置，经softmax影响更新；没有声称梯度通过其内部重新绑定计算。

## 对照与数据边界

四个实际更新臂：ab_native、ab_full、fab_native、fab_full。错配教师在256候选上作无固定点循环置换，保留概率集合与熵；随机参数方向匹配主项累计范数。零更新完整FAB、冻结完整AB直接概率平均、原生AB共识加零步完整读出差分均使用相同A/B信息。全部新臂固定后才解码C；C仅作描述性迁移检查，不参与梯度、回溯、配置选择或准入门。

DEV768与EXP1536均历史曝光，共六方向2304输入、154保护别名；别名不加载特征，统一沿用prepared_A回退，其他无B证据样本零更新，全部仍在分母。B有效资格沿用R125；不按目标标签筛选。C是注册到A锚点的另一份观测描述子，使用A候选与A评分，不代表新场景GT。FINAL未访问。

历史legacy_ABC_context额外使用C，仅作旧结果定位，不作为本轮相同信息胜负门。不得拿过去ABC+rF的绝对值与本轮FAB+rA直接归因；四臂共享新零步FAB+rA。

## 总体结果

| 范围 | 方法 | mIoU% | Acc25% | Acc50% |
|---|---|---:|---:|---:|
| DEV | original_source | 16.811383 | 30.989583 | 14.973958 |
| DEV | prepared_A | 16.228652 | 29.947917 | 12.630208 |
| DEV | legacy_ABC_context | 17.349002 | 31.901042 | 14.453125 |
| DEV | zero_F | 17.279671 | 32.161458 | 14.322917 |
| DEV | zero_C | 16.492047 | 30.729167 | 13.671875 |
| DEV | direct_full_F | 17.192255 | 32.031250 | 14.322917 |
| DEV | direct_full_C | 17.192255 | 32.031250 | 14.322917 |
| DEV | direct_native_bridge_F | 16.764115 | 31.250000 | 13.671875 |
| DEV | direct_native_bridge_C | 16.750924 | 31.250000 | 13.671875 |
| DEV | ab_native_F | 17.353195 | 32.421875 | 14.453125 |
| DEV | ab_native_C | 16.417686 | 30.338542 | 13.541667 |
| DEV | ab_full_F | 17.268993 | 32.291667 | 14.322917 |
| DEV | ab_full_C | 16.567009 | 30.598958 | 13.802083 |
| DEV | fab_native_F | 16.925362 | 31.380208 | 14.062500 |
| DEV | fab_native_C | 16.167003 | 29.947917 | 13.411458 |
| DEV | fab_full_F | 16.954268 | 31.510417 | 14.062500 |
| DEV | fab_full_C | 16.164069 | 29.947917 | 13.411458 |
| DEV | wrong_fab_full_F | 16.860426 | 31.250000 | 14.062500 |
| DEV | wrong_fab_full_C | 16.065208 | 30.078125 | 13.411458 |
| DEV | random_F | 17.279671 | 32.161458 | 14.322917 |
| DEV | random_C | 16.440500 | 30.598958 | 13.671875 |
| EXP | original_source | 16.144619 | 30.598958 | 12.434896 |
| EXP | prepared_A | 16.251067 | 30.598958 | 11.783854 |
| EXP | legacy_ABC_context | 16.499566 | 31.119792 | 12.174479 |
| EXP | zero_F | 16.418423 | 30.989583 | 12.044271 |
| EXP | zero_C | 16.204012 | 30.859375 | 12.174479 |
| EXP | direct_full_F | 16.255326 | 30.729167 | 11.848958 |
| EXP | direct_full_C | 16.255326 | 30.729167 | 11.848958 |
| EXP | direct_native_bridge_F | 16.237094 | 30.794271 | 11.979167 |
| EXP | direct_native_bridge_C | 16.201074 | 30.664062 | 11.653646 |
| EXP | ab_native_F | 16.432006 | 31.054688 | 11.979167 |
| EXP | ab_native_C | 16.232547 | 30.859375 | 12.239583 |
| EXP | ab_full_F | 16.342038 | 30.924479 | 11.914062 |
| EXP | ab_full_C | 16.214160 | 30.794271 | 12.174479 |
| EXP | fab_native_F | 16.416510 | 31.119792 | 12.109375 |
| EXP | fab_native_C | 16.206870 | 30.989583 | 12.239583 |
| EXP | fab_full_F | 16.370022 | 30.924479 | 12.109375 |
| EXP | fab_full_C | 16.143559 | 30.859375 | 12.174479 |
| EXP | wrong_fab_full_F | 16.062342 | 30.078125 | 11.848958 |
| EXP | wrong_fab_full_C | 16.144317 | 30.729167 | 12.304688 |
| EXP | random_F | 16.418423 | 30.989583 | 12.044271 |
| EXP | random_C | 16.215218 | 30.859375 | 12.174479 |

## 六方向结果

| 范围 | source→target | 方法 | mIoU% | Acc25% | Acc50% |
|---|---|---|---:|---:|---:|
| DEV | waymo→drone | original_source | 16.856414 | 31.250000 | 14.843750 |
| DEV | waymo→drone | zero_F | 18.193613 | 33.593750 | 14.843750 |
| DEV | waymo→drone | direct_full_F | 18.088883 | 33.593750 | 14.843750 |
| DEV | waymo→drone | ab_native_F | 18.172477 | 33.593750 | 14.843750 |
| DEV | waymo→drone | ab_full_F | 18.172477 | 33.593750 | 14.843750 |
| DEV | waymo→drone | fab_native_F | 17.785957 | 32.812500 | 14.062500 |
| DEV | waymo→drone | fab_full_F | 17.513606 | 32.031250 | 14.062500 |
| DEV | waymo→drone | wrong_fab_full_F | 17.908923 | 32.812500 | 14.843750 |
| DEV | waymo→drone | random_F | 18.193613 | 33.593750 | 14.843750 |
| DEV | waymo→quad | original_source | 33.347552 | 55.468750 | 41.406250 |
| DEV | waymo→quad | zero_F | 32.566884 | 53.906250 | 40.625000 |
| DEV | waymo→quad | direct_full_F | 33.026598 | 55.468750 | 41.406250 |
| DEV | waymo→quad | ab_native_F | 33.050554 | 55.468750 | 41.406250 |
| DEV | waymo→quad | ab_full_F | 32.545337 | 54.687500 | 40.625000 |
| DEV | waymo→quad | fab_native_F | 33.048366 | 55.468750 | 41.406250 |
| DEV | waymo→quad | fab_full_F | 32.543149 | 54.687500 | 40.625000 |
| DEV | waymo→quad | wrong_fab_full_F | 31.725723 | 52.343750 | 39.843750 |
| DEV | waymo→quad | random_F | 32.566884 | 53.906250 | 40.625000 |
| EXP | waymo→drone | original_source | 14.204359 | 28.515625 | 10.546875 |
| EXP | waymo→drone | zero_F | 15.090863 | 29.687500 | 10.937500 |
| EXP | waymo→drone | direct_full_F | 14.652080 | 28.515625 | 10.546875 |
| EXP | waymo→drone | ab_native_F | 14.996330 | 29.296875 | 10.937500 |
| EXP | waymo→drone | ab_full_F | 14.996330 | 29.296875 | 10.937500 |
| EXP | waymo→drone | fab_native_F | 14.751815 | 28.906250 | 10.546875 |
| EXP | waymo→drone | fab_full_F | 14.652305 | 28.515625 | 10.546875 |
| EXP | waymo→drone | wrong_fab_full_F | 14.980633 | 29.296875 | 10.937500 |
| EXP | waymo→drone | random_F | 15.090863 | 29.687500 | 10.937500 |
| EXP | waymo→quad | original_source | 29.479514 | 49.218750 | 35.937500 |
| EXP | waymo→quad | zero_F | 30.897831 | 51.562500 | 36.328125 |
| EXP | waymo→quad | direct_full_F | 30.832510 | 51.562500 | 36.328125 |
| EXP | waymo→quad | ab_native_F | 30.847909 | 51.953125 | 36.328125 |
| EXP | waymo→quad | ab_full_F | 30.902328 | 52.343750 | 36.328125 |
| EXP | waymo→quad | fab_native_F | 30.994541 | 51.953125 | 36.718750 |
| EXP | waymo→quad | fab_full_F | 30.850023 | 51.562500 | 36.328125 |
| EXP | waymo→quad | wrong_fab_full_F | 30.193169 | 50.390625 | 35.156250 |
| EXP | waymo→quad | random_F | 30.897831 | 51.562500 | 36.328125 |
| DEV | drone→waymo | original_source | 14.437008 | 32.031250 | 8.593750 |
| DEV | drone→waymo | zero_F | 14.711135 | 32.812500 | 8.593750 |
| DEV | drone→waymo | direct_full_F | 14.394515 | 31.250000 | 8.593750 |
| DEV | drone→waymo | ab_native_F | 14.711135 | 32.812500 | 8.593750 |
| DEV | drone→waymo | ab_full_F | 14.711135 | 32.812500 | 8.593750 |
| DEV | drone→waymo | fab_native_F | 14.368402 | 31.250000 | 8.593750 |
| DEV | drone→waymo | fab_full_F | 14.368402 | 31.250000 | 8.593750 |
| DEV | drone→waymo | wrong_fab_full_F | 13.774421 | 30.468750 | 8.593750 |
| DEV | drone→waymo | random_F | 14.711135 | 32.812500 | 8.593750 |
| DEV | drone→quad | original_source | 16.838098 | 34.375000 | 8.593750 |
| DEV | drone→quad | zero_F | 16.631282 | 34.375000 | 6.250000 |
| DEV | drone→quad | direct_full_F | 16.387994 | 33.593750 | 5.468750 |
| DEV | drone→quad | ab_native_F | 16.798358 | 34.375000 | 6.250000 |
| DEV | drone→quad | ab_full_F | 16.798358 | 34.375000 | 6.250000 |
| DEV | drone→quad | fab_native_F | 15.312831 | 31.250000 | 4.687500 |
| DEV | drone→quad | fab_full_F | 16.047930 | 32.812500 | 5.468750 |
| DEV | drone→quad | wrong_fab_full_F | 16.326215 | 34.375000 | 5.468750 |
| DEV | drone→quad | random_F | 16.631282 | 34.375000 | 6.250000 |
| EXP | drone→waymo | original_source | 14.644590 | 33.593750 | 4.296875 |
| EXP | drone→waymo | zero_F | 14.718295 | 33.593750 | 3.906250 |
| EXP | drone→waymo | direct_full_F | 14.409893 | 33.203125 | 3.515625 |
| EXP | drone→waymo | ab_native_F | 14.718295 | 33.593750 | 3.906250 |
| EXP | drone→waymo | ab_full_F | 14.565064 | 33.203125 | 3.906250 |
| EXP | drone→waymo | fab_native_F | 14.429101 | 33.203125 | 3.906250 |
| EXP | drone→waymo | fab_full_F | 14.332351 | 32.812500 | 3.906250 |
| EXP | drone→waymo | wrong_fab_full_F | 14.416654 | 32.812500 | 3.906250 |
| EXP | drone→waymo | random_F | 14.718295 | 33.593750 | 3.906250 |
| EXP | drone→quad | original_source | 17.434916 | 34.765625 | 6.250000 |
| EXP | drone→quad | zero_F | 16.943861 | 33.984375 | 3.906250 |
| EXP | drone→quad | direct_full_F | 16.732885 | 33.593750 | 3.906250 |
| EXP | drone→quad | ab_native_F | 17.209469 | 34.765625 | 3.515625 |
| EXP | drone→quad | ab_full_F | 16.980111 | 34.375000 | 3.515625 |
| EXP | drone→quad | fab_native_F | 17.350779 | 35.546875 | 4.296875 |
| EXP | drone→quad | fab_full_F | 17.215145 | 35.156250 | 4.296875 |
| EXP | drone→quad | wrong_fab_full_F | 16.499573 | 32.031250 | 4.296875 |
| EXP | drone→quad | random_F | 16.943861 | 33.984375 | 3.906250 |
| DEV | quad→waymo | original_source | 8.510254 | 14.843750 | 6.250000 |
| DEV | quad→waymo | zero_F | 9.725965 | 17.187500 | 7.031250 |
| DEV | quad→waymo | direct_full_F | 9.607077 | 17.187500 | 7.031250 |
| DEV | quad→waymo | ab_native_F | 9.725965 | 17.187500 | 7.031250 |
| DEV | quad→waymo | ab_full_F | 9.725965 | 17.187500 | 7.031250 |
| DEV | quad→waymo | fab_native_F | 9.607077 | 17.187500 | 7.031250 |
| DEV | quad→waymo | fab_full_F | 9.607077 | 17.187500 | 7.031250 |
| DEV | quad→waymo | wrong_fab_full_F | 9.725965 | 17.187500 | 7.031250 |
| DEV | quad→waymo | random_F | 9.725965 | 17.187500 | 7.031250 |
| DEV | quad→drone | original_source | 10.878972 | 17.968750 | 10.156250 |
| DEV | quad→drone | zero_F | 11.849145 | 21.093750 | 8.593750 |
| DEV | quad→drone | direct_full_F | 11.648463 | 21.093750 | 8.593750 |
| DEV | quad→drone | ab_native_F | 11.660684 | 21.093750 | 8.593750 |
| DEV | quad→drone | ab_full_F | 11.660684 | 21.093750 | 8.593750 |
| DEV | quad→drone | fab_native_F | 11.429540 | 20.312500 | 8.593750 |
| DEV | quad→drone | fab_full_F | 11.645445 | 21.093750 | 8.593750 |
| DEV | quad→drone | wrong_fab_full_F | 11.701313 | 20.312500 | 8.593750 |
| DEV | quad→drone | random_F | 11.849145 | 21.093750 | 8.593750 |
| EXP | quad→waymo | original_source | 12.257366 | 24.218750 | 9.375000 |
| EXP | quad→waymo | zero_F | 11.330441 | 21.875000 | 8.593750 |
| EXP | quad→waymo | direct_full_F | 11.713898 | 22.656250 | 8.984375 |
| EXP | quad→waymo | ab_native_F | 11.287670 | 21.484375 | 8.593750 |
| EXP | quad→waymo | ab_full_F | 11.287670 | 21.484375 | 8.593750 |
| EXP | quad→waymo | fab_native_F | 11.731730 | 22.656250 | 8.984375 |
| EXP | quad→waymo | fab_full_F | 11.732305 | 22.656250 | 8.984375 |
| EXP | quad→waymo | wrong_fab_full_F | 10.888829 | 20.703125 | 8.203125 |
| EXP | quad→waymo | random_F | 11.330441 | 21.875000 | 8.593750 |
| EXP | quad→drone | original_source | 8.846971 | 13.281250 | 8.203125 |
| EXP | quad→drone | zero_F | 9.529250 | 15.234375 | 8.593750 |
| EXP | quad→drone | direct_full_F | 9.190694 | 14.843750 | 7.812500 |
| EXP | quad→drone | ab_native_F | 9.532363 | 15.234375 | 8.593750 |
| EXP | quad→drone | ab_full_F | 9.320725 | 14.843750 | 8.203125 |
| EXP | quad→drone | fab_native_F | 9.241094 | 14.453125 | 8.203125 |
| EXP | quad→drone | fab_full_F | 9.438000 | 14.843750 | 8.593750 |
| EXP | quad→drone | wrong_fab_full_F | 9.395195 | 15.234375 | 8.593750 |
| EXP | quad→drone | random_F | 9.529250 | 15.234375 | 8.593750 |

## 目标平台与场景频数控制

| 范围 | 目标 | 方法 | mIoU% |
|---|---|---|---:|
| DEV | waymo | zero_F | 12.218550 |
| DEV | waymo | direct_full_F | 12.000796 |
| DEV | waymo | fab_full_F | 11.987740 |
| DEV | drone | zero_F | 15.021379 |
| DEV | drone | direct_full_F | 14.868673 |
| DEV | drone | fab_full_F | 14.579526 |
| DEV | quad | zero_F | 24.599083 |
| DEV | quad | direct_full_F | 24.707296 |
| DEV | quad | fab_full_F | 24.295539 |
| EXP | waymo | zero_F | 13.024368 |
| EXP | waymo | direct_full_F | 13.061895 |
| EXP | waymo | fab_full_F | 13.032328 |
| EXP | drone | zero_F | 12.310056 |
| EXP | drone | direct_full_F | 11.921387 |
| EXP | drone | fab_full_F | 12.045153 |
| EXP | quad | zero_F | 23.920846 |
| EXP | quad | direct_full_F | 23.782698 |
| EXP | quad | fab_full_F | 24.032584 |

| 范围 | 方法 | 场景支持覆盖下限 | 频数后分层mIoU% |
|---|---|---:|---:|
| DEV | zero_F | 1.000000 | 12.517328 |
| DEV | direct_full_F | 1.000000 | 12.456845 |
| DEV | direct_native_bridge_F | 1.000000 | 12.232523 |
| DEV | ab_native_F | 1.000000 | 12.616910 |
| DEV | ab_full_F | 1.000000 | 12.575685 |
| DEV | fab_native_F | 1.000000 | 12.307772 |
| DEV | fab_full_F | 1.000000 | 12.325310 |
| DEV | wrong_fab_full_F | 1.000000 | 12.063669 |
| DEV | random_F | 1.000000 | 12.517328 |
| EXP | zero_F | 0.998149 | 12.880964 |
| EXP | direct_full_F | 0.998149 | 12.754992 |
| EXP | direct_native_bridge_F | 0.998149 | 12.658772 |
| EXP | ab_native_F | 0.998149 | 12.797185 |
| EXP | ab_full_F | 0.998149 | 12.824965 |
| EXP | fab_native_F | 0.998149 | 12.873111 |
| EXP | fab_full_F | 0.998149 | 12.810650 |
| EXP | wrong_fab_full_F | 0.998149 | 12.601510 |
| EXP | random_F | 0.998149 | 12.880964 |

## FACTORIAL_EFFECTS

```json
[
  {
    "cohort": "DEV",
    "complete_minus_native_at_AB_pp": -0.08420280778933531,
    "complete_minus_native_at_FAB_pp": 0.0289060761223503,
    "FAB_minus_AB_at_native_pp": -0.42783331423234117,
    "FAB_minus_AB_at_complete_pp": -0.31472443032065556,
    "two_factor_difference_pp": 0.11310888391168561
  },
  {
    "cohort": "EXP",
    "complete_minus_native_at_AB_pp": -0.08996792320310476,
    "complete_minus_native_at_FAB_pp": -0.04648849692974366,
    "FAB_minus_AB_at_native_pp": -0.015495851465743726,
    "FAB_minus_AB_at_complete_pp": 0.027983574807617373,
    "two_factor_difference_pp": 0.0434794262733611
  }
]
```

## DECISION

```json
{
  "primary": "fab_full_F",
  "cohorts": {
    "DEV": {
      "mIoU_percent": 16.95426823386659,
      "gains_pp": {
        "zero_F": -0.3254025026411931,
        "direct_full_F": -0.2379867796605737,
        "direct_native_bridge_F": 0.1901530593006946,
        "wrong_fab_full_F": 0.09384173611140767,
        "random_F": -0.3254025026411931
      },
      "poststratified_mIoU_percent": 12.325310064863018,
      "poststratified_gains_pp": {
        "zero_F": -0.19201805533725924,
        "direct_full_F": -0.13153542715600963,
        "direct_native_bridge_F": 0.09278723330153049,
        "wrong_fab_full_F": 0.26164147013846595,
        "random_F": -0.19201805533725924
      },
      "passes_all_prelocked_controls": false,
      "unused_C_available_macro_gain_pp": -0.27712900160207604,
      "unused_C_gate": false
    },
    "EXP": {
      "mIoU_percent": 16.370021604572877,
      "gains_pp": {
        "zero_F": -0.04840167209275492,
        "direct_full_F": 0.11469514525808222,
        "direct_native_bridge_F": 0.13292730389986573,
        "wrong_fab_full_F": 0.3076793468505947,
        "random_F": -0.04840167209275492
      },
      "poststratified_mIoU_percent": 12.810650439328,
      "poststratified_gains_pp": {
        "zero_F": -0.07031387665309374,
        "direct_full_F": 0.05565873526351872,
        "direct_native_bridge_F": 0.1518779405076316,
        "wrong_fab_full_F": 0.20914010853397258,
        "random_F": -0.07031387665309374
      },
      "passes_all_prelocked_controls": false,
      "unused_C_available_macro_gain_pp": 0.32727597056341423,
      "unused_C_gate": true
    }
  },
  "FINAL_access": false,
  "posthoc_primary_switch": false,
  "development_gate_pass": false,
  "C_is_descriptive_not_configuration_gate": true,
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
    "gain_pp": 0.0,
    "better": 0,
    "worse": 0
  },
  {
    "cohort": "DEV",
    "source": "drone",
    "target": "waymo",
    "n": 17,
    "gain_pp": 0.0,
    "better": 0,
    "worse": 0
  },
  {
    "cohort": "DEV",
    "source": "quad",
    "target": "drone",
    "n": 10,
    "gain_pp": -0.24047916451033546,
    "better": 0,
    "worse": 1
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
    "gain_pp": -1.9077035362049206,
    "better": 0,
    "worse": 2
  },
  {
    "cohort": "EXP",
    "source": "drone",
    "target": "quad",
    "n": 109,
    "gain_pp": 0.253365986543003,
    "better": 3,
    "worse": 3
  },
  {
    "cohort": "EXP",
    "source": "drone",
    "target": "waymo",
    "n": 40,
    "gain_pp": 0.11522948147476351,
    "better": 2,
    "worse": 0
  },
  {
    "cohort": "EXP",
    "source": "quad",
    "target": "drone",
    "n": 26,
    "gain_pp": 0.9849495344933528,
    "better": 1,
    "worse": 0
  },
  {
    "cohort": "EXP",
    "source": "quad",
    "target": "waymo",
    "n": 40,
    "gain_pp": 0.0,
    "better": 0,
    "worse": 0
  },
  {
    "cohort": "EXP",
    "source": "waymo",
    "target": "drone",
    "n": 26,
    "gain_pp": 0.4917662854442747,
    "better": 1,
    "worse": 0
  },
  {
    "cohort": "EXP",
    "source": "waymo",
    "target": "quad",
    "n": 109,
    "gain_pp": 0.11834453542509157,
    "better": 2,
    "worse": 2
  }
]
```

## CONCENTRATION

```json
[
  {
    "cohort": "DEV",
    "control": "zero_F",
    "n": 768,
    "gain_pp": -0.3254025026411932,
    "better": 5,
    "worse": 12,
    "same": 751,
    "query_changes": 32,
    "largest_positive_single_input_pp": 0.07486691872986709,
    "largest_negative_single_input_pp": -0.08057482712365807
  },
  {
    "cohort": "DEV",
    "control": "direct_full_F",
    "n": 768,
    "gain_pp": -0.2379867796605766,
    "better": 5,
    "worse": 8,
    "same": 755,
    "query_changes": 27,
    "largest_positive_single_input_pp": 0.053708010201132854,
    "largest_negative_single_input_pp": -0.08057482712365807
  },
  {
    "cohort": "DEV",
    "control": "direct_native_bridge_F",
    "n": 768,
    "gain_pp": 0.1901530593006945,
    "better": 16,
    "worse": 12,
    "same": 740,
    "query_changes": 50,
    "largest_positive_single_input_pp": 0.09097962206390664,
    "largest_negative_single_input_pp": -0.07381118895591267
  },
  {
    "cohort": "DEV",
    "control": "ab_native_F",
    "n": 768,
    "gain_pp": -0.39892723810999164,
    "better": 9,
    "worse": 13,
    "same": 746,
    "query_changes": 34,
    "largest_positive_single_input_pp": 0.07486691872986709,
    "largest_negative_single_input_pp": -0.08420280778933663
  },
  {
    "cohort": "DEV",
    "control": "ab_full_F",
    "n": 768,
    "gain_pp": -0.31472443032065495,
    "better": 9,
    "worse": 12,
    "same": 747,
    "query_changes": 33,
    "largest_positive_single_input_pp": 0.07486691872986709,
    "largest_negative_single_input_pp": -0.08057482712365807
  },
  {
    "cohort": "DEV",
    "control": "fab_native_F",
    "n": 768,
    "gain_pp": 0.028906076122351576,
    "better": 4,
    "worse": 3,
    "same": 761,
    "query_changes": 13,
    "largest_positive_single_input_pp": 0.08265650351809108,
    "largest_negative_single_input_pp": -0.08420280778933663
  },
  {
    "cohort": "DEV",
    "control": "wrong_fab_full_F",
    "n": 768,
    "gain_pp": 0.09384173611140595,
    "better": 20,
    "worse": 19,
    "same": 729,
    "query_changes": 63,
    "largest_positive_single_input_pp": 0.0903288215055313,
    "largest_negative_single_input_pp": -0.08713116465832955
  },
  {
    "cohort": "DEV",
    "control": "random_F",
    "n": 768,
    "gain_pp": -0.3254025026411932,
    "better": 5,
    "worse": 12,
    "same": 751,
    "query_changes": 32,
    "largest_positive_single_input_pp": 0.07486691872986709,
    "largest_negative_single_input_pp": -0.08057482712365807
  },
  {
    "cohort": "EXP",
    "control": "zero_F",
    "n": 1536,
    "gain_pp": -0.04840167209275278,
    "better": 13,
    "worse": 17,
    "same": 1506,
    "query_changes": 53,
    "largest_positive_single_input_pp": 0.04272420444360952,
    "largest_negative_single_input_pp": -0.04503449860016842
  },
  {
    "cohort": "EXP",
    "control": "direct_full_F",
    "n": 1536,
    "gain_pp": 0.1146951452580824,
    "better": 17,
    "worse": 11,
    "same": 1508,
    "query_changes": 60,
    "largest_positive_single_input_pp": 0.032817740081825496,
    "largest_negative_single_input_pp": -0.022605595067400355
  },
  {
    "cohort": "EXP",
    "control": "direct_native_bridge_F",
    "n": 1536,
    "gain_pp": 0.13292730389986757,
    "better": 37,
    "worse": 26,
    "same": 1473,
    "query_changes": 111,
    "largest_positive_single_input_pp": 0.05308933266522159,
    "largest_negative_single_input_pp": -0.046892743444216685
  },
  {
    "cohort": "EXP",
    "control": "ab_native_F",
    "n": 1536,
    "gain_pp": -0.06198434839548509,
    "better": 16,
    "worse": 20,
    "same": 1500,
    "query_changes": 73,
    "largest_positive_single_input_pp": 0.04272420444360952,
    "largest_negative_single_input_pp": -0.04503449860016842
  },
  {
    "cohort": "EXP",
    "control": "ab_full_F",
    "n": 1536,
    "gain_pp": 0.027983574807620606,
    "better": 16,
    "worse": 18,
    "same": 1502,
    "query_changes": 68,
    "largest_positive_single_input_pp": 0.04272420444360952,
    "largest_negative_single_input_pp": -0.04503449860016842
  },
  {
    "cohort": "EXP",
    "control": "fab_native_F",
    "n": 1536,
    "gain_pp": -0.04648849692973704,
    "better": 8,
    "worse": 6,
    "same": 1522,
    "query_changes": 28,
    "largest_positive_single_input_pp": 0.041491082509298706,
    "largest_negative_single_input_pp": -0.03826176004744391
  },
  {
    "cohort": "EXP",
    "control": "wrong_fab_full_F",
    "n": 1536,
    "gain_pp": 0.3076793468505956,
    "better": 36,
    "worse": 24,
    "same": 1476,
    "query_changes": 109,
    "largest_positive_single_input_pp": 0.051036878261435326,
    "largest_negative_single_input_pp": -0.04503449860016842
  },
  {
    "cohort": "EXP",
    "control": "random_F",
    "n": 1536,
    "gain_pp": -0.04840167209275278,
    "better": 13,
    "worse": 17,
    "same": 1506,
    "query_changes": 53,
    "largest_positive_single_input_pp": 0.04272420444360952,
    "largest_negative_single_input_pp": -0.04503449860016842
  }
]
```

## COMPONENT_SUMMARY

```json
[
  {
    "cohort": "DEV",
    "arm": "ab_native",
    "active": 258,
    "objective_decreased": 258,
    "accepted_steps": 774,
    "mean_parameter_L2": 0.051555470166626094,
    "mean_objective_change": -0.10386297798469958
  },
  {
    "cohort": "DEV",
    "arm": "ab_full",
    "active": 258,
    "objective_decreased": 258,
    "accepted_steps": 774,
    "mean_parameter_L2": 0.056120285633414764,
    "mean_objective_change": -0.06309479453033291
  },
  {
    "cohort": "DEV",
    "arm": "fab_native",
    "active": 258,
    "objective_decreased": 258,
    "accepted_steps": 774,
    "mean_parameter_L2": 0.03692155008442686,
    "mean_objective_change": -0.0943709873376335
  },
  {
    "cohort": "DEV",
    "arm": "fab_full",
    "active": 258,
    "objective_decreased": 258,
    "accepted_steps": 774,
    "mean_parameter_L2": 0.035750024787659304,
    "mean_objective_change": -0.060485279478422194
  },
  {
    "cohort": "DEV",
    "arm": "wrong_fab_full",
    "active": 258,
    "objective_decreased": 258,
    "accepted_steps": 774,
    "mean_parameter_L2": 0.10387464829923482,
    "mean_objective_change": -0.6670804407804171
  },
  {
    "cohort": "EXP",
    "arm": "ab_native",
    "active": 528,
    "objective_decreased": 528,
    "accepted_steps": 1582,
    "mean_parameter_L2": 0.05289061302327821,
    "mean_objective_change": -0.11941756047944398
  },
  {
    "cohort": "EXP",
    "arm": "ab_full",
    "active": 528,
    "objective_decreased": 528,
    "accepted_steps": 1583,
    "mean_parameter_L2": 0.056671359857138025,
    "mean_objective_change": -0.07019168957656288
  },
  {
    "cohort": "EXP",
    "arm": "fab_native",
    "active": 528,
    "objective_decreased": 528,
    "accepted_steps": 1580,
    "mean_parameter_L2": 0.03839340877149249,
    "mean_objective_change": -0.11662394333078861
  },
  {
    "cohort": "EXP",
    "arm": "fab_full",
    "active": 528,
    "objective_decreased": 528,
    "accepted_steps": 1582,
    "mean_parameter_L2": 0.036925395482103766,
    "mean_objective_change": -0.07192099078285756
  },
  {
    "cohort": "EXP",
    "arm": "wrong_fab_full",
    "active": 528,
    "objective_decreased": 528,
    "accepted_steps": 1584,
    "mean_parameter_L2": 0.10313227110543842,
    "mean_objective_change": -0.7061950740940853
  }
]
```

## ENGINEERING

```json
{
  "state": "PASS24_REAL_FACTORIAL_FULL_PATH_ACTUAL_POINT_ADAPTER_WRITES",
  "n": 24,
  "checks": [
    {
      "cohort": "DEV",
      "source": "waymo",
      "target": "drone",
      "active": true,
      "scope4896": true,
      "finite_difference": [
        {
          "kind": "fab_full",
          "directional_derivative": 0.2227759153593592,
          "autograd_directional": 0.22286589443683624,
          "relative_error": 0.00040373641603804974,
          "epsilon": 3.961693306040547e-05
        }
      ],
      "seconds": 2.8600124200165737
    },
    {
      "cohort": "DEV",
      "source": "waymo",
      "target": "drone",
      "active": false,
      "scope4896": true,
      "finite_difference": [],
      "seconds": 1.288764650002122
    },
    {
      "cohort": "DEV",
      "source": "waymo",
      "target": "quad",
      "active": true,
      "scope4896": true,
      "finite_difference": [
        {
          "kind": "fab_full",
          "directional_derivative": 0.4929296277040045,
          "autograd_directional": 0.4930095374584198,
          "relative_error": 0.00016208561568048098,
          "epsilon": 3.961693306040547e-05
        }
      ],
      "seconds": 2.2326104779785965
    },
    {
      "cohort": "DEV",
      "source": "waymo",
      "target": "quad",
      "active": false,
      "scope4896": true,
      "finite_difference": [],
      "seconds": 1.2893352569954004
    },
    {
      "cohort": "EXP",
      "source": "waymo",
      "target": "drone",
      "active": true,
      "scope4896": true,
      "finite_difference": [
        {
          "kind": "fab_full",
          "directional_derivative": 0.545425549032188,
          "autograd_directional": 0.5441453456878662,
          "relative_error": 0.002352686381436268,
          "epsilon": 3.961693306040547e-05
        }
      ],
      "seconds": 2.2031570749823004
    },
    {
      "cohort": "EXP",
      "source": "waymo",
      "target": "drone",
      "active": false,
      "scope4896": true,
      "finite_difference": [],
      "seconds": 1.21674752200488
    },
    {
      "cohort": "EXP",
      "source": "waymo",
      "target": "quad",
      "active": true,
      "scope4896": true,
      "finite_difference": [
        {
          "kind": "fab_full",
          "directional_derivative": 0.622114480773333,
          "autograd_directional": 0.6235873699188232,
          "relative_error": 0.0023619611565930208,
          "epsilon": 3.961693306040547e-05
        }
      ],
      "seconds": 2.2726012470084243
    },
    {
      "cohort": "EXP",
      "source": "waymo",
      "target": "quad",
      "active": false,
      "scope4896": true,
      "finite_difference": [],
      "seconds": 1.174572421994526
    },
    {
      "cohort": "DEV",
      "source": "drone",
      "target": "waymo",
      "active": true,
      "scope4896": true,
      "finite_difference": [
        {
          "kind": "fab_full",
          "directional_derivative": 0.5695326904816708,
          "autograd_directional": 0.568294107913971,
          "relative_error": 0.00217947458974415,
          "epsilon": 3.2139483012895676e-05
        }
      ],
      "seconds": 2.5043948689999525
    },
    {
      "cohort": "DEV",
      "source": "drone",
      "target": "waymo",
      "active": false,
      "scope4896": true,
      "finite_difference": [],
      "seconds": 1.2450922309944872
    },
    {
      "cohort": "DEV",
      "source": "drone",
      "target": "quad",
      "active": true,
      "scope4896": true,
      "finite_difference": [
        {
          "kind": "fab_full",
          "directional_derivative": 4.427759064326477,
          "autograd_directional": 4.416622638702393,
          "relative_error": 0.002521479993897759,
          "epsilon": 3.2139483012895676e-05
        }
      ],
      "seconds": 2.2408619000052568
    },
    {
      "cohort": "DEV",
      "source": "drone",
      "target": "quad",
      "active": false,
      "scope4896": true,
      "finite_difference": [],
      "seconds": 1.3003306850150693
    },
    {
      "cohort": "EXP",
      "source": "drone",
      "target": "waymo",
      "active": true,
      "scope4896": true,
      "finite_difference": [
        {
          "kind": "fab_full",
          "directional_derivative": 5.227043200450632,
          "autograd_directional": 5.222901821136475,
          "relative_error": 0.000792926893130158,
          "epsilon": 3.2139483012895676e-05
        }
      ],
      "seconds": 2.059413882991066
    },
    {
      "cohort": "EXP",
      "source": "drone",
      "target": "waymo",
      "active": false,
      "scope4896": true,
      "finite_difference": [],
      "seconds": 1.0273948929971084
    },
    {
      "cohort": "EXP",
      "source": "drone",
      "target": "quad",
      "active": true,
      "scope4896": true,
      "finite_difference": [
        {
          "kind": "fab_full",
          "directional_derivative": 5.254666067410018,
          "autograd_directional": 5.258267879486084,
          "relative_error": 0.0006849807120168465,
          "epsilon": 3.2139483012895676e-05
        }
      ],
      "seconds": 1.861352818988962
    },
    {
      "cohort": "EXP",
      "source": "drone",
      "target": "quad",
      "active": false,
      "scope4896": true,
      "finite_difference": [],
      "seconds": 1.03646963898791
    },
    {
      "cohort": "DEV",
      "source": "quad",
      "target": "waymo",
      "active": true,
      "scope4896": true,
      "finite_difference": [
        {
          "kind": "fab_full",
          "directional_derivative": 0.11095920976470922,
          "autograd_directional": 0.11088702082633972,
          "relative_error": 0.0006510134173642305,
          "epsilon": 3.2651687631922875e-05
        }
      ],
      "seconds": 2.695908727007918
    },
    {
      "cohort": "DEV",
      "source": "quad",
      "target": "waymo",
      "active": false,
      "scope4896": true,
      "finite_difference": [],
      "seconds": 1.0536509259836748
    },
    {
      "cohort": "DEV",
      "source": "quad",
      "target": "drone",
      "active": true,
      "scope4896": true,
      "finite_difference": [
        {
          "kind": "fab_full",
          "directional_derivative": 9.072952585620882,
          "autograd_directional": 9.034322738647461,
          "relative_error": 0.004275898491889027,
          "epsilon": 3.2651687631922875e-05
        }
      ],
      "seconds": 1.8619486219831742
    },
    {
      "cohort": "DEV",
      "source": "quad",
      "target": "drone",
      "active": false,
      "scope4896": true,
      "finite_difference": [],
      "seconds": 1.0698603280179668
    },
    {
      "cohort": "EXP",
      "source": "quad",
      "target": "waymo",
      "active": true,
      "scope4896": true,
      "finite_difference": [
        {
          "kind": "fab_full",
          "directional_derivative": 19.15842432012307,
          "autograd_directional": 19.192171096801758,
          "relative_error": 0.0017583616000750534,
          "epsilon": 3.2651687631922875e-05
        }
      ],
      "seconds": 1.8570619380043354
    },
    {
      "cohort": "EXP",
      "source": "quad",
      "target": "waymo",
      "active": false,
      "scope4896": true,
      "finite_difference": [],
      "seconds": 0.9882540419930592
    },
    {
      "cohort": "EXP",
      "source": "quad",
      "target": "drone",
      "active": true,
      "scope4896": true,
      "finite_difference": [
        {
          "kind": "fab_full",
          "directional_derivative": 13.171742159710298,
          "autograd_directional": 13.17520523071289,
          "relative_error": 0.0002628475945497558,
          "epsilon": 3.2651687631922875e-05
        }
      ],
      "seconds": 1.9571247790008783
    },
    {
      "cohort": "EXP",
      "source": "quad",
      "target": "drone",
      "active": false,
      "scope4896": true,
      "finite_difference": [],
      "seconds": 1.0686451060173567
    }
  ]
}
```

## CPU_PROBE

```json
{
  "state": "PASS_INDEPENDENT_CPU_TEACHERS_ALL_READOUTS_OBJECTIVES_STATE_ARITHMETIC_PROVIDERS",
  "n": 6,
  "aliases": 0,
  "active": 6,
  "CPU_head_calls": 162,
  "max_probability_error": 8.650861968440005e-07,
  "max_loss_error": 1.3106932774675784e-06,
  "max_state_error": 9.313225746154785e-10,
  "GT_used": false,
  "FINAL_access": false,
  "full_decoder_backprop_independently_CPU_recomputed": false
}
```

## NATIVE_AUDIT

```json
{
  "state": "PASS_INDEPENDENT_CPU_TEACHERS_ALL_READOUTS_OBJECTIVES_STATE_ARITHMETIC_PROVIDERS",
  "n": 2304,
  "aliases": 154,
  "active": 786,
  "CPU_head_calls": 58050,
  "max_probability_error": 2.9114893704274003e-06,
  "max_loss_error": 5.899247291552001e-06,
  "max_state_error": 1.862645149230957e-09,
  "GT_used": false,
  "FINAL_access": false,
  "full_decoder_backprop_independently_CPU_recomputed": false
}
```

## SCALAR_AUDIT

```json
{
  "state": "PASS_ALL_TASK_SCENE_PAIRED_POSTSTRATIFIED_GATES",
  "n": 2304,
  "aliases": 154,
  "arms": 21,
  "task_values": 48384,
  "new_task_values": 41472,
  "reused_task_values": 6912,
  "FINAL_access": false
}
```

## GEOMETRY_AUDIT

```json
{
  "n": 2304,
  "new_readouts": 41472,
  "reused_readouts": 6912,
  "geometry_max_error": 6.661338147750939e-16
}
```

## 裁决

主版本未通过预先锁定的同信息开发门。不采纳这份固定完整共识配方，不因其他臂较好而事后替换主方法。

输入与读出对齐若有效，仍不等于教师知道目标身份；辅助损失下降也不等于任务改善。C单独报告，不用于挑参数。主指标保留全部输入、方向、拒绝及负例，场景后分层只用旧全量清单频数，EXP缺少4个Waymo场景仅共同支持。

独立CPU审计复算教师、全部head读出、端点保持／KL、参数步进、provider及旧教师h/U桥接；工程样本用中心有限差分检查主完整梯度、实际权重写入与等价点注入。未声称全量CPU重做decoder反向传播。所有预测封存及无标签审计后才评分，独立几何IoU与标量审计另列。

seconds是多臂共享缓存的研究开销，不能包装为单方法端到端延迟。私有图像、原句、数据身份、坐标、query索引、特征、权重及梯度不公开。


## 后验：教师直接错误与更新引入的损伤分开

以下在主结果曝光后，使用已评分且已封存的全部行作描述性分解；没有新模型、IoU计算、标签筛选、方法或参数改变。匿名公开CSV的独立归约与私有评分行一致。物理实例身份不能仅由IoU排序认证。

- DEV12条误改中，6条的直接完整教师读出本来就低于零步；另外6条的直接教师与零步选择相同query，参数更新仍使IoU下降。
- EXP17条误改中，14条直接教师已经低于零步；另外3条直接教师与零步选择相同query，更新仍造成损伤。
- DEV5条改善只有1条超过同信息直接教师的IoU；EXP13条改善有7条超过直接教师。因而不能说参数更新完全没有额外纠错，也不能用这些局部收益抵消总体下降的裁决。
- DEV3/5、EXP7/13的改善至多恢复原source已有IoU，其他改善确实高于原source选框；这些都是指标层解释，不认证新实例知识。

私有图册19页（8个good、11个failure）系统保留每方向最大改善／损伤，另有全部匿名逐例结果。实际查看CASE008/009/010/011：CASE008与CASE010中，直接完整教师和TTA都从0提高到约.575/.656；CASE009中零步、直接教师及AB-native维持约.619，主TTA切到空间分离的框、IoU为0；CASE011中直接教师和TTA均从约.563降至0。这些只作机制解释，不据此补个案规则。投影点是框内种子几何，不是attention或实例mask；缓存图分辨率有限，未独立认证物体身份保持UNKNOWN。

因此，主要问题应保留为两部分：共识本身的任务可靠性，以及全局软分布拟合向最终Top1决策的转化。把输入和完整计算路径对齐，并不足以解决这两部分；也不能仅凭完整读出梯度负增量认定RGB或竞争结构整体无效。完整教师固定sourceA上下文，本轮没有检验重新计算上下文或支持条件化监督。
