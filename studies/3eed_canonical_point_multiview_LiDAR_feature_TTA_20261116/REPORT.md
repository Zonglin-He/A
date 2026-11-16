# R106 — 实际邻观测点特征的原生匹配适应

保留三份source100、source准备的R91双向结构、原A RGB/标定图、A文字坐标视角、1024个A空间seed和原A256。改变的是3D观测来源：真实注册B/C点云在A坐标中经过同一个冻结PointNet，再把原生pre-language特征对齐到固定A位置。未知区域保留A特征，没有新增专家网络或目标身份真值。

每个额外输入16384点；同帧randomA对照使用两次独立16384点采样和同样前向次数。额外真实点不必比同帧重采样更好，新的体素可能来自噪声或配准误差。额外RGB bank没有与本轮混用。

## 观测前提

{
  "state": "SENSOR_INCREMENTAL_SUPPORT_MEASURED_NOT_TASK_EFFICACY",
  "groups": [
    {
      "platform": "waymo",
      "pairs": 40,
      "candidate_sets": 84,
      "fraction_B_within_25cm_of_A": {
        "n": 40,
        "mean": 0.8675146141395711,
        "median": 0.8921767617253107,
        "p90": 0.9551480894069614
      },
      "fraction_B_new_10to25cm_surface_shell": {
        "n": 40,
        "mean": 0.1272172449403563,
        "median": 0.10595683045188628,
        "p90": 0.25498813936579556
      },
      "B_new_voxel_fraction_over_sampleA": {
        "n": 40,
        "mean": 0.607587636610435,
        "median": 0.5940618599349106,
        "p90": 0.8015456301910661
      },
      "sameA_random_new_voxel_fraction": {
        "n": 40,
        "mean": 0.18197099400446376,
        "median": 0.17603298056842775,
        "p90": 0.3027694917236942
      },
      "B_support_where_A_sample_empty": {
        "n": 84,
        "mean": 2.6547619047619047,
        "median": 1.0,
        "p90": 7.0
      },
      "sameA_support_where_A_sample_empty": {
        "n": 84,
        "mean": 1.1547619047619047,
        "median": 1.0,
        "p90": 3.0
      },
      "fraction_candidates_B_increased_count": {
        "n": 84,
        "mean": 0.36997767857142855,
        "median": 0.373046875,
        "p90": 0.5171875
      },
      "fraction_candidates_sameA_increased_count": {
        "n": 84,
        "mean": 0.4128999255952381,
        "median": 0.421875,
        "p90": 0.52734375
      }
    },
    {
      "platform": "drone",
      "pairs": 50,
      "candidate_sets": 100,
      "fraction_B_within_25cm_of_A": {
        "n": 50,
        "mean": 0.8534931080662509,
        "median": 0.8595672764532414,
        "p90": 0.9281268467061214
      },
      "fraction_B_new_10to25cm_surface_shell": {
        "n": 50,
        "mean": 0.3777172948150615,
        "median": 0.36483315800799354,
        "p90": 0.5028742743045178
      },
      "B_new_voxel_fraction_over_sampleA": {
        "n": 50,
        "mean": 0.8253031030433803,
        "median": 0.8336696953170164,
        "p90": 0.8869953106515651
      },
      "sameA_random_new_voxel_fraction": {
        "n": 50,
        "mean": 0.0551517045453672,
        "median": 0.06093415170493166,
        "p90": 0.09257798208450496
      },
      "B_support_where_A_sample_empty": {
        "n": 100,
        "mean": 2.63,
        "median": 2.0,
        "p90": 6.1000000000000085
      },
      "sameA_support_where_A_sample_empty": {
        "n": 100,
        "mean": 0.51,
        "median": 0.0,
        "p90": 2.0
      },
      "fraction_candidates_B_increased_count": {
        "n": 100,
        "mean": 0.43171875,
        "median": 0.421875,
        "p90": 0.5570312500000002
      },
      "fraction_candidates_sameA_increased_count": {
        "n": 100,
        "mean": 0.4413671875,
        "median": 0.439453125,
        "p90": 0.5554687500000001
      }
    },
    {
      "platform": "quad",
      "pairs": 153,
      "candidate_sets": 306,
      "fraction_B_within_25cm_of_A": {
        "n": 153,
        "mean": 0.9008146697528878,
        "median": 0.9098715596330276,
        "p90": 0.9845083165147192
      },
      "fraction_B_new_10to25cm_surface_shell": {
        "n": 153,
        "mean": 0.2066011612266099,
        "median": 0.21533694810224632,
        "p90": 0.33477519054287014
      },
      "B_new_voxel_fraction_over_sampleA": {
        "n": 153,
        "mean": 0.7166497129197883,
        "median": 0.7724494593323931,
        "p90": 0.8755931853338531
      },
      "sameA_random_new_voxel_fraction": {
        "n": 153,
        "mean": 0.19080858871898862,
        "median": 0.19727626459143968,
        "p90": 0.21763712629244705
      },
      "B_support_where_A_sample_empty": {
        "n": 306,
        "mean": 2.1209150326797386,
        "median": 2.0,
        "p90": 5.0
      },
      "sameA_support_where_A_sample_empty": {
        "n": 306,
        "mean": 1.2843137254901962,
        "median": 1.0,
        "p90": 3.0
      },
      "fraction_candidates_B_increased_count": {
        "n": 306,
        "mean": 0.45159313725490197,
        "median": 0.4453125,
        "p90": 0.55859375
      },
      "fraction_candidates_sameA_increased_count": {
        "n": 306,
        "mean": 0.47232434640522875,
        "median": 0.46875,
        "p90": 0.55078125
      }
    }
  ],
  "no_new_model_or_GT": true,
  "FINAL_access": false,
  "all_geometrically_supported_pairs_retained": true
}

## 固定方法与实施

3NN反距离插值；需要A位置与B实测回波距离≤.25m、第三邻seed≤1.6m（实际source SA2尺度）。已知零距离自对应使用one-hot，避免依赖无关邻点。首个工程检查暴露原半径条件误拒孤立自对应，attempt完整保留；在任何正式模型/任务评分之前按预定自对应含义修复，不改变非零距离插值。12条真实原生验证通过自对应、未知回退、原始A输出、实际写参数后完整前向和Jacobian。

q=(p_A+p_B)/2来自同一冻结source准备模型；在A/B两视图拟合相同停止梯度目标，沿用18496参数的末端匹配投影和L2-LM10×6预算。各视图完整竞争残差冻结，因为匹配投影不改变它。C的PointNet/神经预测在所有拟合及验收后计算。C不足时记录拟合成本，最终TTA读出回到相同信息的冻结版本。候选ID始终是A锚点，不跨独立query编号硬对应。

## 全体结果

| 范围 | 方法 | mIoU % | Acc25 % | Acc50 % |
|---|---|---:|---:|---:|
| all_six_equal | original_source | 16.811383 | 30.989583 | 14.973958 |
| all_six_equal | prepared_frozen | 16.228652 | 29.947917 | 12.630208 |
| all_six_equal | frozen_four | 16.249883 | 29.947917 | 12.890625 |
| all_six_equal | direct_FC | 16.241990 | 29.817708 | 12.760417 |
| all_six_equal | joint3_FC | 16.517235 | 30.338542 | 13.020833 |
| all_six_equal | real_frozen_B | 16.352068 | 30.729167 | 12.369792 |
| all_six_equal | real_frozen_C | 16.128363 | 29.557292 | 12.239583 |
| all_six_equal | real_frozen_ensemble | 16.501679 | 30.729167 | 12.500000 |
| all_six_equal | real_frozen_fusedABC | 16.703230 | 30.989583 | 13.151042 |
| all_six_equal | real_TTA_A | 16.693092 | 30.859375 | 13.151042 |
| all_six_equal | real_TTA_B | 16.477955 | 31.250000 | 12.630208 |
| all_six_equal | real_TTA_C | 16.313584 | 29.817708 | 12.760417 |
| all_six_equal | real_TTA_ensemble | 16.496185 | 30.468750 | 12.760417 |
| all_six_equal | real_TTA_fusedABC | 16.574958 | 30.598958 | 13.020833 |
| all_six_equal | randomA_frozen_B | 16.385317 | 30.208333 | 12.630208 |
| all_six_equal | randomA_frozen_C | 16.248944 | 30.078125 | 12.890625 |
| all_six_equal | randomA_frozen_ensemble | 16.406990 | 30.078125 | 13.020833 |
| all_six_equal | randomA_frozen_fusedABC | 16.560112 | 30.598958 | 13.151042 |
| all_six_equal | randomA_TTA_A | 16.318686 | 29.947917 | 12.760417 |
| all_six_equal | randomA_TTA_B | 16.556816 | 30.208333 | 12.760417 |
| all_six_equal | randomA_TTA_C | 16.162271 | 29.817708 | 12.630208 |
| all_six_equal | randomA_TTA_ensemble | 16.545273 | 30.468750 | 13.020833 |
| all_six_equal | randomA_TTA_fusedABC | 16.629515 | 30.729167 | 13.151042 |
| all_six_equal | wrong_frozen_B | 11.808381 | 21.614583 | 8.723958 |
| all_six_equal | wrong_frozen_C | 11.824656 | 21.484375 | 8.463542 |
| all_six_equal | wrong_frozen_ensemble | 15.026155 | 27.213542 | 12.239583 |
| all_six_equal | wrong_frozen_fusedABC | 13.477241 | 24.869792 | 9.895833 |
| all_six_equal | wrong_TTA_A | 16.071049 | 29.687500 | 12.630208 |
| all_six_equal | wrong_TTA_B | 12.549711 | 23.177083 | 9.114583 |
| all_six_equal | wrong_TTA_C | 11.915131 | 21.744792 | 8.333333 |
| all_six_equal | wrong_TTA_ensemble | 15.064358 | 27.473958 | 11.848958 |
| all_six_equal | wrong_TTA_fusedABC | 13.362302 | 24.609375 | 10.026042 |
| all_six_equal | teacher_shuffle_C | 15.983628 | 29.036458 | 12.500000 |
| all_six_equal | teacher_shuffle_fusedABC | 16.280349 | 29.817708 | 12.500000 |
| waymo | original_source | 11.473631 | 23.437500 | 7.421875 |
| waymo | prepared_frozen | 12.615516 | 26.171875 | 5.078125 |
| waymo | frozen_four | 12.546748 | 25.781250 | 5.468750 |
| waymo | direct_FC | 12.755842 | 25.390625 | 5.468750 |
| waymo | joint3_FC | 13.067711 | 26.953125 | 5.859375 |
| waymo | real_frozen_B | 12.944417 | 26.953125 | 5.468750 |
| waymo | real_frozen_C | 12.837875 | 26.562500 | 5.078125 |
| waymo | real_frozen_ensemble | 12.647714 | 26.562500 | 5.078125 |
| waymo | real_frozen_fusedABC | 12.569007 | 26.171875 | 5.078125 |
| waymo | real_TTA_A | 12.787913 | 26.562500 | 5.078125 |
| waymo | real_TTA_B | 12.928248 | 26.953125 | 5.468750 |
| waymo | real_TTA_C | 12.770512 | 26.562500 | 5.078125 |
| waymo | real_TTA_ensemble | 12.727878 | 26.562500 | 5.078125 |
| waymo | real_TTA_fusedABC | 12.663880 | 26.171875 | 5.078125 |
| waymo | randomA_frozen_B | 12.580748 | 26.562500 | 4.687500 |
| waymo | randomA_frozen_C | 12.414522 | 25.390625 | 5.078125 |
| waymo | randomA_frozen_ensemble | 12.307775 | 25.000000 | 5.078125 |
| waymo | randomA_frozen_fusedABC | 12.283274 | 25.000000 | 5.078125 |
| waymo | randomA_TTA_A | 12.607156 | 26.171875 | 5.078125 |
| waymo | randomA_TTA_B | 12.603138 | 26.171875 | 4.687500 |
| waymo | randomA_TTA_C | 12.265264 | 25.000000 | 5.078125 |
| waymo | randomA_TTA_ensemble | 12.358041 | 25.390625 | 5.078125 |
| waymo | randomA_TTA_fusedABC | 12.338104 | 25.390625 | 5.078125 |
| waymo | wrong_frozen_B | 12.276684 | 25.000000 | 5.078125 |
| waymo | wrong_frozen_C | 11.550253 | 23.437500 | 5.078125 |
| waymo | wrong_frozen_ensemble | 12.481666 | 25.390625 | 5.078125 |
| waymo | wrong_frozen_fusedABC | 12.491509 | 26.171875 | 5.078125 |
| waymo | wrong_TTA_A | 12.606481 | 25.781250 | 5.078125 |
| waymo | wrong_TTA_B | 12.673494 | 26.171875 | 5.078125 |
| waymo | wrong_TTA_C | 11.687919 | 23.828125 | 5.078125 |
| waymo | wrong_TTA_ensemble | 12.452954 | 25.000000 | 5.078125 |
| waymo | wrong_TTA_fusedABC | 12.325978 | 25.390625 | 5.078125 |
| waymo | teacher_shuffle_C | 12.732584 | 26.171875 | 5.078125 |
| waymo | teacher_shuffle_fusedABC | 12.723968 | 26.171875 | 5.078125 |
| drone | original_source | 13.867693 | 24.609375 | 12.500000 |
| drone | prepared_frozen | 14.071678 | 24.609375 | 11.718750 |
| drone | frozen_four | 13.894458 | 24.609375 | 11.718750 |
| drone | direct_FC | 13.844147 | 25.000000 | 11.718750 |
| drone | joint3_FC | 14.238577 | 24.609375 | 12.109375 |
| drone | real_frozen_B | 13.984780 | 25.000000 | 11.718750 |
| drone | real_frozen_C | 13.842086 | 23.828125 | 11.718750 |
| drone | real_frozen_ensemble | 14.117679 | 25.000000 | 11.718750 |
| drone | real_frozen_fusedABC | 14.029708 | 25.000000 | 11.718750 |
| drone | real_TTA_A | 14.059728 | 24.609375 | 11.718750 |
| drone | real_TTA_B | 13.992240 | 25.781250 | 11.718750 |
| drone | real_TTA_C | 13.743545 | 23.437500 | 11.718750 |
| drone | real_TTA_ensemble | 14.129121 | 25.000000 | 11.718750 |
| drone | real_TTA_fusedABC | 14.080314 | 25.000000 | 11.718750 |
| drone | randomA_frozen_B | 13.839173 | 23.437500 | 11.718750 |
| drone | randomA_frozen_C | 13.854397 | 24.218750 | 11.718750 |
| drone | randomA_frozen_ensemble | 14.130911 | 25.000000 | 11.718750 |
| drone | randomA_frozen_fusedABC | 14.287690 | 25.781250 | 11.718750 |
| drone | randomA_TTA_A | 13.977816 | 24.218750 | 11.718750 |
| drone | randomA_TTA_B | 13.844311 | 23.437500 | 11.718750 |
| drone | randomA_TTA_C | 13.873365 | 24.218750 | 11.718750 |
| drone | randomA_TTA_ensemble | 14.159680 | 25.000000 | 11.718750 |
| drone | randomA_TTA_fusedABC | 14.261923 | 25.781250 | 11.718750 |
| drone | wrong_frozen_B | 12.644732 | 21.875000 | 11.718750 |
| drone | wrong_frozen_C | 13.213696 | 23.046875 | 11.328125 |
| drone | wrong_frozen_ensemble | 13.367766 | 23.437500 | 11.718750 |
| drone | wrong_frozen_fusedABC | 12.821754 | 22.656250 | 11.718750 |
| drone | wrong_TTA_A | 13.910261 | 24.609375 | 11.718750 |
| drone | wrong_TTA_B | 12.645103 | 21.875000 | 11.718750 |
| drone | wrong_TTA_C | 13.345408 | 23.437500 | 11.328125 |
| drone | wrong_TTA_ensemble | 13.306131 | 23.828125 | 11.718750 |
| drone | wrong_TTA_fusedABC | 12.843616 | 22.656250 | 11.718750 |
| drone | teacher_shuffle_C | 13.788433 | 23.828125 | 11.718750 |
| drone | teacher_shuffle_fusedABC | 13.903656 | 24.218750 | 11.718750 |
| quad | original_source | 25.092825 | 44.921875 | 25.000000 |
| quad | prepared_frozen | 21.998762 | 39.062500 | 21.093750 |
| quad | frozen_four | 22.308444 | 39.453125 | 21.484375 |
| quad | direct_FC | 22.125981 | 39.062500 | 21.093750 |
| quad | joint3_FC | 22.245417 | 39.453125 | 21.093750 |
| quad | real_frozen_B | 22.127007 | 40.234375 | 19.921875 |
| quad | real_frozen_C | 21.705129 | 38.281250 | 19.921875 |
| quad | real_frozen_ensemble | 22.739645 | 40.625000 | 20.703125 |
| quad | real_frozen_fusedABC | 23.510976 | 41.796875 | 22.656250 |
| quad | real_TTA_A | 23.231633 | 41.406250 | 22.656250 |
| quad | real_TTA_B | 22.513377 | 41.015625 | 20.703125 |
| quad | real_TTA_C | 22.426695 | 39.453125 | 21.484375 |
| quad | real_TTA_ensemble | 22.631557 | 39.843750 | 21.484375 |
| quad | real_TTA_fusedABC | 22.980680 | 40.625000 | 22.265625 |
| quad | randomA_frozen_B | 22.736029 | 40.625000 | 21.484375 |
| quad | randomA_frozen_C | 22.477914 | 40.625000 | 21.875000 |
| quad | randomA_frozen_ensemble | 22.782285 | 40.234375 | 22.265625 |
| quad | randomA_frozen_fusedABC | 23.109373 | 41.015625 | 22.656250 |
| quad | randomA_TTA_A | 22.371085 | 39.453125 | 21.484375 |
| quad | randomA_TTA_B | 23.222999 | 41.015625 | 21.875000 |
| quad | randomA_TTA_C | 22.348183 | 40.234375 | 21.093750 |
| quad | randomA_TTA_ensemble | 23.118100 | 41.015625 | 22.265625 |
| quad | randomA_TTA_fusedABC | 23.288519 | 41.015625 | 22.656250 |
| quad | wrong_frozen_B | 10.503728 | 17.968750 | 9.375000 |
| quad | wrong_frozen_C | 10.710019 | 17.968750 | 8.984375 |
| quad | wrong_frozen_ensemble | 19.229034 | 32.812500 | 19.921875 |
| quad | wrong_frozen_fusedABC | 15.118459 | 25.781250 | 12.890625 |
| quad | wrong_TTA_A | 21.696406 | 38.671875 | 21.093750 |
| quad | wrong_TTA_B | 12.330538 | 21.484375 | 10.546875 |
| quad | wrong_TTA_C | 10.712067 | 17.968750 | 8.593750 |
| quad | wrong_TTA_ensemble | 19.433988 | 33.593750 | 18.750000 |
| quad | wrong_TTA_fusedABC | 14.917313 | 25.781250 | 13.281250 |
| quad | teacher_shuffle_C | 21.429868 | 37.109375 | 20.703125 |
| quad | teacher_shuffle_fusedABC | 22.213422 | 39.062500 | 20.703125 |

## 六方向改善与恶化

| Source | Target | 方法 | 对照 | Δ pp | 改善/恶化/不变 |
|---|---|---|---|---:|---:|
| waymo | drone | real_TTA_fusedABC | prepared_frozen | +0.483695 | 7/2/119 |
| waymo | drone | real_TTA_fusedABC | real_frozen_fusedABC | +0.080305 | 1/0/127 |
| waymo | drone | real_TTA_fusedABC | real_frozen_ensemble | -0.017309 | 2/2/124 |
| waymo | drone | real_TTA_fusedABC | randomA_TTA_fusedABC | +0.025621 | 6/4/118 |
| waymo | drone | real_TTA_fusedABC | wrong_TTA_fusedABC | +1.992086 | 13/4/111 |
| waymo | drone | real_TTA_fusedABC | teacher_shuffle_fusedABC | +0.353317 | 3/0/125 |
| waymo | drone | real_TTA_fusedABC | joint3_FC | +0.179411 | 11/7/110 |
| waymo | drone | real_frozen_fusedABC | randomA_frozen_fusedABC | -0.106218 | 4/4/120 |
| waymo | drone | real_frozen_ensemble | randomA_frozen_ensemble | -0.014623 | 3/2/123 |
| waymo | drone | real_TTA_C | real_frozen_C | -0.133435 | 1/2/125 |
| waymo | drone | randomA_TTA_fusedABC | randomA_frozen_fusedABC | -0.051533 | 1/1/126 |
| waymo | quad | real_TTA_fusedABC | prepared_frozen | +1.245537 | 7/5/116 |
| waymo | quad | real_TTA_fusedABC | real_frozen_fusedABC | -1.183173 | 1/5/122 |
| waymo | quad | real_TTA_fusedABC | real_frozen_ensemble | +0.309877 | 4/4/120 |
| waymo | quad | real_TTA_fusedABC | randomA_TTA_fusedABC | +0.397903 | 9/7/112 |
| waymo | quad | real_TTA_fusedABC | wrong_TTA_fusedABC | +11.314726 | 42/4/82 |
| waymo | quad | real_TTA_fusedABC | teacher_shuffle_fusedABC | +0.893596 | 6/4/118 |
| waymo | quad | real_TTA_fusedABC | joint3_FC | +1.179418 | 8/5/115 |
| waymo | quad | real_frozen_fusedABC | randomA_frozen_fusedABC | +1.611195 | 11/5/112 |
| waymo | quad | real_frozen_ensemble | randomA_frozen_ensemble | +0.564774 | 9/5/114 |
| waymo | quad | real_TTA_C | real_frozen_C | +1.716004 | 12/4/112 |
| waymo | quad | randomA_TTA_fusedABC | randomA_frozen_fusedABC | +0.030119 | 2/2/124 |
| drone | waymo | real_TTA_fusedABC | prepared_frozen | -0.141657 | 1/3/124 |
| drone | waymo | real_TTA_fusedABC | real_frozen_fusedABC | +0.160329 | 1/0/127 |
| drone | waymo | real_TTA_fusedABC | real_frozen_ensemble | +0.002915 | 1/1/126 |
| drone | waymo | real_TTA_fusedABC | randomA_TTA_fusedABC | +0.319865 | 5/2/121 |
| drone | waymo | real_TTA_fusedABC | wrong_TTA_fusedABC | +0.683474 | 7/4/117 |
| drone | waymo | real_TTA_fusedABC | teacher_shuffle_fusedABC | -0.120176 | 3/5/120 |
| drone | waymo | real_TTA_fusedABC | joint3_FC | -0.322602 | 2/6/120 |
| drone | waymo | real_frozen_fusedABC | randomA_frozen_fusedABC | +0.159536 | 4/2/122 |
| drone | waymo | real_frozen_ensemble | randomA_frozen_ensemble | +0.267949 | 2/2/124 |
| drone | waymo | real_TTA_C | real_frozen_C | +0.098597 | 1/1/126 |
| drone | waymo | randomA_TTA_fusedABC | randomA_frozen_fusedABC | +0.000000 | 0/0/128 |
| drone | quad | real_TTA_fusedABC | prepared_frozen | +0.718299 | 12/7/109 |
| drone | quad | real_TTA_fusedABC | real_frozen_fusedABC | +0.122581 | 6/5/117 |
| drone | quad | real_TTA_fusedABC | real_frozen_ensemble | +0.172194 | 10/5/113 |
| drone | quad | real_TTA_fusedABC | randomA_TTA_fusedABC | -1.013580 | 8/11/109 |
| drone | quad | real_TTA_fusedABC | wrong_TTA_fusedABC | +4.812007 | 33/14/81 |
| drone | quad | real_TTA_fusedABC | teacher_shuffle_fusedABC | +0.640920 | 10/5/113 |
| drone | quad | real_TTA_fusedABC | joint3_FC | +0.291107 | 12/6/110 |
| drone | quad | real_frozen_fusedABC | randomA_frozen_fusedABC | -0.807988 | 8/12/108 |
| drone | quad | real_frozen_ensemble | randomA_frozen_ensemble | -0.650054 | 9/13/106 |
| drone | quad | real_TTA_C | real_frozen_C | -0.272872 | 6/6/116 |
| drone | quad | randomA_TTA_fusedABC | randomA_frozen_fusedABC | +0.328173 | 4/5/119 |
| quad | waymo | real_TTA_fusedABC | prepared_frozen | +0.238386 | 1/0/127 |
| quad | waymo | real_TTA_fusedABC | real_frozen_fusedABC | +0.029418 | 1/0/127 |
| quad | waymo | real_TTA_fusedABC | real_frozen_ensemble | +0.029418 | 1/0/127 |
| quad | waymo | real_TTA_fusedABC | randomA_TTA_fusedABC | +0.331687 | 2/0/126 |
| quad | waymo | real_TTA_fusedABC | wrong_TTA_fusedABC | -0.007670 | 1/1/126 |
| quad | waymo | real_TTA_fusedABC | teacher_shuffle_fusedABC | +0.000000 | 0/0/128 |
| quad | waymo | real_TTA_fusedABC | joint3_FC | -0.485059 | 3/5/120 |
| quad | waymo | real_frozen_fusedABC | randomA_frozen_fusedABC | +0.411929 | 3/0/125 |
| quad | waymo | real_frozen_ensemble | randomA_frozen_ensemble | +0.411929 | 3/0/125 |
| quad | waymo | real_TTA_C | real_frozen_C | -0.233322 | 0/2/126 |
| quad | waymo | randomA_TTA_fusedABC | randomA_frozen_fusedABC | +0.109659 | 1/0/127 |
| quad | drone | real_TTA_fusedABC | prepared_frozen | -0.466423 | 4/7/117 |
| quad | drone | real_TTA_fusedABC | real_frozen_fusedABC | +0.020907 | 1/0/127 |
| quad | drone | real_TTA_fusedABC | real_frozen_ensemble | -0.057420 | 3/4/121 |
| quad | drone | real_TTA_fusedABC | randomA_TTA_fusedABC | -0.388839 | 4/5/119 |
| quad | drone | real_TTA_fusedABC | wrong_TTA_fusedABC | +0.481311 | 9/4/115 |
| quad | drone | real_TTA_fusedABC | teacher_shuffle_fusedABC | +0.000000 | 0/0/128 |
| quad | drone | real_TTA_fusedABC | joint3_FC | -0.495937 | 7/9/112 |
| quad | drone | real_frozen_fusedABC | randomA_frozen_fusedABC | -0.409745 | 3/5/120 |
| quad | drone | real_frozen_ensemble | randomA_frozen_ensemble | -0.011842 | 4/3/121 |
| quad | drone | real_TTA_C | real_frozen_C | -0.063649 | 2/2/124 |
| quad | drone | randomA_TTA_fusedABC | randomA_frozen_fusedABC | +0.000000 | 0/0/128 |

## 裁决

{
  "primary": "real_TTA_fusedABC",
  "mIoU_percent": 16.5749581665426,
  "gains_pp": {
    "prepared_frozen": 0.34630600735392036,
    "real_frozen_fusedABC": -0.1282721264162454,
    "real_frozen_ensemble": 0.07327898222197504,
    "randomA_TTA_fusedABC": -0.054557070170741184,
    "wrong_TTA_fusedABC": 3.2126558846379965,
    "teacher_shuffle_fusedABC": 0.29460946474850047,
    "joint3_FC": 0.05772311689432996
  },
  "passes_development_gates": false,
  "robust_adoption": false,
  "decision": "do_not_adopt_this_actual_LiDAR_interpolation_TTA_recipe",
  "n": 768,
  "no_experts": true,
  "source_retraining": false,
  "FINAL_access": false,
  "primary_not_switched": true
}

## 边界

全部768条保留，不能只报配准或更新成功子集。真正新增3D观测的贡献需胜过同帧重采样；参数更新需胜过同信息冻结特征融合和概率集成，也需检查错配。IoU变化不作为物体身份标签，案例只解释机制、不筛分母。不是Latte复现；共享场景、离线候选邻帧、历史开发暴露均限制泛化主张，见LITERATURE.md。FINAL未访问。若没过门槛，不通过扩大步数或调插值半径挽救。

## 已封存监督与最终读出的分解诊断

{
  "groups": [
    {
      "direction": "waymo_to_drone",
      "teacher_mean_AB_mIoU_percent_all_with_prepared_fallback": 17.626724552305884,
      "teacher_available_n": 28,
      "Top1_switches_vs_frozen_fusedABC": 3,
      "harmful_changes_teacher_supports_new": 0,
      "harmful_changes_teacher_opposes_new": 0,
      "beneficial_changes_teacher_supports_new": 0,
      "beneficial_changes_teacher_opposes_new": 1
    },
    {
      "direction": "waymo_to_quad",
      "teacher_mean_AB_mIoU_percent_all_with_prepared_fallback": 30.74404103652858,
      "teacher_available_n": 80,
      "Top1_switches_vs_frozen_fusedABC": 9,
      "harmful_changes_teacher_supports_new": 2,
      "harmful_changes_teacher_opposes_new": 3,
      "beneficial_changes_teacher_supports_new": 1,
      "beneficial_changes_teacher_opposes_new": 0
    },
    {
      "direction": "drone_to_waymo",
      "teacher_mean_AB_mIoU_percent_all_with_prepared_fallback": 14.9447720035553,
      "teacher_available_n": 21,
      "Top1_switches_vs_frozen_fusedABC": 3,
      "harmful_changes_teacher_supports_new": 0,
      "harmful_changes_teacher_opposes_new": 0,
      "beneficial_changes_teacher_supports_new": 0,
      "beneficial_changes_teacher_opposes_new": 1
    },
    {
      "direction": "drone_to_quad",
      "teacher_mean_AB_mIoU_percent_all_with_prepared_fallback": 14.543135354968495,
      "teacher_available_n": 80,
      "Top1_switches_vs_frozen_fusedABC": 16,
      "harmful_changes_teacher_supports_new": 2,
      "harmful_changes_teacher_opposes_new": 3,
      "beneficial_changes_teacher_supports_new": 2,
      "beneficial_changes_teacher_opposes_new": 4
    },
    {
      "direction": "quad_to_waymo",
      "teacher_mean_AB_mIoU_percent_all_with_prepared_fallback": 10.342729692568943,
      "teacher_available_n": 21,
      "Top1_switches_vs_frozen_fusedABC": 1,
      "harmful_changes_teacher_supports_new": 0,
      "harmful_changes_teacher_opposes_new": 0,
      "beneficial_changes_teacher_supports_new": 1,
      "beneficial_changes_teacher_opposes_new": 0
    },
    {
      "direction": "quad_to_drone",
      "teacher_mean_AB_mIoU_percent_all_with_prepared_fallback": 10.372314732081922,
      "teacher_available_n": 28,
      "Top1_switches_vs_frozen_fusedABC": 1,
      "harmful_changes_teacher_supports_new": 0,
      "harmful_changes_teacher_opposes_new": 0,
      "beneficial_changes_teacher_supports_new": 0,
      "beneficial_changes_teacher_opposes_new": 1
    }
  ],
  "postseal_descriptive_not_method_selection": true,
  "teacher_targets_already_sealed_before_GT": true,
  "candidate_changes_not_identity_truth": true
}

这里只评价正式预测前已封存的q目标，属于事后机制诊断；不按GT筛选、反转监督或选择逐样本输出。

## 收益来源与实际看图复核

[
  {
    "scope": "all",
    "n": 768,
    "frozen_fusion_better_than_A": 33,
    "frozen_fusion_worse_than_A": 26,
    "TTA_better_than_fusion": 11,
    "TTA_improvements_exceed_A": 6,
    "TTA_improvements_do_not_exceed_A": 5,
    "TTA_harms_fusion_improvement_over_A": 6,
    "TTA_harms_even_when_A_was_at_least_as_good_as_fusion": 4
  },
  {
    "scope": "waymo_to_drone",
    "n": 128,
    "frozen_fusion_better_than_A": 6,
    "frozen_fusion_worse_than_A": 3,
    "TTA_better_than_fusion": 1,
    "TTA_improvements_exceed_A": 1,
    "TTA_improvements_do_not_exceed_A": 0,
    "TTA_harms_fusion_improvement_over_A": 0,
    "TTA_harms_even_when_A_was_at_least_as_good_as_fusion": 0
  },
  {
    "scope": "waymo_to_quad",
    "n": 128,
    "frozen_fusion_better_than_A": 9,
    "frozen_fusion_worse_than_A": 5,
    "TTA_better_than_fusion": 1,
    "TTA_improvements_exceed_A": 0,
    "TTA_improvements_do_not_exceed_A": 1,
    "TTA_harms_fusion_improvement_over_A": 3,
    "TTA_harms_even_when_A_was_at_least_as_good_as_fusion": 2
  },
  {
    "scope": "drone_to_waymo",
    "n": 128,
    "frozen_fusion_better_than_A": 1,
    "frozen_fusion_worse_than_A": 3,
    "TTA_better_than_fusion": 1,
    "TTA_improvements_exceed_A": 0,
    "TTA_improvements_do_not_exceed_A": 1,
    "TTA_harms_fusion_improvement_over_A": 0,
    "TTA_harms_even_when_A_was_at_least_as_good_as_fusion": 0
  },
  {
    "scope": "drone_to_quad",
    "n": 128,
    "frozen_fusion_better_than_A": 13,
    "frozen_fusion_worse_than_A": 8,
    "TTA_better_than_fusion": 6,
    "TTA_improvements_exceed_A": 3,
    "TTA_improvements_do_not_exceed_A": 3,
    "TTA_harms_fusion_improvement_over_A": 3,
    "TTA_harms_even_when_A_was_at_least_as_good_as_fusion": 2
  },
  {
    "scope": "quad_to_waymo",
    "n": 128,
    "frozen_fusion_better_than_A": 1,
    "frozen_fusion_worse_than_A": 0,
    "TTA_better_than_fusion": 1,
    "TTA_improvements_exceed_A": 1,
    "TTA_improvements_do_not_exceed_A": 0,
    "TTA_harms_fusion_improvement_over_A": 0,
    "TTA_harms_even_when_A_was_at_least_as_good_as_fusion": 0
  },
  {
    "scope": "quad_to_drone",
    "n": 128,
    "frozen_fusion_better_than_A": 3,
    "frozen_fusion_worse_than_A": 7,
    "TTA_better_than_fusion": 1,
    "TTA_improvements_exceed_A": 1,
    "TTA_improvements_do_not_exceed_A": 0,
    "TTA_harms_fusion_improvement_over_A": 0,
    "TTA_harms_even_when_A_was_at_least_as_good_as_fusion": 0
  }
]

{
  "generated_pages": 8,
  "pages_actually_viewed": 4,
  "viewed_aliases": [
    "CASE0002",
    "CASE0003",
    "CASE0005",
    "CASE0006"
  ],
  "not_identity_gold_annotation": true,
  "unviewed_pages_remain_UNKNOWN": true,
  "observed_types": [
    "overlapping vehicle hypotheses",
    "visible different-person switch",
    "restoring a prepared selection after fusion damage",
    "erasing a frozen-fusion correction"
  ],
  "no_case_specific_rules_added": true
}

实际查看4/8页，其余保持UNKNOWN。存在不同人物之间的切换，不能把本轮变化全部归为重复框抖动；也存在只恢复原prepared选择的good case，不能全部称为新实例纠错。按完整总体指标裁决，不据个例筛选方法。冻结真实点融合的正增量值得保留，但相对原始3EED16.811383%仍有差距：冻结融合16.703230%、更新16.574958%，因此未建立优于原系统的稳定TTA。

## 补齐直接采用监督分布的同信息对照

正式34臂矩阵没有单列直接q读出。为完整评价“把监督写入参数是否有额外收益”，这里保留原矩阵，并从当时已封存的q、A/C预测构造固定补充读出；没有新模型调用或按GT选择权重。direct_all在B可用时直接采用q；direct_matched只在与正式TTA相同的B/C可用条件采用q，否则使用原冻结融合；direct_matched_C在相同条件将q与冻结C等权平均，否则使用原冻结ensemble。补充预测再次封存后才评分，属于已暴露开发集诊断，不是独立确认。

[
  {
    "arm": "teacher_direct_all",
    "mIoU_percent": 16.428952895334852,
    "Acc25_percent": 30.46875,
    "Acc50_percent": 12.760416666666666,
    "TTA_minus_direct_pp": 0.14600527120774603
  },
  {
    "arm": "teacher_direct_matched",
    "mIoU_percent": 16.494893611409946,
    "Acc25_percent": 30.598958333333332,
    "Acc50_percent": 12.890625,
    "TTA_minus_direct_pp": 0.08006455513265089
  },
  {
    "arm": "teacher_direct_matched_C",
    "mIoU_percent": 16.36329343606346,
    "Acc25_percent": 30.338541666666668,
    "Acc50_percent": 12.5,
    "TTA_minus_direct_pp": 0.2116647304791367
  }
]
