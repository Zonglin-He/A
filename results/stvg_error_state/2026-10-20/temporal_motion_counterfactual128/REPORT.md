# motion时间均值反事实：HC128冻结能力检查

每offset、每motion cell/channel沿T均值，仅改变有效motion输入；appearance/text/pos不改。同一Source原生融合、query/gate两pass重新运行。不是静态视频或纯先验；动作特征均值可能缩范数、混合前景背景并分布外。所有空间坐标冻结。

旧noun-caption quotient阴性保留；此处只换视觉输入干预，不复现官方VCD的图像噪声/文本token流程。Contrast=normalize(p0²/qMeanMotion)，PriorSquare控制泛化锐化，单位系数固定；没有新专家/训练/学生或OPD。

| 对照 | 完整v增量pp [CI] | 候选期望GTv增量pp [CI] |
|---|---|---|
| Contrast_minus_Native | -0.703678 [-2.199280991041901, 0.8350602886602365] | -1.086355 [-2.1831152584975215, -0.06701287684569938] |
| Contrast_minus_PriorSquare | -0.370338 [-1.9479246115278654, 1.2157892623430229] | -2.021553 [-3.1464440194766743, -0.934739306244956] |
| Contrast_minus_HiResAuxMix | -2.310112 [-4.163598854530627, -0.4148500004608768] | -0.896120 [-2.3166447802644, 0.49089923811907926] |
| MeanMotion_minus_Native | -0.988139 [-2.727993400411195, 0.8748700878825222] | -0.403526 [-1.6026727117591157, 0.8449054775657053] |

全部RC128旧曝光父query，空间三seed父内平均，TC109未选参。预测及prefix全封存后GT；MeanMotion原始输出只诊断，不事后晋升。新fusion encoder计算如实计费，未新RGB/专家观测。

```json
{
  "primary_hard": {
    "mean": -0.7036783701876285,
    "CI95": [
      -2.199280991041901,
      0.8350602886602365
    ],
    "parents": 128
  },
  "primary_expected": {
    "mean": -1.0863547800740458,
    "CI95": [
      -2.1831152584975215,
      -0.06701287684569938
    ],
    "parents": 128
  },
  "decision": "close_this_mean_motion_quotient_configuration",
  "not_OPD": true,
  "not_independent_confirmation": true,
  "no_coefficient_norm_mean_search": true,
  "goal_achieved": false
}
{
  "GPU_seconds": 89.24649882107042,
  "early_fusion_encoder_reencodes": 512,
  "full_native_prefix_replays": 256,
  "max_CDF_error": 2.872051554914101e-16,
  "new_Source_RGB_encoder": 0,
  "new_RGB": 0,
  "new_DINO": 0,
  "new_ReferDINO": 0,
  "network_training": 0,
  "spatial_updates": 0,
  "new_student_updates": 0
}
{
  "status": "pass",
  "native_encoder_prefix_final_time_exact_parents": 128,
  "independent_FP64_mean_values": 4555264,
  "mean_absolute_max_error": 9.5367431640625e-07,
  "mean_error_over_prelocked_bound_max": 0.02686965302773836,
  "actual_encoder_input_hashes": 256,
  "nonmotion_padding_pos_exact": true,
  "changed_motion_scalar_values": 148045091,
  "motion_norm_ratio_mean": 0.8760916570236936,
  "motion_norm_ratio_min": 0.7890567831954108,
  "quotient_scalar_checks": 499968,
  "independent_CDFs": 256,
  "max_CDF_error": 2.872051554914101e-16,
  "frozen_spatial_arrays_exact": 512,
  "new_GT": true,
  "official_Dense_output_checks": 2560,
  "scalar_candidate_quality_calls": 798720,
  "GT_provenance": {
    "data/hcstvg2_official_metadata/val_v2.json": "2e22d9370e311ca6c420b90867f5128507f5ab145e4fc120495b7a6336e41156"
  },
  "all128_before_GT": true,
  "Native_HiResAuxMix_exact_previous": true
}
```

hard读出和完整分布质量分别判定，不以hard阴性否定所有OPD。无单位系数/范数/均值范围/混合搜索，失败只关闭本配置。CURRENT/MAIN/PAPER及空间不动。
