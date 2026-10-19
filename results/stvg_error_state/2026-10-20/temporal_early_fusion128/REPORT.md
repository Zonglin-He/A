# 早期跨模态motion区域先验：HC128教师能力

同一旧late motion prior仅前移到逐帧encoder的motion keys；appearance/text prior1不代表attention mass不变。原生查询/门控从新memory重建，不能把收益唯一归memory。没有新跨帧encoder层；全空间数组冻结、无新专家/训练/OPD。

| 对照 | EarlyRefer完整v增量pp [CI] | 候选期望GTv增量pp [CI] |
|---|---|---|
| Native | +0.056446 [-0.9138201090501284, 1.1949047891406879] | -0.073717 [-0.6122120258011357, 0.5098091686449618] |
| LateRefer | +0.287562 [-1.013976740577217, 1.3420104109663846] | +0.486288 [-0.05170259998385369, 0.9851990012729873] |
| EarlyShift | +0.510421 [-0.6577266822443222, 1.8297581804412506] | +0.755265 [-0.10473882682102578, 1.7547441815825797] |
| HiResAuxMix | -1.549988 [-2.980243066133324, -0.017953315080256733] | +0.116518 [-0.9152081813596572, 1.2298805863255042] |

HC128均已曝光，3空间seed父内平均，不按GT选头/样本。早期输入原缓存未保存，本轮实际补同Source捕获并完整计费；新Source空间框只控数值，不进入结果。旧LateRefer/强HiResAuxMix逐值复用。Raw Action/SA边界负结果不改。

```json
{
  "primary_hard": {
    "mean": 0.0564459465156503,
    "CI95": [
      -0.9138201090501284,
      1.1949047891406879
    ],
    "parents": 128
  },
  "primary_expected": {
    "mean": -0.07371730269490488,
    "CI95": [
      -0.6122120258011357,
      0.5098091686449618
    ],
    "parents": 128
  },
  "decision": "close_this_early_motion_prior_configuration",
  "no_OPD_result": true,
  "not_independent_confirmation": true,
  "no_strength_mixture_layer_search": true,
  "goal_achieved": false
}
{
  "GPU_seconds": 505.99663058598526,
  "new_Source_queries": 128,
  "Source_offset_captures": 256,
  "Source_input_image_slots": 8320,
  "early_encoder_reencodes": 772,
  "frozen_prefix_full_native_replays": 384,
  "max_posterior_CDF_error": 2.7145838948547905e-16,
  "new_RGB_positions": 0,
  "new_DINO": 0,
  "new_ReferDINO": 0,
  "network_training": 0,
  "spatial_updates": 0,
  "new_student_updates": 0
}
{
  "status": "pass",
  "native_Source_prefix_final_time_exact_parents": 128,
  "fullframe_exact_pilots": 2,
  "encoder_bool_masks_unchanged": 256,
  "old_prior_geometry_replays": 512,
  "diagnostic_bitexact_geometry_replays": 1,
  "actual_attention_layouts_audited": 3072,
  "scalar_geometry_cells": 1156610,
  "geometry_max_relative_scaled_error": 2.7775765384631623e-07,
  "geometry_tolerance_relative_16eps32": 1.9073486328125e-06,
  "independent_CDF_tables": 384,
  "max_CDF_error": 2.7145838948547905e-16,
  "frozen_spatial_arrays_exact": 512,
  "new_GT": true,
  "official_Dense_output_checks": 2560,
  "scalar_candidate_quality_calls": 798720,
  "GT_provenance": {
    "data/hcstvg2_official_metadata/val_v2.json": "2e22d9370e311ca6c420b90867f5128507f5ab145e4fc120495b7a6336e41156"
  },
  "all128_before_GT": true,
  "Native_LateRefer_HiResAuxMix_exact_previous": true
}
```

单一EarlyRefer候选，无强度/层/模态/混合参数搜索，hard失败不是OPD失败；teacher期望质量只是能力，非已兑现方法增益。保留全128父、负尾、unknown；CURRENT/MAIN/PAPER及空间不动。
