# 固定合理性条件：HC128纯缓存能力检查

R22负结果保留，此轮不是工程修复或原官方LVLM算法复现。按原Native固定H={I:p0>=.1maxp0,p0>0}，三Head共用H，限制belief不限制原MBR动作集合。词表合理性不保证事件区间正确；旧Native、Square、Contrast、MeanMotion、HiResAuxMix原样保存。只有ContrastHead为候选，不以控制赢家事后晋升。

| 对比 | 完整v增量pp [CI] | 候选期望GTv增量pp [CI] |
|---|---|---|
| ContrastHead_minus_Native | -0.607639 [-1.9611881720769722, 0.8117675796990204] | -0.077131 [-1.0543478609613728, 0.8543021765829448] |
| ContrastHead_minus_NativeHead | -0.250954 [-1.443960253745581, 1.0906191376239291] | -0.627695 [-1.5856570836052533, 0.2610659854574978] |
| ContrastHead_minus_SquareHead | +0.058060 [-1.081394159611975, 1.3564348142015683] | -1.075665 [-2.039016651479463, -0.13563108772610927] |
| ContrastHead_minus_HiResAuxMix | -2.214074 [-4.018437779821845, -0.3736143072712471] | +0.113104 [-1.2991586120645757, 1.5019541996262655] |
| ContrastHead_minus_Contrast | +0.096039 [-0.2941354083923215, 0.5198419947058915] | +1.009224 [0.6765874476834554, 1.3588902402744196] |
| NativeHead_minus_Native | -0.356685 [-0.8327052209595663, -0.03341817133666413] | +0.550564 [0.3502821048269622, 0.7538914418889429] |
| SquareHead_minus_Native | -0.665699 [-1.26909320888429, -0.19857095670369165] | +0.998534 [0.6293551910852517, 1.3770298204243672] |

旧曝光RC128/三冻结空间seed父内平均，所有新输出seal后GT，TC109未选参。只beta=.1一次预锁，无新模型/图像/专家/训练/学生或空间update。超过旧Contrast不等于净收益；三Head共同改善不等于motion知识增量。

```json
{
  "primary_hard": {
    "mean": -0.6076394047250995,
    "CI95": [
      -1.9611881720769722,
      0.8117675796990204
    ],
    "parents": 128
  },
  "primary_expected": {
    "mean": -0.07713119390064588,
    "CI95": [
      -1.0543478609613728,
      0.8543021765829448
    ],
    "parents": 128
  },
  "decision": "close_current_mean_motion_quotient_family_in_this_information_budget",
  "not_OPD": true,
  "not_independent_confirmation": true,
  "no_beta_coefficient_norm_search": true,
  "goal_achieved": false
}
{
  "CPU_prediction_seconds": 1.2318499519024044,
  "GPU_seconds": 0,
  "new_models": 0,
  "new_RGB": 0,
  "new_Source_encoder": 0,
  "new_DINO": 0,
  "new_ReferDINO": 0,
  "network_training": 0,
  "spatial_updates": 0,
  "new_student_updates": 0
}
{
  "status": "pass",
  "independent_H_parents": 128,
  "independent_normalization_scalar_checks": 96990,
  "same_H_support_all_three": true,
  "old_posteriors_intervals_exact": 640,
  "frozen_spatial_arrays_exact": 512,
  "independent_MBR_original_action_utilities": 7677,
  "max_error": 4.996003610813204e-16,
  "H_count_mean": 252.578125,
  "H_count_min": 28,
  "H_count_max": 904,
  "MBR_outside_H_counts": {
    "NativeHead": 0,
    "SquareHead": 0,
    "ContrastHead": 0
  },
  "MBR_actions_unrestricted": true,
  "new_GT": true,
  "official_Dense_output_checks": 4096,
  "scalar_candidate_quality_calls": 798720,
  "GT_provenance": {
    "data/hcstvg2_official_metadata/val_v2.json": "2e22d9370e311ca6c420b90867f5128507f5ab145e4fc120495b7a6336e41156"
  },
  "all128_before_GT": true,
  "old_metrics_exact_previous": true
}
```

hard与分布质量分别判定，非OPD/独立确认，不扫beta/系数/范数/均值范围。CURRENT/MAIN/PAPER与空间保持不变。
