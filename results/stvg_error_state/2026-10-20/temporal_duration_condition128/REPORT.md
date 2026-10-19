# 保留节点数边际的时间证据融合：HC128缓存能力检查

旧Raw负结果保留；本轮利用原Native完整长度边际和同长度事件位置证据。节点数不是精确物理持续时间，belief边际保持也不保证MBR最终区间长度保持。唯一DurationConditional，不挑控制晋升；无新专家/模型/训练/学生或空间更新。

| 对比 | 完整v增量pp [CI] | 候选期望GTv增量pp [CI] |
|---|---|---|
| DurationConditional_minus_Native | -0.482295 [-1.1080731492386773, 0.08429839714938928] | +0.833746 [0.47243914122496355, 1.2113912235068298] |
| DurationConditional_minus_GlobalFuse | +2.390276 [0.867690998930921, 3.9351026992127713] | +2.086821 [0.8734997492747145, 3.430505743568356] |
| DurationConditional_minus_DurationShift32 | +17.772085 [13.434063844975944, 22.179132606085258] | +16.662783 [13.017534155633573, 20.418913897748485] |
| DurationConditional_minus_DurationWrongParent | +7.758501 [4.511702726010324, 11.117674108617548] | +7.283167 [4.693961543615713, 10.085271861964427] |
| DurationConditional_minus_HiResAuxMix | -2.088729 [-3.3700487578694176, -0.8538524765975886] | +1.023982 [-0.023420007937849506, 2.112411114831437] |
| DurationConditional_minus_RawTeacher | +2.952797 [1.3662459670001375, 4.623468593893929] | +3.663459 [2.307681229712846, 5.133394144668382] |
| GlobalFuse_minus_Native | -2.872571 [-4.621499214964319, -1.143710262836043] | -1.253074 [-2.7239136655086784, 0.13792129440689455] |
| DurationShift32_minus_Native | -18.254379 [-22.60212044885398, -14.017337152611388] | -15.829037 [-19.36114255744298, -12.386687524699457] |

RC128反复曝光、三冻结空间seed父内平均、全父全128预测封存后GT。TC109不选参。本轮非OPD/独立确认，不以hard失败否定所有蒸馏；也不以KL/边际守恒或旧Raw改善充当真实净收益。

```json
{
  "primary_hard": {
    "mean": -0.4822947132513976,
    "CI95": [
      -1.1080731492386773,
      0.08429839714938928
    ],
    "parents": 128
  },
  "primary_expected": {
    "mean": 0.8337464353976393,
    "CI95": [
      0.47243914122496355,
      1.2113912235068298
    ],
    "parents": 128
  },
  "decision": "close_this_duration_conditional_configuration",
  "not_OPD": true,
  "not_independent_confirmation": true,
  "no_temperature_duration_weight_step_search": true,
  "goal_achieved": false
}
{
  "CPU_prediction_seconds": 5.201798995025456,
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
  "independent_scalar_normalization_checks": 999936,
  "old_posteriors_intervals_exact": 384,
  "frozen_spatial_arrays_exact": 512,
  "independent_MBR_original_action_utilities": 10237,
  "max_error": 1.0824674490095276e-15,
  "node_count_marginal_preserved_all128": true,
  "old_raw_energy_exact": true,
  "MBR_actions_unrestricted": true,
  "new_GT": true,
  "official_Dense_output_checks": 3584,
  "scalar_candidate_quality_calls": 798720,
  "GT_provenance": {
    "data/hcstvg2_official_metadata/val_v2.json": "2e22d9370e311ca6c420b90867f5128507f5ab145e4fc120495b7a6336e41156"
  },
  "all128_before_GT": true
}
```

Source自身特征可能重复计入，保留错误长度会限制位置修复，事件logits并非校准概率。CURRENT/MAIN/PAPER与空间保持不变。
