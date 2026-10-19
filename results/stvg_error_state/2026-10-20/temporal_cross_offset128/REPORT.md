# 仅开放TimeDecoder跨offset自注意力：HC128能力检查

原第二pass query/位置与逐帧CA memory/FFN/head/两流envelope均保留；只有SA允许另一流状态。WrongBinding固定移动另一流content和V、不移动位置。不是原65完整Source重编码，也不更新空间或网络。原位置存在两相位局部索引重合边界，不能宣称完整物理时间对齐。

| 对比 | 完整v增量pp [CI] | 候选期望GTv增量pp [CI] |
|---|---|---|
| CrossOffset_minus_Native | -0.004897 [-0.7744873895849618, 0.8354810334409838] | -0.242779 [-0.5147142946713877, 0.021433475708003878] |
| CrossOffset_minus_WrongBinding | +2.710997 [-0.08628028724976708, 5.59764172088431] | +2.307487 [0.39321579254671096, 4.299587239869593] |
| CrossOffset_minus_HiResAuxMix | -1.611331 [-3.0432138466185155, -0.18563270460285813] | -0.052544 [-1.0364032972160566, 0.9316445969135524] |
| WrongBinding_minus_Native | -2.715893 [-5.38354981238118, -0.0937988529977185] | -2.550265 [-4.470786670960566, -0.722120913723365] |

主比较正/负/零 26/29/73；低于−3pp 7，低于−5pp 6，最坏 [{'alias': 'RC059', 'delta_pp': -24.757545402312417}, {'alias': 'RC025', 'delta_pp': -16.143660357982558}, {'alias': 'RC028', 'delta_pp': -6.587750950111726}, {'alias': 'RC107', 'delta_pp': -5.764328718160609}, {'alias': 'RC044', 'delta_pp': -5.4991475935373675}, {'alias': 'RC115', 'delta_pp': -5.472161948599563}, {'alias': 'RC096', 'delta_pp': -3.3111379987383502}, {'alias': 'RC065', 'delta_pp': -2.9995296744759816}]。
RC128反复曝光、三个冻结空间seed父内平均、全128先封存后GT。静态/期望质量不等于OPD价值；本轮没有学生更新，不以硬读出失败否定全部蒸馏，不按控制晋升。固定GT空间与未观察实际节点sIoU不变。

```json
{
  "primary_hard": {
    "mean": -0.004896732808728044,
    "CI95": [
      -0.7744873895849618,
      0.8354810334409838
    ],
    "parents": 128
  },
  "primary_expected": {
    "mean": -0.24277873197576644,
    "CI95": [
      -0.5147142946713877,
      0.021433475708003878
    ],
    "parents": 128
  },
  "decision": "close_this_cross_offset_SA_configuration",
  "not_OPD": true,
  "not_independent_confirmation": true,
  "no_layer_strength_position_sweep": true,
  "goal_achieved": false
}
{
  "GPU_seconds": 41.26538455183618,
  "native_logits_before_after_exact": 512,
  "disabled_exact": 4,
  "custom_two_stream_decoder_layer_groups": 256,
  "custom_offset_equivalent_decodes": 512,
  "call_attempts_and_completed": {
    "decoder_attempts": 516,
    "decoder_completed": 516,
    "SA_attempts": 6168,
    "SA_completed": 6168,
    "CA_attempts": 6168,
    "CA_completed": 6168,
    "head_attempts": 1028,
    "head_completed": 1028
  },
  "actual_attention_audits": 48,
  "max_attention_scaled_error": 8.311128885152598e-07,
  "GPU_peak_allocated_bytes": 1018696192,
  "GPU_peak_reserved_bytes": 1035993088,
  "torch_version": "2.7.0+cu128",
  "cuda_version": "12.8",
  "cudnn_version": 90701,
  "new_Source_encoder": 0,
  "new_RGB": 0,
  "new_DINO": 0,
  "new_ReferDINO": 0,
  "new_training": 0,
  "spatial_updates": 0,
  "new_student_updates": 0
}
{
  "status": "pass",
  "independent_CDF_checks": 256,
  "independent_actual_SA_checks": 48,
  "max_attention_scaled_error": 8.311128887847857e-07,
  "old_posteriors_intervals_exact": 256,
  "frozen_spatial_arrays_exact": 512,
  "independent_MBR_original_action_utilities": 5117,
  "max_error": 5.551115123125783e-16,
  "native_logits_exact": true,
  "new_GT": true,
  "official_Dense_output_checks": 2048,
  "scalar_candidate_quality_calls": 798720,
  "GT_provenance": {
    "data/hcstvg2_official_metadata/val_v2.json": "2e22d9370e311ca6c420b90867f5128507f5ab145e4fc120495b7a6336e41156"
  },
  "all128_before_GT": true,
  "all_temporal_arms_sIoU_identical": true
}
```

Source Vid检查点固定，0新RGB/encoder/专家/网络训练；旧缓存创建成本不免费。不扫层/强度/位置码，空间/CURRENT/MAIN/PAPER保持原样。
