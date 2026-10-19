# 保留全图上下文的注册RGB-motion：HC2 RC128

实际评分入口：`scripts/score_stvg_temporal_registered_motion128_audit001.py`。原scorer预GT浮点坐标审计失败保留；修订001匹配本机CPU vectorized contracted FP32坐标，再独立FP64插值，原16eps32门槛、93测量pins/PLAN/barrier和预测均不变，无模型重跑。见NUMERICAL_AUDIT_REVISION001_PUBLIC.json及同名audit001协议。额外失败/诊断CPU时间未分别精确计时，不能将下面成功评分时间称全部研究CPU成本。

原appearance/text融合输入逐值保留；实际局部RGB先经同Swin/project2，再按均匀有效cell/面积覆盖率注册，仅替换原motion。FeatureWarp在原motion上裁/重采样后走同注册，区分新RGB与重采样。位置/布尔padding/65物理帧及两offset/MBR均不动，新空间输出全丢弃，无学生或网络更新。本轮不是OPD。

| 对比 | 完整v增量pp [CI] | 候选期望GTv增量pp [CI] |
|---|---|---|
| RGBRegisteredMix_minus_Native | -0.349982 [-0.8978247080985634, 0.1357694658846642] | -0.148397 [-0.4112037079440263, 0.09772983744872199] |
| RGBRegisteredMix_minus_HiResAuxMix | -1.956416 [-3.1104703965670604, -0.8655258706900355] | +0.041838 [-0.7972449634986712, 0.8910829606597609] |
| RGBRegisteredMix_minus_MovingLive | +2.077860 [0.054641280609596034, 4.119103294976723] | +2.361499 [0.8665416774101343, 3.889374830522054] |
| RGBRegisteredMix_minus_FeatureWarpMix | -0.069087 [-0.4720669492249774, 0.32829616704380205] | +0.015762 [-0.2920872725063375, 0.3266940333472975] |
| RGBRegisteredMix_minus_RGBRegistered | +0.036469 [-0.3001524354881384, 0.3612309121808003] | +0.148397 [-0.09772983744872304, 0.4112037079440279] |
| RGBRegistered_minus_Native | -0.386451 [-1.103872191496689, 0.3164684099685028] | -0.296793 [-0.8224074158880543, 0.19545967489744503] |
| RGBRegistered_minus_HiResAuxMix | -1.992885 [-3.1869904285312898, -0.8735414081084995] | -0.106558 [-1.0084240072203574, 0.8100037394265021] |
| RGBRegistered_minus_FeatureWarp | +0.295387 [-0.3131563806091738, 0.9278952131922614] | +0.031525 [-0.584174545012676, 0.6533880666945948] |
| RGBRegistered_minus_MovingLive | +2.041391 [0.012343817999778493, 4.074725565946332] | +2.213103 [0.6988762214114527, 3.770430677789732] |
| FeatureWarpMix_minus_Native | -0.280895 [-0.8160136955442416, 0.044407226324283795] | -0.164159 [-0.3732022379523341, -0.0002222194312863372] |
| FeatureWarp_minus_Native | -0.681838 [-1.369285438001676, -0.167226621584347] | -0.328318 [-0.7464044759046666, -0.0004444388625712031] |
| HiResAuxMix_minus_Native | +1.606434 [0.5915612220621151, 2.666302924676544] | -0.190235 [-1.0486135913597796, 0.6718222797720916] |

主比较正/负/零 31/47/50；<-3 9，<-5 4；最坏[{'alias': 'RC059', 'delta_pp': -21.09895807410737}, {'alias': 'RC107', 'delta_pp': -16.560447706261158}, {'alias': 'RC125', 'delta_pp': -6.217391787197609}, {'alias': 'RC100', 'delta_pp': -5.094997882286273}, {'alias': 'RC027', 'delta_pp': -4.8366580850498835}, {'alias': 'RC056', 'delta_pp': -4.154964423290986}, {'alias': 'RC082', 'delta_pp': -3.801089069621111}, {'alias': 'RC112', 'delta_pp': -3.733493457248077}]。
RC128反复曝光、三冻结空间seed父内平均、全部128先封存后GT，不是独立确认。均匀feature cell注册不是精确CNN感受野；不能逆转Swin在移动crop中已混合的运动参照，也可能丢上下文或回粗格丢细节。原appearance/text只在fusion输入固定，融合后输出可变。未超过FeatureWarp不归因新RGB；未超过Strong不称方法领先。

```json
{
  "primary_hard": {
    "mean": -0.34998215621467654,
    "CI95": [
      -0.8978247080985634,
      0.1357694658846642
    ],
    "parents": 128
  },
  "pure_teacher_expected": {
    "mean": -0.2967934063673676,
    "CI95": [
      -0.8224074158880543,
      0.19545967489744503
    ],
    "parents": 128
  },
  "new_RGB_versus_FeatureWarp_expected": {
    "mean": 0.03152490630068347,
    "CI95": [
      -0.584174545012676,
      0.6533880666945948
    ],
    "parents": 128
  },
  "decision": "close_this_registered_motion_configuration",
  "versus_Strong": {
    "mean": -1.956416263458394,
    "CI95": [
      -3.1104703965670604,
      -0.8655258706900355
    ],
    "parents": 128
  },
  "not_OPD": true,
  "not_independent_confirmation": true,
  "no_crop_alpha_mixture_temperature_sweep": true,
  "goal_achieved": false
}
{
  "GPU_seconds": 269.7891099078115,
  "measured_lease_seconds": 266.7891099078115,
  "cleanup_reserve_seconds": 3,
  "actual_calls": {
    "image_attempts": 4,
    "image_completed": 4,
    "motion_attempts": 260,
    "motion_completed": 260,
    "projection_attempts": 260,
    "projection_completed": 260,
    "fusion_attempts": 772,
    "fusion_completed": 772,
    "decoder_attempts": 1552,
    "decoder_completed": 1552,
    "head_attempts": 776,
    "head_completed": 776
  },
  "Source_control_attempts": 2,
  "Source_control_completed": 2,
  "standalone_motion_attempts": 256,
  "standalone_motion_completed": 256,
  "GPU_peak_allocated_bytes": 2750670336,
  "GPU_peak_reserved_bytes": 3344957440,
  "new_DINO": 0,
  "new_ReferDINO": 0,
  "new_network_training": 0,
  "spatial_updates": 0,
  "new_student_updates": 0
}
{
  "status": "pass",
  "frozen_spatial_arrays_exact": 512,
  "old_posteriors_intervals_exact": 384,
  "independent_CDF_checks": 384,
  "max_CDF_error": 2.92300905702092e-16,
  "all_geometry_cells_independent": 578305,
  "independent_bilinear_channel_scalars": 204800,
  "max_bilinear_relative_error": 0.0004839977191295475,
  "max_FP16_local_relative_error": 0.0004839977191295475,
  "max_FP32_registered_relative_error": 2.017502781000126e-07,
  "numerical_oracle": "CPU vectorized contracted FP32 coordinates plus independent FP64 interpolation",
  "actual_fusion_inputs_SHA_reconstructed": true,
  "independent_MBR_sample_utilities": 12532,
  "Source_controls": {
    "status": "pass",
    "Native_boxes_logits_q_raw_motion_exact": true,
    "Crop_logits_q_exact_R28": true,
    "GT_read": false
  },
  "Source_control_raw_receipt_and_tensor_values_exact": true,
  "new_GT": true,
  "official_Dense_output_checks": 3584,
  "scalar_candidate_quality_calls": 798720,
  "old_metrics_exact": 1536,
  "GT_provenance": {
    "data/hcstvg2_official_metadata/val_v2.json": "2e22d9370e311ca6c420b90867f5128507f5ab145e4fc120495b7a6336e41156"
  },
  "all128_before_GT": true,
  "all_temporal_arms_sIoU_identical": true
}
```

固定GT支持和未观察actual sIoU不变；3584最终输出official独立核，798720候选质量仅Dense逐项计算，MBR是预定抽点核，不夸大全候选双实现。旧缓存不免费，新Swin/fusion/两pass成本已计。关闭精确配置不否定全部空间条件时间/OPD；无crop/alpha/混合救援，不晋升空间或CURRENT/MAIN/PAPER。
