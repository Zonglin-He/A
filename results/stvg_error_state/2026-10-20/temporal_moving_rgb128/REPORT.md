# 冻结轨迹引导的逐帧RGB时间观察：HC2 RC128

沿用旧cycle的每帧2倍框、图内最小16像素、224方形INTER_LINEAR；真实RGB重新进入同Source Vid。MovingLive由预锁冻结Live41指导，MovingSource为原Source轨迹指导；新框全部丢弃，原时间全跨度65帧/两流envelope/MBR不变。旧HC1八＋Vid八不是本轮确认，其已有移动RGB前向不重新命名新颖性。原空间和网络不更新，本轮不是OPD。

| 对比 | 完整v增量pp [CI] | 候选期望GTv增量pp [CI] |
|---|---|---|
| MovingLiveMix_minus_Native | -0.560749 [-2.099211371081747, 0.9523654257438182] | -1.254948 [-2.031565429981637, -0.489280706178335] |
| MovingLiveMix_minus_HiResAuxMix | -2.167184 [-3.8041624365567155, -0.5176907818645516] | -1.064713 [-2.0220806079758917, -0.08711630049003372] |
| MovingLiveMix_minus_UnionCrop0Mix | -0.590227 [-2.1232872775777483, 0.9208265044948224] | -1.233756 [-2.0019240994901386, -0.4713618773877041] |
| MovingLiveMix_minus_MovingSourceMix | -0.401523 [-1.1992430273459396, 0.28264995882412963] | -0.242072 [-0.6366489105596452, 0.14023360971482535] |
| MovingLiveMix_minus_MovingLive | +1.867093 [0.7965109910706454, 2.952063303010023] | +1.254948 [0.48928070617833136, 2.0315654299816344] |
| MovingLive_minus_Native | -2.427842 [-4.522387207840138, -0.37406735495026655] | -2.509896 [-4.063130859963272, -0.9785614123566662] |
| MovingLive_minus_HiResAuxMix | -4.034276 [-6.037425441019651, -2.0609758288961295] | -2.319661 [-3.858369765974473, -0.7665475720060291] |
| MovingLive_minus_MovingSource | +0.032856 [-1.0550202527484949, 1.1064202189656736] | -0.484144 [-1.2732978211192898, 0.28046721942965264] |
| MovingSourceMix_minus_Native | -0.159227 [-1.7177713403734054, 1.3942376092757387] | -1.012876 [-1.8609880513880785, -0.16345406738156795] |
| MovingSource_minus_Native | -2.460698 [-4.652385008222869, -0.32693877453745135] | -2.025752 [-3.7219761027761558, -0.3269081347631348] |
| UnionCrop0Mix_minus_Native | +0.029478 [-0.13474684439349485, 0.20599779590956876] | -0.021192 [-0.13874904845784272, 0.10095147526274521] |
| HiResAuxMix_minus_Native | +1.606434 [0.5915612220621151, 2.666302924676544] | -0.190235 [-1.0486135913597796, 0.6718222797720916] |

主比较正/负/零 49/63/16；<-3pp 29，<-5pp 17；最坏 [{'alias': 'RC082', 'delta_pp': -44.645766439610625}, {'alias': 'RC042', 'delta_pp': -33.7495079439399}, {'alias': 'RC014', 'delta_pp': -22.19475707291392}, {'alias': 'RC008', 'delta_pp': -19.247573630318314}, {'alias': 'RC076', 'delta_pp': -14.678580111276107}, {'alias': 'RC109', 'delta_pp': -10.991986032976236}, {'alias': 'RC122', 'delta_pp': -9.931327019655184}, {'alias': 'RC024', 'delta_pp': -9.9044892020509}]。
RC128反复曝光、三冻结空间seed父内平均、全128先封存后GT。直接/混合读出不等于OPD贡献。旧缓存不免费；本轮完整Source实际成本包含真实RGB解码/重编码、原生spatial/temporal解码和FP32 replay。裁剪同时改变尺度/身份/上下文/运动参照与形变，可能放大抖动或消除真实移动，不能单独解释为像素细节优势。未超过Source指导/Strong时不声称WHERE→WHEN或方法领先；硬失败不单独否定分布转移资格。

```json
{
  "primary_hard": {
    "mean": -0.5607494495165946,
    "CI95": [
      -2.099211371081747,
      0.9523654257438182
    ],
    "parents": 128
  },
  "pure_teacher_expected": {
    "mean": -2.5098960443222516,
    "CI95": [
      -4.063130859963272,
      -0.9785614123566662
    ],
    "parents": 128
  },
  "decision": "close_this_moving_RGB_configuration",
  "versus_Strong": {
    "mean": -2.1671835567603117,
    "CI95": [
      -3.8041624365567155,
      -0.5176907818645516
    ],
    "parents": 128
  },
  "versus_SourceGuidance": {
    "mean": -0.4015227956480867,
    "CI95": [
      -1.1992430273459396,
      0.28264995882412963
    ],
    "parents": 128
  },
  "not_OPD": true,
  "not_independent_confirmation": true,
  "no_crop_seed_mixture_temperature_sweep": true,
  "goal_achieved": false
}
{
  "GPU_seconds": 689.2380110369995,
  "measured_lease_seconds": 686.2380110369995,
  "cleanup_reserve_seconds": 3,
  "actual_calls": {
    "Source_attempts": 257,
    "Source_completed": 257,
    "image_attempts": 514,
    "image_completed": 514,
    "motion_attempts": 514,
    "motion_completed": 514
  },
  "native_control_Source_attempts": 1,
  "native_control_Source_calls": 1,
  "new_view_Source_attempts": 256,
  "new_view_Source_calls": 256,
  "independent_actual_RGB_checks": 4,
  "GPU_peak_allocated_bytes": 2180305920,
  "GPU_peak_reserved_bytes": 2550136832,
  "opencv_version": "4.11.0",
  "opencv_threads": 2,
  "new_DINO": 0,
  "new_ReferDINO": 0,
  "new_network_training": 0,
  "spatial_updates": 0,
  "new_student_updates": 0
}
{
  "status": "pass",
  "independent_scalar_rectangle_checks": 16640,
  "independent_real_RGB_redecode_view_checks": 4,
  "original_all_RGB_SHA_matches_R13": true,
  "independent_CDF_checks": 256,
  "old_posteriors_intervals_exact": 384,
  "frozen_spatial_arrays_exact": 512,
  "independent_MBR_original_action_utilities": 12537,
  "max_CDF_error": 2.883609319479386e-16,
  "Native_full_Source_boxes_logits_q_exact": true,
  "new_GT": true,
  "official_Dense_output_checks": 3584,
  "scalar_candidate_quality_calls": 798720,
  "GT_provenance": {
    "data/hcstvg2_official_metadata/val_v2.json": "2e22d9370e311ca6c420b90867f5128507f5ab145e4fc120495b7a6336e41156"
  },
  "all128_before_GT": true,
  "all_temporal_arms_sIoU_identical": true
}
```

固定GT支持及未观察实际节点sIoU不变；不扫crop/seed/混合/温度，不晋升空间/CURRENT/MAIN/PAPER。
