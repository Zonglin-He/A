# 冻结时间模块条件MC积分：HC128能力检查

同Native memory和第二pass query，48完整后验积分；只保留原TimeDecoder .1和原两次head .3随机位置。HeadOnly/OutputOnly共享head/output masks。固定网络参数，没有model.train/新训练/新图像/专家/空间或学生update。三16draw流不是独立父样本，不选最优种子。

| 对比 | 完整v增量pp [CI] | 候选期望GTv增量pp [CI] |
|---|---|---|
| FullMC_minus_Native | -0.067960 [-0.4194412326813705, 0.2922916393484585] | -0.068888 [-0.23761477467428715, 0.11031270689531095] |
| FullMC_minus_HeadOnlyMC | +0.575677 [0.08316440289621177, 1.263687083297666] | +0.042389 [-0.11570969633258135, 0.22370135512060427] |
| FullMC_minus_OutputOnlyMC | +0.620878 [0.1314428674995109, 1.3005559240113884] | +0.036186 [-0.122865058142872, 0.21734860394304706] |
| FullMC_minus_HiResAuxMix | -1.674394 [-2.7409608867846993, -0.6300772022485793] | +0.121347 [-0.7679522277543857, 1.0097778632946928] |
| HeadOnlyMC_minus_Native | -0.643637 [-1.2232046065194198, -0.17480052554201425] | -0.111277 [-0.27897756982832217, 0.059709373535574366] |
| OutputOnlyMC_minus_Native | -0.688838 [-1.2624480775811122, -0.21278910577534202] | -0.105073 [-0.27085788465734545, 0.06319371239497767] |

RC128反复曝光、三冻结空间seed父内平均、全128预测封存后GT。TC109不选参。条件MC并非完整Bayesian推断，实际checkpoint训练日志缺失；不能以简单噪声平滑为新事件知识。本轮非OPD/独立确认，不以hard失败否定所有蒸馏，不选控制晋升。原2080动作MBR不变。

```json
{
  "primary_hard": {
    "mean": -0.0679598753035506,
    "CI95": [
      -0.4194412326813705,
      0.2922916393484585
    ],
    "parents": 128
  },
  "primary_expected": {
    "mean": -0.06888785464023918,
    "CI95": [
      -0.23761477467428715,
      0.11031270689531095
    ],
    "parents": 128
  },
  "decision": "close_this_conditional_MC_configuration",
  "not_OPD": true,
  "not_independent_confirmation": true,
  "no_rate_draw_seed_mixture_search": true,
  "goal_achieved": false
}
{
  "GPU_seconds": 144.1414544209838,
  "time_decoder_offset_calls": 12808,
  "projection_head_calls": 25100,
  "call_attempts_and_completed": {
    "decoder_attempts": 12808,
    "decoder_completed": 12808,
    "head_attempts": 25100,
    "head_completed": 25100
  },
  "native_logits_before_after_exact": 512,
  "zero_rate_exact": 4,
  "same_seed_replay_exact": 4,
  "GPU_peak_allocated_bytes": 1011191296,
  "GPU_peak_reserved_bytes": 1033895936,
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
  "independent_CDF_draw_checks": 18432,
  "paired_head_and_output_mask_multiplication_checks": 36864,
  "old_posteriors_intervals_exact": 256,
  "frozen_spatial_arrays_exact": 512,
  "independent_MBR_original_action_utilities": 7678,
  "max_error": 1.1518563880486e-15,
  "all_48_draw_means_exact": true,
  "native_logits_exact": true,
  "new_GT": true,
  "official_Dense_output_checks": 2560,
  "scalar_candidate_quality_calls": 798720,
  "GT_provenance": {
    "data/hcstvg2_official_metadata/val_v2.json": "2e22d9370e311ca6c420b90867f5128507f5ab145e4fc120495b7a6336e41156"
  },
  "all128_before_GT": true
}
```

Gal&Ghahramani2016原始依据：https://proceedings.mlr.press/v48/gal16.html；该理论不保证本冻结checkpoint校准或STVG收益。CURRENT/MAIN/PAPER与空间保持不变。
