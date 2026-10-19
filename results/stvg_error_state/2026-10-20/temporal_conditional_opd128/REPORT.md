# 缓存教师下的双边界条件OPD：HC128

教师T=.5旧224+.5新448 Source后验固定。只优化每视频新初始化的130临时时间logit偏置，不训练网络/定位头/评价器，不动空间。Native/TeacherDirect完整复用；DirectJoint为完整教师KL，DirectConditional为teacher-forced状态同条件损失，Fixed每步fresh初始分布采样，Live每步fresh当前分布采样，paired41/42/43。两边界条件标签来自构造的T，不是原生Source条件decoder概率。

| 输出 | 完整vIoU% | tIoU% | v增量pp [CI95] |
|---|---:|---:|---|
| Native | 27.928474 | 51.439514 | 0 |
| TeacherDirect | 29.161192 | 54.763483 | +1.232718 [+0.322089,+2.199435] |
| DirectJoint | 29.101432 | 54.747075 | +1.172958 [+0.256879,+2.140166] |
| DirectConditional | 29.068660 | 54.676134 | +1.140186 [+0.249314,+2.080985] |
| Fixed | 28.965874 | 53.923800 | +1.037400 [+0.250724,+1.856093] |
| Live | 29.086330 | 54.725496 | +1.157856 [+0.266823,+2.096643] |

主Mean Fixed/Live只取同seed时间/空间配对再父内均值；其他crossseed值不当独立样本。Direct确定性时间在三个冻结空间上平均。全部父及负尾保留；RC128反复曝光，任何开发正值不独立证明。没有重查RGB，Live刷新的是当前学生区间状态及相应两个条件目标；每个32采样区间请求两份至多65类分布，不称32次标量reward。

```json
{
  "primary": "paired-seed Mean_Live",
  "net": {
    "mean": 1.157855814159922,
    "CI95": [
      0.26682275946200523,
      2.0966428755460202
    ],
    "parents": 128
  },
  "versus_controls": {
    "TeacherDirect": {
      "mean": -0.07486245570695134,
      "CI95": [
        -0.18121507547625548,
        0.02193881290203161
      ],
      "parents": 128
    },
    "DirectJoint": {
      "mean": -0.015102592876510595,
      "CI95": [
        -0.09508594992306353,
        0.06105821652076957
      ],
      "parents": 128
    },
    "DirectConditional": {
      "mean": 0.017670064904505522,
      "CI95": [
        -0.05650929065819181,
        0.09502346826469502
      ],
      "parents": 128
    },
    "Fixed": {
      "mean": 0.12045540222511121,
      "CI95": [
        -0.05960917876391385,
        0.31405386301067195
      ],
      "parents": 128
    }
  },
  "decision": "no_supported_large_independent_OPD_advantage_in_this_configuration",
  "not_independent_confirmation": true,
  "no_parameter_search_authorized_by_this_round": true
}
```

## 成本及独立核验

```json
{
  "CPU_seconds": 33.73017764906399,
  "fits": 1024,
  "actual_updates": 40960,
  "scheduled_updates": 40960,
  "temporary_parameters_per_fit": 130,
  "first_Fixed_Live_exact": 384,
  "max_manual_gradient_error": 4.163336342344337e-16,
  "new_GPU": 0,
  "new_visual_calls": 0,
  "new_experts": 0,
  "network_training": 0,
  "spatial_updates": 0
}
{
  "status": "pass",
  "independent_numpy_Adam_gradient_sampling_steps": 40960,
  "maxerror": 7.105427357601002e-15,
  "first_Fixed_Live_exact": 384,
  "Live_steps_differ_from_initial_sampler": 14976,
  "new_models": 0,
  "new_target_GT_scored": true,
  "official_DenseTube_checks": 5120,
  "spatial_arrays_exact": 512,
  "updates": 40960,
  "Native_TeacherDirect_exact_previous": true,
  "all128_before_GT": true,
  "GT_provenance": {
    "data/hcstvg2_official_metadata/val_v2.json": "2e22d9370e311ca6c420b90867f5128507f5ab145e4fc120495b7a6336e41156"
  }
}
```

完整signed/unknown、逐seed差值/负尾、每父实际更新数在RESULTS_PUBLIC.json。原Source448>=2pp gate仍未通过，选择半数融合教师是本轮新开发决定。当前结果不自动更改Current/Main/Paper或空间配置，也不泛化所有条件OPD。
