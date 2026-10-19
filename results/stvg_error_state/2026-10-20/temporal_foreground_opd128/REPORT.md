# 原生前景教师下的双边界条件OPD：HC128

教师T=原65节点Raw Action inclusive mask energy指数归一化，仅p0正支持；不乘p0。只优化每视频新初始化的130临时时间logit偏置，不训练网络/定位头/评价器，不动空间。Native/HiResAuxMix精确复用，TeacherDirect由本轮Raw Action构建；DirectJoint为完整教师KL，DirectConditional为teacher-forced状态同条件损失，Fixed每步fresh初始分布采样，Live每步fresh当前分布采样，paired41/42/43。两边界条件标签来自构造的T，不是原生Source条件decoder概率。

| 输出 | 完整vIoU% | tIoU% | v增量pp [CI95] |
|---|---:|---:|---|
| Native | 27.928474 | 51.439514 | 0 |
| TeacherDirect | 24.493382 | 44.709238 | -3.435092 [-5.299860,-1.673892] |
| WrongTeacherDirect | 13.014028 | 23.066640 | -14.914446 [-18.686571,-11.273025] |
| HiResAuxMix | 29.534908 | 55.920522 | +1.606434 [+0.591561,+2.666303] |
| DirectJoint | 25.474886 | 46.469269 | -2.453588 [-4.160530,-0.774641] |
| DirectConditional | 25.587365 | 46.724924 | -2.341109 [-3.943431,-0.722415] |
| Fixed | 24.196548 | 44.234586 | -3.731926 [-5.626989,-1.894217] |
| Live | 25.340629 | 46.265096 | -2.587845 [-4.250794,-0.889908] |

主Mean Fixed/Live只取同seed时间/空间配对再父内均值；其他crossseed值不当独立样本。Direct确定性时间在三个冻结空间上平均。全部父及负尾保留；RC128反复曝光，任何开发正值不独立证明。没有重查RGB，Live刷新的是当前学生区间状态及相应两个条件目标；每个32采样区间请求两份至多65类分布，不称32次标量reward。

```json
{
  "primary": "paired-seed Mean_Live",
  "net": {
    "mean": -2.5878451861264082,
    "CI95": [
      -4.250794028185184,
      -0.8899082811207583
    ],
    "parents": 128
  },
  "versus_controls": {
    "TeacherDirect": {
      "mean": 0.8472463378468975,
      "CI95": [
        0.3246753217256939,
        1.4729802457750079
      ],
      "parents": 128
    },
    "HiResAuxMix": {
      "mean": -4.194279293370125,
      "CI95": [
        -6.318055944887501,
        -2.0758525419829463
      ],
      "parents": 128
    },
    "DirectJoint": {
      "mean": -0.1342571790538516,
      "CI95": [
        -0.6732091297891171,
        0.34112260357133556
      ],
      "parents": 128
    },
    "DirectConditional": {
      "mean": -0.246736678152707,
      "CI95": [
        -0.6966634852463942,
        0.15022017075077015
      ],
      "parents": 128
    },
    "Fixed": {
      "mean": 1.1440807591447386,
      "CI95": [
        0.3487927533036645,
        2.0680558441339745
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
  "CPU_seconds": 34.896401311969385,
  "fits": 1024,
  "actual_updates": 40960,
  "scheduled_updates": 40960,
  "temporary_parameters_per_fit": 130,
  "first_Fixed_Live_exact": 384,
  "max_manual_gradient_error": 3.0531133177191805e-16,
  "new_GPU": 0,
  "new_visual_calls": 0,
  "new_experts": 0,
  "network_training": 0,
  "spatial_updates": 0
}
{
  "status": "pass",
  "independent_numpy_Adam_gradient_sampling_steps": 40960,
  "maxerror": 1.0658141036401503e-14,
  "first_Fixed_Live_exact": 384,
  "Live_steps_differ_from_initial_sampler": 14976,
  "new_models": 0,
  "independent_scalar_target_checks": 249984,
  "max_target_error": 0.0,
  "new_target_GT_scored": true,
  "official_DenseTube_checks": 6144,
  "spatial_arrays_exact": 512,
  "updates": 40960,
  "Native_HiResAuxMix_exact_previous": true,
  "all128_before_GT": true,
  "GT_provenance": {
    "data/hcstvg2_official_metadata/val_v2.json": "2e22d9370e311ca6c420b90867f5128507f5ab145e4fc120495b7a6336e41156"
  }
}
```

完整signed/unknown、逐seed差值/负尾、每父实际更新数在RESULTS_PUBLIC.json。R18主排序资格与post-main wrong-parent对照支持此单次转移，不支持校准概率或预期2–3pp；旧structured_temporal已有Bernoulli代数。当前结果不自动更改Current/Main/Paper或空间配置，也不泛化所有条件OPD。


## 独立期望质量核验与裁决

完整798720次逐区间DenseTube调用＋3456个官方spot核对通过，最大误差8.88e−16。Raw教师候选期望v相对原p0−2.829713pp CI[−4.382178,−1.401110]，相对旧强HiResAuxMix−2.639478 CI负；不是只有最终hard读出阴性。Live−Fixed +1.144081[+.348793,+2.068056]、Live−RawTeacher +.847246[+.324675,+1.472980]，但Live−Native −2.587845[−4.250794,−.889908]；DirectJoint/Conditional亦净负，Live相对两Direct区间跨0。可说on-policy减轻此错误目标下的损伤，不能说有效净收益。

41改善/74下降/13不变，43低于−3pp、34低于−5pp；RC059−37.959831、RC028−30.819630、RC063−27.371928全保留，leave-one-out均值始终负。实际1024fit/40960更新34.896401s CPU；数值审计＋评分55.114189s，完整candidate核验另计。0GPU/新RGB/模型/专家/训练/空间更新。

Wrong仅静态读出控制，不能据此严格识别OPD的video-query特异因果；no-feedback指teacher==p0，Raw Action全零生成uniform不必noop。原R18排序资格保留，但不支持Raw指数后分布校准；已有Bernoulli语义来源未变。本配置关闭，不继续扫温度/先验/步数，目标仍active，不泛化所有时间OPD失败。
