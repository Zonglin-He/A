# 原生时间解码阶段与六层时间头：HC128开发

固定原Source224融合特征和E/Live三seed空间数组。严格原生temporal/spatial frame gate及attention query factory复原第一次时间解码，再由原actionness gate复原第二次query；第二次query、最后层logits和原Native必须exact。读取两次解码各六层既有时间头，不加新头/参数/像素。源码及配置为中间层提供aux loss，但无实际训练调用日志，不能宣称第一次唯一被训练或一定更强。

| 输出 | full vIoU% | tIoU% | 相对Native v增量pp [CI95] |
|---|---:|---:|---|
| Native | 27.928474 | 51.439514 | 0 |
| HiResMix | 29.161192 | 54.763483 | +1.232718 [+0.322089,+2.199435] |
| P1L1 | 27.021294 | 51.607015 | -0.907180 [-3.090046,+1.180566] |
| P1L2 | 28.226554 | 52.039159 | +0.298080 [-0.942732,+1.436676] |
| P1L3 | 28.161232 | 51.379292 | +0.232758 [-0.762627,+1.149815] |
| P1L4 | 28.442608 | 52.366830 | +0.514134 [-0.148187,+1.163998] |
| P1L5 | 28.237419 | 51.862945 | +0.308945 [-0.074094,+0.750440] |
| P1L6 | 27.955224 | 51.444696 | +0.026750 [-0.428523,+0.483745] |
| P2L1 | 26.523251 | 50.719756 | -1.405223 [-3.473847,+0.584618] |
| P2L2 | 28.149061 | 51.518241 | +0.220587 [-0.839928,+1.248997] |
| P2L3 | 27.932454 | 50.967165 | +0.003980 [-0.904296,+0.864816] |
| P2L4 | 28.369333 | 52.228795 | +0.440859 [-0.134712,+0.998038] |
| P2L5 | 28.021696 | 51.548147 | +0.093222 [-0.198580,+0.376278] |
| P2L6 | 27.928474 | 51.439514 | +0.000000 [+0.000000,+0.000000] |
| FirstPassMix | 27.980093 | 51.523048 | +0.051619 [-0.188495,+0.277462] |
| Pass1Mean6 | 28.610232 | 52.661978 | +0.681759 [-0.030817,+1.399397] |
| Pass2Mean6 | 28.450143 | 52.330179 | +0.521669 [-0.122597,+1.163628] |
| BothMean12 | 28.607162 | 52.560347 | +0.678688 [+0.033458,+1.312214] |
| HiResAuxMix | 29.534908 | 55.920522 | +1.606434 [+0.591561,+2.666303] |
| NativeUniformMix | 28.626558 | 53.083433 | +0.698084 [-0.176411,+1.537390] |
| HiResUniformMix | 28.918837 | 55.349733 | +0.990364 [-0.857032,+2.898032] |

主HiResAuxMix=.5已有Source448 raw+.5第二次六层posterior均值，Avg6退化Native时精确回到已有HiResMix到舍入误差。FirstPassMix、Pass1/Pass2Mean6、BothMean12与主方案五eligible只在整个开发父均值选择。逐层P1/P2L1..6只是诊断，不按GT挑一层晋升；Uniform同正质量支持的两控制，检查收益是否只是变平。所有后验先逐层独立建立两个offset envelope，再等权平均，不能先平均logits后声称相同。

```json
{
  "primary": "HiResAuxMix",
  "selected": "HiResAuxMix",
  "selected_delta": {
    "mean": 1.6064341072437172,
    "CI95": [
      0.5915612220621151,
      2.666302924676544
    ],
    "parents": 128
  },
  "decision": "no_supported_meaningful_readout",
  "ranking_diagnostics_available": true,
  "not_OPD": true,
  "independent_confirmation": false
}
```

## 成本与独立核验

```json
{
  "GPU_seconds": 73.34568843408488,
  "time_decoder_offset_calls": 512,
  "attribute_and_action_gate_head_calls": 1280,
  "time_projection_head_calls": 512,
  "native_query_and_logits_exact_checks": 256,
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
  "official_DenseTube_checks": 5388,
  "spatial_arrays_exact": 512,
  "direct_MBR_utilities": 48367,
  "vectorized_quality_checks": 9728,
  "max_error": 1.1102230246251565e-15,
  "all128_before_GT": true,
  "numeric_independent": {
    "status": "pass",
    "all128_before_GT": true,
    "raw_head_posterior_CDF_checks": 1536,
    "native_gate_checks": 512,
    "independent_native_query_means": 512,
    "native_second_query_and_last_logits_exact": 256,
    "fixed_scaled_query_and_gate_tolerance": 1.9073486328125e-06,
    "max_scaled_query_error": 1.1674107471546119e-07,
    "max_gate_sigmoid_error": 8.021063668817163e-08,
    "max_CDF_error": 3.089434090497445e-16,
    "posterior_mixture_checks": 896,
    "no_new_model_calls": true,
    "no_GT": true
  },
  "GT_provenance": {
    "data/hcstvg2_official_metadata/val_v2.json": "2e22d9370e311ca6c420b90867f5128507f5ab145e4fc120495b7a6336e41156"
  }
}
```

所有128父和所有输出signed/未知/负尾全留；全部封存后GT，三个空间seed父内平均。原生阶段改变不是新的on-policy学生更新；无新OPD成绩。若只有静态teacher读出改善，不能归OPD。没有改空间或CURRENT/MAIN/PAPER。先确认有效教师分布再有依据设计实际Direct/Fixed/Live，不以层选择或优化强度延长阴性配置。
