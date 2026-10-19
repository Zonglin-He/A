# 原生时间decoder边界条件SA：HC128能力资格

重用旧event_query的S0与真实R7 Live41 S1及各四个样本，不是本轮新学生更新。仅六层SA keys按当前另一边界截prefix/suffix；Wrong等keys数量保留相反侧。原native query/full CA逐帧memory仍包含全局上下文，不称严格因果隔离、原生校准条件概率或一致联合teacher。固定另外一侧边界的条件期望GT fullv/t是能力统计，不是新增最终系统vIoU。

| 配对 | 条件期望fullv增量pp [CI] | t增量pp [CI] |
|---|---|---|
| S0_Correct_minus_Native | -0.438259 [-0.7823834203757765, -0.09869843002062792] | -1.088737 [-1.6298409903688846, -0.5583788507700216] |
| S0_Correct_minus_Wrong | -0.107214 [-0.6898837639378866, 0.4820192288947191] | -0.460272 [-1.3765583893463171, 0.45444394626182283] |
| S1_Correct_minus_Native | -0.383408 [-0.7423264136456217, -0.0187454188463207] | -0.948550 [-1.5305978419493744, -0.3755401833761306] |
| S1_Correct_minus_Wrong | -0.093103 [-0.6741303620303716, 0.5131979624593884] | -0.420533 [-1.3757447451729168, 0.5357214382875225] |

全部128父、两方向/四样本/三个冻结空间seed先父内平均，保留null与负尾，不以多个查询/种子虚增n。S1只是响应检查，不把旧S1收益归本轮刷新；没有最终全局MBR或OPD成绩。

```json
{
  "primary": {
    "mean": -0.43825901361277125,
    "CI95": [
      -0.7823834203757765,
      -0.09869843002062792
    ],
    "parents": 128
  },
  "matched_wrong_control": {
    "mean": -0.10721417749670976,
    "CI95": [
      -0.6898837639378866,
      0.4820192288947191
    ],
    "parents": 128
  },
  "decision": "close_this_boundary_SA_mask_configuration",
  "conditional_not_system_gain": true,
  "not_OPD": true,
  "no_mask_layer_strength_sweep": true,
  "goal_achieved": false
}
{
  "GPU_seconds": 60.73073051893152,
  "conditioned_temporal_offset_decodes": 4580,
  "native_temporal_checks": 256,
  "attention_layer_calls_audited": 29016,
  "max_blocked_attention": 0.0,
  "max_posterior_CDF_error": 2.914335439641036e-16,
  "new_Source_encoder": 0,
  "new_RGB": 0,
  "new_DINO": 0,
  "new_ReferDINO": 0,
  "network_training": 0,
  "spatial_updates": 0,
  "new_student_updates": 0
}
{
  "status": "pass",
  "physical_masks_exact": 8192,
  "conditional_labels_exact": 4096,
  "independent_CDF_tables": 4096,
  "maxerror": 2.914335439641036e-16,
  "frozen_spatial_arrays_exact": 512,
  "all_inputs_outputs_before_GT": true,
  "new_GPU": 0,
  "official_Dense_checks": 3456,
  "scalar_candidate_quality_calls": 798720,
  "GT_provenance": {
    "data/hcstvg2_official_metadata/val_v2.json": "2e22d9370e311ca6c420b90867f5128507f5ab145e4fc120495b7a6336e41156"
  },
  "all128_before_GT": true,
  "attention_layer_calls_audited": 29016,
  "max_blocked_attention": 0.0
}
```

旧RawAction teacher净负与相对减损都保留；本轮没有重调其温度/先验/步长。0新encoder/RGB/专家/网络训练/空间/学生update，GPU只是冻结时间decoder重解码，缓存成本不免费。CURRENT/MAIN/PAPER不变，全部RC128重复开发曝光，仍需独立确认才方法结论。


## 结论

S0 Correct−Native条件期望fullv−.438259pp CI[−.782383,−.098698]，−Wrong−.107214CI跨0；S1对应−.383408CI负与−.093103CI跨0。TV .172926/.171662，非未接入；没有足够条件纠错信息，不进入完整OPD、不扫mask宽度/层/强度。本结论是反馈资格阴性，不是新最终系统vIoU或所有OPD失败。主38正89负1平、9低于−3/3低于−5条件pp，leave-one-out均值始终负。

实际4580唯一条件offset/256原生、GPU60.730731s含3秒清理预留；29016 SA层屏蔽key权重全0，8192物理mask/4096条件标签/CDF、512空间数组exact、798720逐候选质量/3456官方核对通过；独立score CPU43.079105s。0新encoder/RGB/专家/训练/空间或学生更新。启动前unique计费与allfalse复用修订完整保留，不涉及新GT/模型后的改规则。
