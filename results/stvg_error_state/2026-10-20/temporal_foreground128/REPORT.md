# 冻结事件前景头：HC128能力检查

复用原第二pass时间输入；原生source已训练actionness头与endpoint头，不训练、不新读RGB/专家，不改任何空间坐标。RC128均为曝光开发父。Full/Fluct为旧报告既有规则，本轮检验当前基线与目标条件记忆，不能包装成新算法或OPD。

| 条件与读出 | full vIoU% | tIoU% | 相对原MBR v增量pp [CI95] |
|---|---:|---:|---|
| NativeMBR | 27.928474 | 51.439514 | 0 |
| NativeFull | 28.172563 | 51.517960 | +0.244089 [-1.336667,+1.996112] |
| NativeFluct | 28.563299 | 52.070928 | +0.634825 [-0.888002,+2.301143] |
| E_ROIFull | 26.979103 | 49.489736 | -0.949371 [-2.331264,+0.423698] |
| E_ROIFluct | 27.958633 | 51.497595 | +0.030159 [-1.148143,+1.194521] |
| Live41_ROIFull | 27.109813 | 49.714791 | -0.818661 [-2.226593,+0.588425] |
| Live41_ROIFluct | 28.023233 | 51.612606 | +0.094760 [-1.120622,+1.322003] |
| Live42_ROIFull | 27.122842 | 49.797503 | -0.805632 [-2.213240,+0.611434] |
| Live42_ROIFluct | 28.023233 | 51.612606 | +0.094760 [-1.120622,+1.322003] |
| Live43_ROIFull | 27.103536 | 49.666809 | -0.824938 [-2.236646,+0.589833] |
| Live43_ROIFluct | 28.064067 | 51.710896 | +0.135594 [-1.074775,+1.361828] |
| ReferROIFull | 27.447208 | 49.439285 | -0.481266 [-2.087732,+1.290762] |
| ReferROIFluct | 27.998592 | 50.705981 | +0.070118 [-1.476145,+1.797608] |
| ShiftedReferROIFull | 27.046623 | 49.731775 | -0.881851 [-2.094739,+0.301624] |
| ShiftedReferROIFluct | 27.282519 | 50.023106 | -0.645955 [-1.957409,+0.616194] |

全部条件和全部父保留，NativeFluct为先定主假设；四个eligible开发选优有选择偏差。空间s按构造固定。旧64为不同空间基线，不宣称本轮独立确认；条件变化不是监督新增。与错位/E/Own对照完整在全结果。

```json
{
  "selected": "NativeFluct",
  "development_selected_delta": {
    "mean": 0.6348250124364143,
    "CI95": [
      -0.8880019976435115,
      2.301142953771549
    ],
    "parents": 128
  },
  "primary_delta": {
    "mean": 0.6348250124364143,
    "CI95": [
      -0.8880019976435115,
      2.301142953771549
    ],
    "parents": 128
  },
  "decision": "no_supported_meaningful_candidate",
  "not_OPD": true,
  "independent_confirmation": false
}
```

## 计费与核验

```json
{
  "GPU_seconds": 85.27383880899288,
  "temporal_offset_decodes": 1792,
  "new_RGB": 0,
  "new_Source_encoder": 0,
  "new_experts": 0,
  "network_training": 0,
  "spatial_updates": 0,
  "attempts": [
    {
      "GPU_seconds": 85.27383880899288,
      "temporal_offset_decodes": 1792,
      "new_Source_encoder": 0,
      "new_DINO": 0,
      "new_ReferDINO": 0,
      "network_updates": 0,
      "spatial_updates": 0
    }
  ]
}
{
  "status": "pass",
  "official_DenseTube_checks": 7680,
  "frozen_spatial_arrays": 512,
  "independent_slice_projection_values": 1835008,
  "native_old_metrics_exact": true,
  "endpoints_all_conditions_exact_prior_round": true,
  "all128_before_GT": true,
  "weights_unchanged": true,
  "GT_provenance": {
    "data/hcstvg2_official_metadata/val_v2.json": "2e22d9370e311ca6c420b90867f5128507f5ab145e4fc120495b7a6336e41156"
  }
}
```

全逐父×空间×条件/读出signed和unknown保留，负尾完整在RESULTS_PUBLIC.json。生产/论文/研究配置不自动晋升。若能力成立还需确认和Direct/Fixed/Live，不能把projection有效称OPD有效。
