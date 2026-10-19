# 目标区域运动池化：HC128能力检查

复用原第二pass融合运动与文本，区域池化后读取已有source监督的TemporalSampling。全局原算子/valid全局/对象区域/横移错误区域同分类器，原endpoint prior完全相同；不训练、不读新RGB/专家、不改空间。RC128均为曝光开发父。Full/Fluct为旧报告既有规则，本轮检验当前基线与目标条件记忆，不能包装成新算法或OPD。

| 条件与读出 | full vIoU% | tIoU% | 相对原MBR v增量pp [CI95] |
|---|---:|---:|---|
| NativeMBR | 27.928474 | 51.439514 | 0 |
| NativeGlobalFull | 28.059682 | 49.910501 | +0.131208 [-1.824467,+2.271874] |
| NativeGlobalFluct | 28.316537 | 50.809311 | +0.388063 [-1.583188,+2.520918] |
| ValidGlobalFull | 28.059682 | 49.910501 | +0.131208 [-1.824467,+2.271874] |
| ValidGlobalFluct | 28.316537 | 50.809311 | +0.388063 [-1.583188,+2.520918] |
| E_ROIFull | 27.409984 | 49.086711 | -0.518490 [-2.612740,+1.695604] |
| E_ROIFluct | 27.667878 | 49.709882 | -0.260596 [-2.391633,+1.986575] |
| Live41_ROIFull | 27.448191 | 48.930718 | -0.480283 [-2.576095,+1.723415] |
| Live41_ROIFluct | 27.797896 | 49.900456 | -0.130578 [-2.224328,+2.099247] |
| Live42_ROIFull | 27.456583 | 48.970842 | -0.471891 [-2.567151,+1.734714] |
| Live42_ROIFluct | 27.799599 | 49.908990 | -0.128875 [-2.222668,+2.100913] |
| Live43_ROIFull | 27.462596 | 48.993686 | -0.465878 [-2.562372,+1.737924] |
| Live43_ROIFluct | 27.799599 | 49.908990 | -0.128875 [-2.222668,+2.100913] |
| ReferROIFull | 27.440890 | 49.048069 | -0.487583 [-2.422957,+1.570319] |
| ReferROIFluct | 27.946775 | 50.108459 | +0.018301 [-2.030462,+2.253588] |
| ShiftedReferROIFull | 27.376691 | 48.793274 | -0.551782 [-2.550956,+1.629074] |
| ShiftedReferROIFluct | 27.533961 | 48.980502 | -0.394513 [-2.466234,+1.819870] |

全部条件和全部父保留，ReferROIFluct为先定主假设；四个eligible开发选优有选择偏差。空间s按构造固定。旧64为不同空间基线，不宣称本轮独立确认；池化变化不是新模型或新增视觉观测。与错位/E/Own对照完整在全结果。

```json
{
  "selected": "NativeGlobalFluct",
  "development_selected_delta": {
    "mean": 0.38806274233552507,
    "CI95": [
      -1.583187594410951,
      2.5209176325020124
    ],
    "parents": 128
  },
  "primary_delta": {
    "mean": 0.018301051798870427,
    "CI95": [
      -2.030462157755898,
      2.2535884303683162
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
  "GPU_seconds": 82.3163278491702,
  "temporal_offset_decodes": 0,
  "temporal_classifier_forwards": 2048,
  "equivalent_native_classifier_checks": 256,
  "new_RGB": 0,
  "new_Source_encoder": 0,
  "new_experts": 0,
  "network_training": 0,
  "spatial_updates": 0,
  "attempts": [
    {
      "GPU_seconds": 82.3163278491702,
      "temporal_offset_decodes": 0,
      "temporal_classifier_forwards": 2048,
      "equivalent_native_classifier_checks": 256,
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
  "official_DenseTube_checks": 8704,
  "frozen_spatial_arrays": 512,
  "normalized_pool_checks": 58240,
  "native_prepool_exact_checks": 256,
  "independent_slice_projection_values": 2097152,
  "native_old_metrics_exact": true,
  "native_endpoints_same_prior_all_conditions": true,
  "all128_before_GT": true,
  "weights_unchanged": true,
  "GT_provenance": {
    "data/hcstvg2_official_metadata/val_v2.json": "2e22d9370e311ca6c420b90867f5128507f5ab145e4fc120495b7a6336e41156"
  }
}
```

全逐父×空间×条件/读出signed和unknown保留，负尾完整在RESULTS_PUBLIC.json。生产/论文/研究配置不自动晋升。若能力成立还需确认和Direct/Fixed/Live，不能把projection有效称OPD有效。

独立NumPy面积交集核验49920位置，权重max误差1.192093e-7；194空ROI使用预定ValidGlobal。ValidGlobal原预测使用global_pool，对应审计字段valid_weights；其meta.weights为并行计算而未用于该分类器输入的ReferROI权重。此字段澄清不修改任何封存预测或配置，完整核验码和JSON同目录公开。
