# 完整描述与静态对象描述：HC128时间对比

同一Source、原65物理帧，两offset。静态文本仅使用旧缓存parses.old.phrase，127可用/1缺失noop，不补类别、不用GT改提示。所有新空间输出丢弃，最终空间四轨迹保持冻结。Contrast=p0²/qStatic，仅固定单位系数；PriorSquare=p0²控制单纯锐化。全部仍用同一finalMBR，不把语义消融等同纯事件因果分解或校准概率。本轮仅能力探针，非OPD。

| 读出 | 完整vIoU% | tIoU% |
|---|---:|---:|
| Native | 27.928474 | 51.439514 |
| PriorSquare | 27.595133 | 50.639424 |
| Contrast | 22.992588 | 42.117526 |

```json
{
  "primary": "Contrast",
  "delta": {
    "mean": -4.935885527727274,
    "CI95": [
      -6.762099784722605,
      -3.0736168854895163
    ],
    "parents": 128
  },
  "semantic_increment_over_sharpening": {
    "mean": -4.602545132126291,
    "CI95": [
      -6.425445738589111,
      -2.771425870936399
    ],
    "parents": 128
  },
  "decision": "no_supported_meaningful_semantic_increment",
  "not_OPD": true,
  "independent_confirmation": false
}
```

全部RC128反复曝光开发，未做独立确认。任何正均值不自动晋升；能力成立后仍需冻结确认/Direct/Fixed/Live才可谈OPD。完整signed/unknown、双对照与负尾在RESULTS_PUBLIC.json，不删无反馈父或挑种子。

## 成本与审计

```json
{
  "GPU_seconds": 624.8036127309315,
  "new_Source_queries": 127,
  "Source_offset_forwards": 254,
  "new_DINO": 0,
  "new_ReferDINO": 0,
  "network_training": 0,
  "spatial_updates": 0,
  "attempts": [
    {
      "GPU_seconds": 624.8036127309315,
      "new_Source_queries": 127,
      "Source_offset_forwards": 254,
      "new_DINO": 0,
      "new_ReferDINO": 0,
      "network_training": 0,
      "spatial_updates": 0
    }
  ]
}
{
  "status": "pass",
  "official_DenseTube_checks": 1536,
  "spatial_arrays_exact": 512,
  "independent_static_CDF": 127,
  "longdouble_direct_posterior_checks": 256,
  "direct_MBR_utilities": 6910,
  "maximum_error": 6.661338147750939e-16,
  "missing_static_noop": 1,
  "all128_before_GT": true,
  "native_metrics_exact": true,
  "weights_unchanged": true,
  "GT_provenance": {
    "data/hcstvg2_official_metadata/val_v2.json": "2e22d9370e311ca6c420b90867f5128507f5ab145e4fc120495b7a6336e41156"
  }
}
```

同RGB被新文本重新编码照常计费，不称免费缓存推理；不新增DINO/Refer/预训练权重或空间state更新。原Source checkpoint、current spatial config和登记不变。
