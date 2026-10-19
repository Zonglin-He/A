# 逐帧动作属性反馈：HC128开发能力探针

同一个冻结Source的34维多标签motion-attribute head，读取时间均值之前的类别logits；此前binary foreground和attention探针不是这个输出。不训练头/语义映射，不猜类名。不新增编码或专家，所有E/Live三seed空间数组逐字冻结。全查询融合和聚合BCE可能包含语言捷径，逐帧/ROI也非其原训练目标，不能称事件概率。

| 输出 | full vIoU% | tIoU% | 相对Native v增量pp [CI95] |
|---|---:|---:|---|
| Native | 27.928474 | 51.439514 | 0 |
| HiResMix | 29.161192 | 54.763483 | +1.232718 [+0.322089,+2.199435] |
| Student1 | 29.124965 | 54.744359 | +1.196491 [+0.314999,+2.130079] |
| GlobalQuery | 27.596167 | 50.782008 | -0.332307 [-0.985035,+0.167209] |
| GlobalMax | 27.531834 | 50.726012 | -0.396640 [-1.030971,+0.096603] |
| EQuery | 27.410518 | 50.706054 | -0.517956 [-1.232125,+0.172681] |
| LiveQuery | 27.484750 | 50.799890 | -0.443724 [-1.175036,+0.249531] |
| ReferQuery | 27.497441 | 50.708117 | -0.431033 [-1.140705,+0.247777] |
| ReferClassShift | 27.583882 | 50.828000 | -0.344592 [-1.013003,+0.338506] |
| ReferTimeShift | 28.136267 | 51.795837 | +0.207793 [-0.238418,+0.665835] |
| ReferOcc0 | 27.656522 | 50.997820 | -0.271952 [-1.015858,+0.475996] |
| ReferOcc1 | 27.663526 | 51.018008 | -0.264948 [-0.978586,+0.423576] |

主ReferQuery，固定lambda.5/原inside-minus-outside标准化/MBR，三eligible仅全开发父均值选，不逐视频挑。ClassShift/TimeShift为语义/时间错配对照；GlobalMax区别通用动作性；E/Live/Refer仅更改冻结区域先验。Occ0/Occ1来自真实既有学生时间状态的类别原型，不是新RGB或OPD已更新。对比同一S1和teacher分布期望GT质量完整保留。单点读出失败不普遍否定OPD；若教师本身无有利排序，不以扫梯度复活。

```json
{
  "primary": "ReferQuery",
  "selected": "ReferOcc0",
  "selected_delta": {
    "mean": -0.2719519392438186,
    "CI95": [
      -1.01585843717431,
      0.4759959328111246
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
  "GPU_seconds": 69.37000644695945,
  "motion_attribute_head_calls": 1280,
  "native_full_geometry_exact_checks": 256,
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
  "official_DenseTube_checks": 6144,
  "spatial_arrays_exact": 512,
  "direct_MBR_utilities": 27640,
  "vectorized_quality_checks": 9728,
  "max_error": 8.881784197001252e-16,
  "all128_before_GT": true,
  "numeric_independent": {
    "status": "pass",
    "all128_before_GT": true,
    "attention_layers_actual_numeric_checks": 12,
    "attention_scaled_maxerror": 3.358651744161738e-07,
    "fixed_attention_tolerance": 2e-05,
    "fractional_area_cells": 1734915,
    "geometry_maxerror": 2.9802322387695312e-08,
    "prototype_checks": 384,
    "interval_feedback_checks": 2396160,
    "posterior_maxerror": 1.2906342661267445e-15,
    "no_GT_read": true,
    "no_new_model_calls": true
  },
  "GT_provenance": {
    "data/hcstvg2_official_metadata/val_v2.json": "2e22d9370e311ca6c420b90867f5128507f5ab145e4fc120495b7a6336e41156"
  }
}
```

全部128父/所有signed、unknown、无更新和严重负尾保留，所有输出封存后才GT；384seed不是独立样本。已有编码/专家缓存上游成本不免费。未动CURRENT/MAIN/PAPER/冻结空间配置；只有开发大收益且语义对照支持才冻结父独立确认，不将探针当OPD成绩。
