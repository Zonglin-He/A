# 冻结空间轨迹条件的视频RGB视图：HC128开发

原65时间节点/原Source224/Vid权重/完整时间覆盖，八帧专家证据与全部空间坐标冻结；只从真实RGB形成整段共用上下文区域再编码。窗口来自冻结Live41在真实学生I0/I1支持上的框hull及插值邻居，固定2倍context，不用GT，不逐帧跟踪裁剪，不以新框替换空间。保留原Source双offset/FP32 replay和MBR，原Native完整视图exact控制。Crop0是初始学生，Crop1是旧实际40步OPD后的学生，仅作状态反馈能力，不是新OPD。

| 输出 | full vIoU% | tIoU% | 相对Native v增量pp [CI95] |
|---|---:|---:|---|
| Native | 27.928474 | 51.439514 | 0 |
| HiResMix | 29.161192 | 54.763483 | +1.232718 [+0.322089,+2.199435] |
| Student1 | 29.124965 | 54.744359 | +1.196491 [+0.314999,+2.130079] |
| Crop0 | 27.780568 | 50.978889 | -0.147906 [-0.487904,+0.189977] |
| Crop0Mix | 27.957952 | 51.444642 | +0.029478 [-0.134747,+0.205998] |
| Crop1 | 27.673603 | 51.051856 | -0.254871 [-0.643708,+0.100271] |
| Crop1Mix | 27.877921 | 51.426592 | -0.050553 [-0.215368,+0.123191] |
| SameS1Fixed | 28.623829 | 52.944302 | +0.695355 [+0.132502,+1.312320] |
| SameS1Refresh | 28.511498 | 52.931397 | +0.583024 [+0.010500,+1.202386] |

主Crop0Mix=.5p0+.5Crop0，eligible仅Crop0/Crop0Mix；所有父共同配置，不逐视频选。HiResMix为已有强参考。SameS1Fixed=.5p1+.5Crop0，SameS1Refresh=.5p1+.5Crop1，在同一真实S1上比较刷新；相同RGB窗口复用确定教师，不把重复调用当新反馈。原空间固定，窗口仅控制时间teacher观察；语言中的空间关系可能因裁剪改变，且错身份/错误初始时间可能导致无效区域。不同上下文与图像尺度同时改变，不能单独归因高分辨率。

```json
{
  "primary": "Crop0Mix",
  "selected": "Crop0Mix",
  "selected_delta": {
    "mean": 0.02947766195330087,
    "CI95": [
      -0.13474684439349485,
      0.20599779590956876
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
  "GPU_seconds": 249.10393270500936,
  "crop_Source_calls": 35,
  "native_control_Source_calls": 1,
  "total_Source_offset_forwards": 72,
  "new_DINO": 0,
  "new_ReferDINO": 0,
  "new_network_training": 0,
  "spatial_updates": 0,
  "new_student_updates": 0
}
{
  "status": "pass",
  "official_DenseTube_checks": 4608,
  "spatial_arrays_exact": 512,
  "direct_MBR_utilities": 20731,
  "vectorized_quality_checks": 9728,
  "max_error": 7.771561172376096e-16,
  "all128_before_GT": true,
  "numeric_independent": {
    "status": "pass",
    "all128_before_GT": true,
    "independent_windows": 256,
    "actual_first_video_and_crop_RGB_digests": 3,
    "independent_Source_envelope_CDFs": 256,
    "max_CDF_error": 2.7731489770552594e-16,
    "all_original_physical65_nodes": true,
    "unique_nonfull_views": 35,
    "changed_actual_student_windows": 13,
    "all_frozen_spatial_arrays_exact": true,
    "no_GT_read": true,
    "new_model_calls": 0
  },
  "GT_provenance": {
    "data/hcstvg2_official_metadata/val_v2.json": "2e22d9370e311ca6c420b90867f5128507f5ab145e4fc120495b7a6336e41156"
  }
}
```

全部128父/所有signed、unknown、无变化和严重负尾保留。所有输出seal后GT，三个空间seed父内平均不计独立N。额外Source重编码实际计费，既有专家缓存不免费。未改CURRENT/MAIN/PAPER/冻结空间配置；若teacher自身无有利排序，不直接启动OPD或扫温度。确认仍需未参与当前选择的父，历史曝光披露。
