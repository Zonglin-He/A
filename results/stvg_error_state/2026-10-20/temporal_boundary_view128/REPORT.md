# 全跨度边界观察分配：HC128开发

同一冻结TA-STVG Source，原65偶索引33全局帧+32个起止边际分位数帧，保留原时间跨度。Actual真实新RGB、Nearest相同新时间戳但最近旧RGB；原八专家位置及E/Live三seed空间坐标全部固定。S1计划只无GT检查覆盖，不前向，未运行新OPD。

| 输出 | full vIoU% | tIoU% | 相对Native v增量pp [CI95] |
|---|---:|---:|---|
| Native | 27.928474 | 51.439514 | 0 |
| HiResAuxMix | 29.534908 | 55.920522 | +1.606434 [+0.591561,+2.666303] |
| ActualRawOldActions | 26.245665 | 47.816420 | -1.682808 [-2.734540,-0.712489] |
| ActualMix | 26.868979 | 49.243193 | -1.059495 [-1.877133,-0.351020] |
| ActualAlignedMix | 26.677520 | 48.919268 | -1.250954 [-2.128089,-0.474932] |
| NearestRawOldActions | 26.584468 | 48.431831 | -1.344006 [-2.324320,-0.423190] |
| NearestMix | 27.012723 | 49.401816 | -0.915751 [-1.724023,-0.209723] |
| NearestAlignedMix | 26.885136 | 49.254146 | -1.043338 [-1.926199,-0.231938] |

主ActualMix用物理时间上的异构假设分布在原2080动作上做精确MBR，.5p0+.5新q的期望utility；不把新旧矩阵直接相加。AlignedMix仅原始logits插值，不是概率运输。完整父均值选三个Actual读出，Nearest用于机制控制，HiResAuxMix为已有较强静态参照。硬读出与分布期望GT质量分别记录，二者不等价。

```json
{
  "primary": "ActualMix",
  "selected": "ActualMix",
  "selected_delta": {
    "mean": -1.0594948196336493,
    "CI95": [
      -1.8771332876934723,
      -0.3510201696110399
    ],
    "parents": 128
  },
  "decision": "no_supported_meaningful_readout",
  "ranking_diagnostics_available": true,
  "not_OPD": true,
  "independent_confirmation": false
}
```

## 成本与核验

```json
{
  "GPU_seconds": 716.4303875688929,
  "view_Source_calls": 256,
  "native_control_Source_calls": 1,
  "total_Source_offset_forwards": 514,
  "new_unique_Source_frames": 3871,
  "new_DINO": 0,
  "new_ReferDINO": 0,
  "new_network_training": 0,
  "spatial_updates": 0,
  "new_student_updates": 0
}
{
  "status": "pass",
  "official_DenseTube_checks": 2508,
  "spatial_arrays_exact": 512,
  "vectorized_quality_checks": 19456,
  "all128_before_GT": true,
  "numeric_independent": {
    "status": "pass",
    "all128_before_GT": true,
    "independent_schedules": 256,
    "independent_CDFs": 256,
    "max_CDF_error": 2.742404692007358e-16,
    "actual_RGB_checks": 2,
    "direct_heterogeneous_MBR_utilities": 53436,
    "two_full_old_action_grids_each_arm": true,
    "spatial_arrays_exact": 512,
    "no_GT_read": true,
    "new_model_calls": 0
  },
  "GT_provenance": {
    "data/hcstvg2_official_metadata/val_v2.json": "2e22d9370e311ca6c420b90867f5128507f5ab145e4fc120495b7a6336e41156"
  }
}
```

新增Source前向/真实唯一帧单独计费，不能说总观测或总计算不增加。128父均保留、先全封存后GT、三个空间seed父内平均；RC128已有反复开发曝光。无独立确认、无OPD成绩，不改空间或CURRENT/MAIN/PAPER。此配置不足不排除其他观察策略；不继续扫分配比例、采样相位或反馈强度挽救结果。


主ActualMix−Native −1.059495pp，区间全负；36改善/70损伤/22平，19父损伤超过3pp、9父超过5pp。保留RC042 −34.363681pp及其他严重负例。留一父均值范围为[-1.1413421164512036, -0.7972571325962512]pp，非删掉一个极端即可转正。ActualMix−NearestMix−.143744pp区间跨0，不足以将退化唯一归于新增RGB；Nearest同时间表也负，信息丢失/非均匀运动采样/ordinal与先验错配等共同因素尚未分离。

实际new-grid raw教师期望GTv−.478672pp区间跨0，AlignedMix期望−.527168pp区间负；没有建立更有用的稠密目标，不能只因S1计划128父都变化便开Live更新，也不能将此次未运行的OPD称失败。关闭该观察分配配置，不扫分配比例、相位或强度。更强HiResAuxMix既有+1.606434pp全部512空间输出保存指标逐值exact。

成本预算716.430388秒包含3秒cleanup预留，实际持有直到finally记录713.430388秒；非纯kernel计时。257全视图/514offset、16705 Source输入帧，3871新唯一Source时间点；原专家8位置固定，0专家/网络训练/空间/新学生更新。GPU4GiB floor在根进度点核查，host6/disk30GiB自动每view核查，allocator<=.8/5GiB reserve，不扩大为全部资源自动监护。

ActualMix−NearestMix 95%CI[−.382004,+.085539]，不支持新增RGB额外收益不等于二者等效；raw教师期望GTv CI[−1.471553,+.445305]，AlignedMix CI[−1.069616,−.012835]。
