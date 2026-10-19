# 固定空间下的时空联合决策效用：HC128开发

空间预测/专家/观测位置/模型权重全部不动。现有Temporal MBR只评价区间交并比；此探针将同一个事件区间后验与学生-ReferDINO空间IoU代理组合为expected pseudo-tube utility。空间IoU不是事件概率，也不是校准的GT质量。八帧teacher只在其支持内插值，不能称逐帧独立专家观察。无新模型/训练/专家/OPD。

| 输出 | full vIoU% | tIoU% | event recall% | v增量pp [CI95] |
|---|---:|---:|---:|---|
| N_Time | 27.928474 | 51.439514 | 72.973346 | 0 |
| N_Risk | 26.769750 | 46.984100 | 62.662194 | -1.158723 [-2.693297,+0.376868] |
| N_Shift | 22.920805 | 41.590489 | 60.966647 | -5.007669 [-7.603024,-2.706196] |
| N_Flat | 27.928474 | 51.439514 | 72.973346 | +0.000000 [+0.000000,+0.000000] |
| N_Target | 27.628718 | 50.999113 | 72.493071 | -0.299756 [-0.761836,+0.009693] |
| N_FlatTarget | 28.018063 | 51.523468 | 73.211481 | +0.089589 [-0.016575,+0.204783] |
| H_Time | 29.161192 | 54.763483 | 78.356197 | +1.232718 [+0.322089,+2.199435] |
| H_Risk | 28.153600 | 50.559398 | 69.094206 | +0.225127 [-1.298680,+1.722772] |
| H_Shift | 23.998385 | 44.227940 | 63.316390 | -3.930089 [-6.778068,-1.420928] |
| H_Flat | 29.161192 | 54.763483 | 78.356197 | +1.232718 [+0.322089,+2.199435] |
| H_Target | 29.190646 | 54.568746 | 77.963543 | +1.262172 [+0.325654,+2.259742] |
| H_FlatTarget | 29.144436 | 54.739145 | 78.378120 | +1.215962 [+0.274307,+2.210271] |
| A_Time | 29.534908 | 55.920522 | 79.488810 | +1.606434 [+0.591561,+2.666303] |
| A_Risk | 28.267355 | 50.943317 | 69.486013 | +0.338881 [-1.141495,+1.758728] |
| A_Shift | 23.712874 | 43.829962 | 62.894593 | -4.215600 [-7.063796,-1.658073] |
| A_Flat | 29.534908 | 55.920522 | 79.488810 | +1.606434 [+0.591561,+2.666303] |
| A_Target | 29.337431 | 55.275933 | 78.641021 | +1.408957 [+0.334287,+2.518465] |
| A_FlatTarget | 29.582262 | 55.932889 | 79.521034 | +1.653788 [+0.624027,+2.734785] |

N原Native，H旧HiResMix，A旧HiResAuxMix；Time原timeMBR、Risk最大expected pseudo-tube utility、Shift时间错位空间质量、Flat相同均值常曲线必须回到Time。Target为prior×joint utility归一化后的静态诊断，FlatTarget控制仅self-risk重加权；二者不eligible挑方法。主A_Risk，三个Risk全128共同选择，不按父/seed挑赢家。与各自Time的差才是新增联合读出。

GT只封存后解释，全部父/signed/unknown/尾部保留。若fullv上升但t/recall下降，结论是联合指标取舍，不能称事件边界理解更好；固定空间真实节点指标全部不变。teacher软目标期望与风险排序诊断也保留，不用单点阴性宣布所有OPD失败。

```json
{
  "primary": "A_Risk",
  "selected": "A_Risk",
  "selected_delta": {
    "mean": 0.3388806991004179,
    "CI95": [
      -1.1414952897588213,
      1.758728227554654
    ],
    "parents": 128
  },
  "selected_temporal_delta": {
    "mean": -0.4961973601090218,
    "CI95": [
      -3.3584996719854514,
      2.333214425894601
    ],
    "parents": 128
  },
  "decision": "no_supported_2pp_joint_decision",
  "point_readout_not_OPD": true,
  "spatial_proxy_not_event_probability": true,
  "temporal_tradeoff_must_be_reported": true,
  "independent_confirmation": false
}
```

## 成本与审计

```json
{
  "CPU_seconds": 55.68826191709377,
  "GPU_seconds": 0,
  "joint_utility_computations": 4608,
  "flat_original_time_exact_checks": 1536,
  "new_Source_encoder": 0,
  "new_RGB": 0,
  "new_DINO": 0,
  "new_ReferDINO": 0,
  "new_training": 0,
  "spatial_updates": 0,
  "new_OPD_updates": 0
}
{
  "status": "pass",
  "official_DenseTube_unique_interval_checks": 3827,
  "spatial_arrays_exact": 512,
  "all_outputs_have_same_fixed_spatial_sIoU": true,
  "native_metrics_exact": true,
  "vectorized_GT_quality_checks": 9728,
  "all128_before_GT": true,
  "risk_independent": {
    "status": "pass",
    "all128_before_GT": true,
    "independent_scalar_quality_curves": 512,
    "actual_physical_frame_qualities": 269116,
    "max_curve_error": 4.440892098500626e-15,
    "direct_utility_values": 89082,
    "max_direct_utility_error": 2.942091015256665e-15,
    "full_primary_action_grids": 3,
    "flat_original_time_exact_checks": 1536,
    "target_distributions": 3072,
    "no_GT": true,
    "no_model_calls": true,
    "all_spatial_arrays_exact": true
  },
  "GT_provenance": {
    "data/hcstvg2_official_metadata/val_v2.json": "2e22d9370e311ca6c420b90867f5128507f5ab145e4fc120495b7a6336e41156"
  }
}
```

所有成本为缓存后新增CPU推理，原Source/ReferDINO/空间适应成本不是免费。模型/空间/CURRENT/MAIN/PAPER未变。>=2pp及正区间只支持冻结父独立确认；如果确认成立再设计实际Direct/Fixed/Live，静态联合读出不算OPD贡献。
