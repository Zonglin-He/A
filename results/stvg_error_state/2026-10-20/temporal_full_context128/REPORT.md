# 原65帧完整Source上下文教师：HC128开发

同Source224/同原65帧/同文本/同checkpoint；一个完整65帧Source前向与原FP32两pass replay，而非两个独立33/32再envelope。同时改变完整多模态融合上下文与decoder长度，不能将效果单独归某一层。并非dense129新增帧，也非新OPD。第一旧NH001完整65 stream FP32 logits必须exact复现历史缓存，不读它的GT；原旧Dense−9.2619是HC16而不是HC32，不能自动判单stream。当前128通常是不同clip，不能偷用旧16logits替代；本轮全128重算Source。空间E/Live三seed数组严格不动，新Source框全部丢弃。

| 输出 | 完整vIoU% | tIoU% | v增量pp [CI95] |
|---|---:|---:|---|
| Native | 27.928474 | 51.439514 | 0 |
| Full65 | 21.589732 | 37.870519 | -6.338742 [-8.915228,-3.766580] |
| Full65Mix | 27.154510 | 49.237089 | -0.773964 [-1.808453,+0.283812] |
| HiResMix | 29.161192 | 54.763483 | +1.232718 [+0.322089,+2.199435] |

Full65Mix固定.5p0+.5q65；只两个eligible在整个128开发父均值选。原65物理决策支持、相同MBR，没有GT选流/逐视频选赢家。缓存HiResMix强比较保留。点结果/teacher分布期望GT质量分别报告，均不是学生已完成OPD。>=2pp且CI正才冻结独立确认，开发选优CI未经多重校正，仍需新父确认。

```json
{
  "primary": "Full65Mix",
  "selected_development_route": "Full65Mix",
  "selected_delta": {
    "mean": -0.773963865633309,
    "CI95": [
      -1.8084525483623841,
      0.28381224839324337
    ],
    "parents": 128
  },
  "decision": "no_supported_meaningful_point_readout",
  "posterior_quality_diagnostic": {
    "Full65": {
      "v": {
        "mean": -9.309696975149047,
        "CI95": [
          -11.181347148693632,
          -7.514509068447478
        ],
        "parents": 128
      },
      "t": {
        "mean": -18.081789845957076,
        "CI95": [
          -20.59522628122103,
          -15.551646781779592
        ],
        "parents": 128
      }
    },
    "Full65Mix": {
      "v": {
        "mean": -4.654848487574523,
        "CI95": [
          -5.590673574346814,
          -3.7572545342237396
        ],
        "parents": 128
      },
      "t": {
        "mean": -9.04089492297854,
        "CI95": [
          -10.297613140610515,
          -7.775823390889795
        ],
        "parents": 128
      }
    },
    "HiResMix": {
      "v": {
        "mean": 0.1965867635077584,
        "CI95": [
          -0.5546348005989372,
          0.9558333339511718
        ],
        "parents": 128
      },
      "t": {
        "mean": 1.0829412063146677,
        "CI95": [
          -0.111756392295481,
          2.2986681501083135
        ],
        "parents": 128
      }
    }
  },
  "point_readout_failure_does_not_prove_OPD_failure": true,
  "not_OPD": true,
  "independent_confirmation": false
}
```

## 成本与核验

```json
{
  "GPU_seconds": 244.88122086506337,
  "current_full65_Source_attempts": 128,
  "old_full65_control_attempts": 1,
  "full_Source_encoder_forwards": 129,
  "AMP_decoder_passes": 258,
  "FP32_replay_decoder_passes": 258,
  "new_DINO": 0,
  "new_ReferDINO": 0,
  "new_network_training": 0,
  "spatial_updates": 0
}
{
  "status": "pass",
  "official_DenseTube_checks": 2048,
  "spatial_arrays_exact": 512,
  "single_head_softmax_checks": 128,
  "direct_MBR_utilities": 9212,
  "vectorized_quality_checks": 9728,
  "max_error": 6.661338147750939e-16,
  "native_full65_control": {
    "status": "pass",
    "first_old_numeric_control": "NH001",
    "FP32_full65_logits_exact": true,
    "original65_nodes_exact": true,
    "old_actions_gates_not_cached": true,
    "new_actions_finite": true,
    "seconds": 0.918413280043751,
    "info": {
      "frame_count": 65,
      "normalized_shape": [
        65,
        3,
        224,
        291
      ],
      "feature_shape": [
        7,
        10
      ],
      "full_Source_forward_calls": 1,
      "FP32_original_two_pass_replay": true,
      "original_reference_function_only_output_assembly_changed": true
    },
    "GT_read": false
  },
  "max_GPU_allocated_bytes": 5751874048,
  "native_metrics_exact": true,
  "all128_before_GT": true,
  "weights_unchanged": true,
  "GT_provenance": {
    "data/hcstvg2_official_metadata/val_v2.json": "2e22d9370e311ca6c420b90867f5128507f5ab145e4fc120495b7a6336e41156"
  }
}
```

新增Source实际完整65调用与旧无GT数值控制计费。原DINO/Refer/空间fit和Source缓存成本不是免费；无新专家/网络训练/空间更新。预测全128 seal后新GT，所有signed/unknown/尾部保留，生产/论文/空间登记不改。
