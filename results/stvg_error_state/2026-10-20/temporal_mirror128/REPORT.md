# 同Source真实反射观察时间教师：HC128开发

原65实际帧/448/同Vid权重；119条原RGB水平反射重编码，9条caption或subject含方向词预锁跳过并保持旧448教师。原时间轴不翻转、文本不改、新Source框不输出；E/Live三seed空间严格冻结。官方训练含hflip(.7)给接口依据，不保证目标教师更强；旧空间SSL视图不等于本轮时间概率融合。不是OPD。全部128开发输出先seal后GT。

| 读出 | 完整vIoU% | tIoU% | v增量pp [CI95] |
|---|---:|---:|---|
| Native | 27.928474 | 51.439514 | 0 |
| HiRes | 28.613528 | 54.809109 | +0.685054 [-1.092755,+2.524703] |
| HiResMix | 29.161192 | 54.763483 | +1.232718 [+0.322089,+2.199435] |
| Mirror | 28.212959 | 52.962846 | +0.284486 [-1.531113,+2.153209] |
| MirrorEnsemble | 28.928410 | 54.695527 | +0.999936 [-0.755453,+2.820767] |
| MirrorMix | 29.138719 | 54.482085 | +1.210245 [+0.331810,+2.128412] |

唯一主MirrorMix=.5p0+.25q448+.25qFlip448；MirrorEnsemble=.5q448+.5qFlip448。预锁方向词9条m=h不删父，也不按旧得失筛选。只原65决策与同MBR；单配置不扫融合权重或再换主。主>=2pp/正CI且对HiResMix正CI才冻结确认。Teacher/readout不是OPD贡献，普通hard读出不通过亦不否定所有OPD。

```json
{
  "primary": "MirrorMix",
  "primary_delta": {
    "mean": 1.2102449946465974,
    "CI95": [
      0.3318101093974519,
      2.128411943966042
    ],
    "parents": 128
  },
  "reflection_increment": {
    "mean": -0.022473275220276007,
    "CI95": [
      -0.31522658524069586,
      0.267031351903294
    ],
    "parents": 128
  },
  "decision": "no_supported_meaningful_point_readout",
  "point_readout_failure_does_not_prove_OPD_failure": true,
  "not_OPD": true,
  "independent_confirmation": false
}
```

## 成本与核验

```json
{
  "GPU_seconds": 507.97269966220483,
  "attempts": [
    {
      "GPU_seconds": 507.97269966220483,
      "mirror_Source_attempts": 119,
      "Source_offset_forwards": 238,
      "new_DINO": 0,
      "new_ReferDINO": 0,
      "network_training": 0,
      "spatial_updates": 0
    }
  ],
  "mirror_Source_attempts": 119
}
{
  "status": "pass",
  "official_DenseTube_checks": 3072,
  "spatial_arrays_exact": 512,
  "Source_CDF_checks": 119,
  "direct_MBR_utilities": 13821,
  "vectorized_quality_checks": 9728,
  "max_error": 6.661338147750939e-16,
  "native_actual448_control": {
    "status": "pass",
    "alias": "RC001",
    "Source448_logits_actions_q_exact": true,
    "trace": {
      "calls": [
        {
          "offset": 0,
          "num_frames": 33,
          "physical_indices": [
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62,
            64
          ],
          "input_sha256": "b33c89c5f831b204885137ef4011ef7f2bc0592b79c148fcc4c881e988546513",
          "output_shape": [
            33,
            768,
            14,
            19
          ],
          "output_sha256": "0d468d9d2c36a20628a95e80d5b308cd7086b470dca966b287edb997a90362e2"
        },
        {
          "offset": 1,
          "num_frames": 32,
          "physical_indices": [
            1,
            3,
            5,
            7,
            9,
            11,
            13,
            15,
            17,
            19,
            21,
            23,
            25,
            27,
            29,
            31,
            33,
            35,
            37,
            39,
            41,
            43,
            45,
            47,
            49,
            51,
            53,
            55,
            57,
            59,
            61,
            63
          ],
          "input_sha256": "eb70bb13759f7b5aab0b8889367fbdd92773673ba1771a6038cd472815cc53c3",
          "output_shape": [
            32,
            768,
            14,
            19
          ],
          "output_sha256": "eb26cbe632686906fcf6ec4e21a4504343675c6b2a04e201f5e7564b40a410e9"
        }
      ],
      "actual_encoder_calls": 2,
      "joint": false,
      "full_shape": [
        65,
        3,
        448,
        597
      ],
      "full_sha256": "9ee0880692608f109dee45df1a39e6d4d24d7f39bc4a28d1ed7320adaf176f8d",
      "restored": true
    },
    "seconds": 3.5689894719980657,
    "GT_read": false
  },
  "reflected119_and_veto9": true,
  "max_GPU_allocated_bytes": 8673604096,
  "native_metrics_exact": true,
  "all128_before_GT": true,
  "weights_unchanged": true,
  "GT_provenance": {
    "data/hcstvg2_official_metadata/val_v2.json": "2e22d9370e311ca6c420b90867f5128507f5ab145e4fc120495b7a6336e41156"
  }
}
```

119次真实Source反射观察有新增编码成本；旧Source/DINO/Refer/空间fit缓存不是免费，新增DINO/Refer/训练/空间更新0。原生448 control精确核验复用上轮。所有signed/unknown/尾部见全量JSON；9veto仍128主分母；所有生产/论文/空间登记不改。
