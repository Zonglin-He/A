# 同帧联合运动编码时间教师：HC128开发

冻结同Source/448/原65实际帧；仅VideoSwin联合编码一次65，再物理选回33/32，原appearance/text/decoder长度不变。所有Source新box丢弃；既有E/Live三seed空间严格冻结。不是旧全模型dense2x，也不是完整OPD。全部128重复开发，全部输出seal后新GT。

| 读出 | 完整vIoU% | tIoU% | v增量pp [CI95] |
|---|---:|---:|---|
| Native | 27.928474 | 51.439514 | 0 |
| HiRes | 28.613528 | 54.809109 | +0.685054 [-1.092755,+2.524703] |
| HiResMix | 29.161192 | 54.763483 | +1.232718 [+0.322089,+2.199435] |
| Joint | 28.590573 | 55.073069 | +0.662100 [-1.237624,+2.616619] |
| JointMix | 29.279840 | 55.086551 | +1.351366 [+0.341943,+2.436427] |
| HiResFluct | 29.484879 | 56.462219 | +1.556405 [-0.561314,+3.756410] |
| JointFluct | 29.073658 | 56.109627 | +1.145184 [-1.043240,+3.437527] |

唯一主JointMix固定.5融合；Joint/JointFluct签名诊断不事后换主。既有HiRes/HiResMix/HiResFluct精确缓存复用。混合与读出增量不是蒸馏增量。原448已是开发选择，本轮匹配保持，仍有训练分布变化风险。只有同时>=2pp净收益、正CI且相对HiResMix正CI才冻候选确认。其他正趋势不追称通过。

```json
{
  "primary": "JointMix",
  "primary_delta": {
    "mean": 1.3513662171394956,
    "CI95": [
      0.3419432863697057,
      2.436426566269362
    ],
    "parents": 128
  },
  "context_increment": {
    "mean": 0.11864794727262205,
    "CI95": [
      -0.30732119936158786,
      0.5562876799164913
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
  "GPU_seconds": 623.6800512240734,
  "attempts": [
    {
      "GPU_seconds": 623.6800512240734,
      "highres_Source_attempts": 128,
      "native448_Source_attempts": 1,
      "Source_offset_forwards": 258,
      "actual_VideoSwin_forwards": 130,
      "joint_VideoSwin_input_frames": 8320,
      "new_DINO": 0,
      "new_ReferDINO": 0,
      "network_training": 0,
      "spatial_updates": 0
    }
  ],
  "highres_Source_attempts": 128,
  "native448_Source_attempts": 1
}
{
  "status": "pass",
  "official_DenseTube_checks": 3584,
  "spatial_arrays_exact": 512,
  "Source_CDF_checks": 128,
  "direct_MBR_utilities": 11518,
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
  "joint65_motion_traces": 128,
  "max_GPU_allocated_bytes": 15734430720,
  "native_metrics_exact": true,
  "all128_before_GT": true,
  "weights_unchanged": true,
  "GT_provenance": {
    "data/hcstvg2_official_metadata/val_v2.json": "2e22d9370e311ca6c420b90867f5128507f5ab145e4fc120495b7a6336e41156"
  }
}
```

新增Source/VideoSwin实际次数、单次联合65/两offset读取、历史Native448 control全部计入；既有Source448/原Source/DINO/Refer/空间fit成本不计本轮新增，不能当免费。负例/严重尾/unknown全见RESULTS_PUBLIC；未更新任何生产/论文/空间登记。
