# 目标×学生事件状态条件时间query：HC128能力检查

仅已有Source224融合motion/text；冻结Source原始时间decoder/head，保留全局memory/time/mask。用已有Refer八框插值区域和当前时间占有率形成norm匹配query。Source空间decoder0次；最终空间仍原E/Live三数组。所有网络参数不动。G/R分别全图/Refer区域，All/Occ0分别全时/原p0的节点事件占有率，Occ1来自旧真实OPD Live41最终分布。另对S0/S1各四个实际采样区间查询教师，不用GT选候选。不是新OPD成绩。

| 读出 | 完整vIoU% | tIoU% | v增量pp [CI95] |
|---|---:|---:|---|
| Native | 27.928474 | 51.439514 | 0 |
| Student1 | 29.124965 | 54.744359 | +1.196491 [+0.314999,+2.130079] |
| GAll | 28.053051 | 51.655189 | +0.124577 [-0.324135,+0.569367] |
| GOcc0 | 27.966009 | 51.511643 | +0.037535 [-0.412800,+0.435969] |
| RAll | 27.968325 | 51.528094 | +0.039851 [-0.437757,+0.433228] |
| ROcc0 | 28.017338 | 51.525688 | +0.088864 [-0.396789,+0.475003] |
| ROcc1 | 28.022613 | 51.569364 | +0.094139 [-0.395284,+0.484661] |
| Sample0_0 | 28.390603 | 51.919755 | +0.462129 [-0.260894,+1.386351] |
| Sample0_1 | 28.124310 | 51.666894 | +0.195837 [-0.324854,+0.648088] |
| Sample0_2 | 28.096167 | 51.625641 | +0.167694 [-0.332313,+0.609865] |
| Sample0_3 | 28.033590 | 51.618315 | +0.105116 [-0.357017,+0.496731] |
| Sample1_0 | 28.327182 | 51.890771 | +0.398708 [-0.316404,+1.392508] |
| Sample1_1 | 28.051219 | 51.582839 | +0.122745 [-0.329394,+0.535302] |
| Sample1_2 | 28.087089 | 51.565202 | +0.158615 [-0.336221,+0.594204] |
| Sample1_3 | 27.985661 | 51.529673 | +0.057187 [-0.421672,+0.425659] |
| GAllMix | 27.961375 | 51.511854 | +0.032901 [-0.312339,+0.294399] |
| GOcc0Mix | 27.811131 | 51.267071 | -0.117343 [-0.431048,+0.078558] |
| RAllMix | 27.827696 | 51.276531 | -0.100778 [-0.478083,+0.161931] |
| ROcc0Mix | 27.955739 | 51.492870 | +0.027266 [-0.349571,+0.293016] |
| ROcc1Mix | 27.975313 | 51.516136 | +0.046839 [-0.331726,+0.311734] |
| Sample0Mean | 28.056011 | 51.534601 | +0.127537 [-0.368857,+0.536350] |
| Sample1Mean | 28.026405 | 51.552755 | +0.097932 [-0.376073,+0.496692] |

所有Mix固定.5p0+.5对应teacher；Sample均值为四份teacher posterior平均，不是学生最终适应。两个eligible仅GOcc0Mix/ROcc0Mix，按全128开发父均值选，点CI非选优校正。>=2pp且CI正才冻结确认；WHERE→WHEN因果还看相对GOcc0Mix/RAllMix，不用总收益替代。软目标期望/采样纠错与点读出分开；hard阴性不等于OPD普遍失败，target变化也不等于正确。旧ROIquery32与固定memory128负结果仍成立于各自配置。

```json
{
  "primary": "ROcc0Mix",
  "selected_development_route": "ROcc0Mix",
  "selected_delta": {
    "mean": 0.02726556124621446,
    "CI95": [
      -0.34957110061459146,
      0.2930155241313579
    ],
    "parents": 128
  },
  "decision": "no_supported_meaningful_point_readout",
  "sample_teacher_expected_quality_vs_fixed": {
    "Sample0Mean": {
      "Native": {
        "v": {
          "mean": 0.058784693849083,
          "CI95": [
            -0.22448602948269344,
            0.30533030994150656
          ],
          "parents": 128
        },
        "t": {
          "mean": 0.12977205641991368,
          "CI95": [
            -0.25826897298482016,
            0.48607319807100113
          ],
          "parents": 128
        }
      },
      "ROcc0": {
        "v": {
          "mean": -0.013352180454837381,
          "CI95": [
            -0.13445138102881984,
            0.06896870197692907
          ],
          "parents": 128
        },
        "t": {
          "mean": 0.011683462854546895,
          "CI95": [
            -0.1447342104798764,
            0.13541373685932812
          ],
          "parents": 128
        }
      }
    },
    "Sample1Mean": {
      "Native": {
        "v": {
          "mean": 0.054631407707928806,
          "CI95": [
            -0.1977556570688686,
            0.27957831278834383
          ],
          "parents": 128
        },
        "t": {
          "mean": 0.14629682181040432,
          "CI95": [
            -0.23342967552168345,
            0.5044371080900465
          ],
          "parents": 128
        }
      },
      "ROcc0": {
        "v": {
          "mean": -0.017505466595991562,
          "CI95": [
            -0.10529370207394236,
            0.047174054408081346
          ],
          "parents": 128
        },
        "t": {
          "mean": 0.02820822824503754,
          "CI95": [
            -0.11567666015663257,
            0.20153645096534625
          ],
          "parents": 128
        }
      }
    },
    "fresh_occupancy_vs_fixed": {
      "v": {
        "mean": 0.019468624502289283,
        "CI95": [
          -0.02568276394994676,
          0.07078716833630426
        ],
        "parents": 128
      },
      "t": {
        "mean": 0.051878341825080575,
        "CI95": [
          -0.02000201442934005,
          0.13652786583751503
        ],
        "parents": 128
      }
    }
  },
  "teacher_state_changes_do_not_imply_correctness": true,
  "not_OPD": true,
  "independent_confirmation": false
}
```

## 计费与核验

```json
{
  "GPU_seconds": 86.34479788085446,
  "conditioned_temporal_offset_decodes": 3328,
  "native_temporal_checks": 256,
  "spatial_attention_head_calls": 256,
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
  "official_DenseTube_checks": 11776,
  "spatial_arrays_exact": 512,
  "teacher_CDF_checks": 1664,
  "direct_MBR_utilities": 50686,
  "vectorized_quality_checks": 9728,
  "max_error": 8.881784197001252e-16,
  "native224_temporal_checks": 256,
  "query_conditions_per_view": 13,
  "native_metrics_exact": true,
  "all128_before_GT": true,
  "weights_unchanged": true,
  "GT_provenance": {
    "data/hcstvg2_official_metadata/val_v2.json": "2e22d9370e311ca6c420b90867f5128507f5ab145e4fc120495b7a6336e41156"
  }
}
```

新增为冻结TemporalDecoder/原spatial-attention classifier调用，无Source encoder/RGB/DINO/Refer/网络或空间训练；旧缓存计算不是免费。所有128先seal后GT；全signed/unknown/负尾及教师期望质量见完整JSON，生产/论文/空间登记不改。
