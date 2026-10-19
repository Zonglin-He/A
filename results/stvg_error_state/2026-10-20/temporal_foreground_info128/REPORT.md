# 原生前景信息资格：HC128纯缓存诊断

没有新最终区间、MBR赢家或OPD成绩。Source既有Action/两前置motion/appearance BCE头；原65节点、全部128父及3冻结空间seed，GT只在所有字段/能量封存后读取。ActionEOS=z+log(.3)仅理想class-weight反校准假设，官方配置不代表真实训练日志或目标域校准。旧structured_temporal已有Bernoulli结构，不声称新方法。

| field | AUROC (定义父数) | Brier | 同长度pair正确率 | 有益/有害候选AUC (定义父数) |
|---|---|---:|---:|---|
| Action | {'mean': 0.8091471397895001, 'defined_parents': 128} | 0.18401346147877912 | 0.7570926115937463 | {'mean': 0.6813081582060789, 'defined_parents': 105} |
| ActionEOS | {'mean': 0.8091471397895001, 'defined_parents': 128} | 0.22187676707297826 | 0.7570926115937463 | {'mean': 0.542423382817018, 'defined_parents': 105} |
| ActionFlat | {'mean': 0.5, 'defined_parents': 128} | 0.2595926510865445 | 0.5 | {'mean': 0.42233959115024056, 'defined_parents': 105} |
| ActionShift | {'mean': 0.2686268925054386, 'defined_parents': 128} | 0.3727829712698234 | 0.2739056964699374 | {'mean': 0.2708178927076225, 'defined_parents': 105} |
| Motion | {'mean': 0.7514064527987376, 'defined_parents': 128} | 0.26943050973016897 | 0.7064389874409962 | {'mean': 0.44196229800049736, 'defined_parents': 105} |
| MotionFlat | {'mean': 0.5, 'defined_parents': 128} | 0.311651705187154 | 0.5 | {'mean': 0.3407460476898, 'defined_parents': 105} |
| MotionShift | {'mean': 0.36194552679754505, 'defined_parents': 128} | 0.32149773517922897 | 0.3371379591566063 | {'mean': 0.30302785005774796, 'defined_parents': 105} |
| Appearance | {'mean': 0.7590826260020319, 'defined_parents': 128} | 0.271086324482225 | 0.7276774062019506 | {'mean': 0.4456233984287482, 'defined_parents': 105} |
| AppearanceFlat | {'mean': 0.5, 'defined_parents': 128} | 0.3124456260277372 | 0.5 | {'mean': 0.3407460476898, 'defined_parents': 105} |
| AppearanceShift | {'mean': 0.3539368743788879, 'defined_parents': 128} | 0.3246232212987582 | 0.3159043046264025 | {'mean': 0.29636041177894284, 'defined_parents': 105} |

所有区间在原65格点2080支持；同长度指相同节点数j−i（物理时长可差1帧），纠错候选标记相对同父原Native完整v至少±2pp，GT仅诊断不作在线门控。所有可区分GT质量的同长度pair都保留，教师tie算.5；条件未定义记null不填0。总体排序可混有时长偏置，必须与同长度和Shift控制分开。三个seed只父内平均，独立n仍128。

EOS修正的AUROC及同长度排序逐父严格相同；Brier/固定.5工作点改变不是新定位语义。若只有校准/长度项改善，不进入新读出。即使有时间特异信息，也不自动证明2–3pp或OPD优势；需要对后续能力机会单独裁决。

```json
{
  "CPU_seconds_export": 3.337566829053685,
  "CPU_seconds_score": 55.596554605988786,
  "GPU_seconds": 0,
  "new_models": 0,
  "new_RGB": 0,
  "new_experts": 0,
  "new_spatial_or_student_updates": 0,
  "upstream_cost_not_free": true
}
{
  "status": "pass",
  "official_candidate_quality_checks": 4608,
  "frozen_spatial_cases": 512,
  "all128_fields_before_GT": true,
  "EOS_AUROC_and_equal_length_preference_identity_all_parents": true,
  "numeric_independent": {
    "status": "pass",
    "no_GT": true,
    "raw_source_fields_exact": 24960,
    "independent_scalar_mask_sums": 24320,
    "max_error": 4.973799150320701e-13,
    "all_EOS_length_identities": true,
    "all_Flat_Shift_controls_exact": true,
    "CPU_seconds": 1.1561679940205067,
    "GPU_seconds": 0,
    "no_new_method_outputs": true
  },
  "GT_provenance": {
    "data/hcstvg2_official_metadata/val_v2.json": "2e22d9370e311ca6c420b90867f5128507f5ab145e4fc120495b7a6336e41156"
  }
}
```


修订001：辅助within_length_rank_correlation由于Flat的FP64伪tie排序无效，统一置null，不用于解释。原结果/报告按原字节存档；主AUROC/Brier/带容差pair accuracy/有益AUC及区间逐项exact。此后附加一个post-main固定错视频对照，见ADDENDUM_001.md，不追认为prelock。


## Post-main固定错父视频控制

按预定alias+1及归一化physicalclock，插值后只匹配接收者均值/std；不声称完整分布相同。128控制全部先seal后新GT，主原统计重算exact。此修订在主GT后，非原prelock或独立确认。

```json
{
  "means": {
    "classification": {
      "AUROC": {
        "mean": 0.5801136567338077,
        "defined_parents": 128
      },
      "Brier": {
        "mean": 0.27182222448483173,
        "defined_parents": 128
      },
      "logloss": {
        "mean": 0.8466472516831681,
        "defined_parents": 128
      },
      "predicted_positive_fraction": {
        "mean": 0.2811298076923077,
        "defined_parents": 128
      },
      "GT_positive_fraction": {
        "mean": 0.35877403846153855,
        "defined_parents": 128
      },
      "recall": {
        "mean": 0.3041513569169147,
        "defined_parents": 128
      },
      "precision": {
        "mean": 0.37519973264508494,
        "defined_parents": 112
      }
    },
    "ranking": {
      "equal_node_length_pair_accuracy": {
        "mean": 0.5349727290226642,
        "defined_parents": 127
      },
      "equal_length_quality_pairs": {
        "mean": 33358.765625,
        "defined_parents": 128
      },
      "within_length_rank_correlation": {
        "mean": null,
        "defined_parents": 0
      },
      "beneficial_vs_harmful_AUC": {
        "mean": 0.45177632611749163,
        "defined_parents": 105
      },
      "beneficial_candidates": {
        "mean": 338.5234375,
        "defined_parents": 128
      },
      "harmful_candidates": {
        "mean": 1392.671875,
        "defined_parents": 128
      },
      "candidate_oracle_extra_pp": {
        "mean": 22.97587183102908,
        "defined_parents": 128
      }
    }
  },
  "Action_minus_Wrong": {
    "classification": {
      "AUROC": {
        "mean": 0.2290334830556925,
        "CI95": [
          0.1625187836819784,
          0.2961966956191715
        ],
        "parents": 128
      },
      "Brier": {
        "mean": -0.08780876300605264,
        "CI95": [
          -0.1104725576457796,
          -0.06552362369480283
        ],
        "parents": 128
      },
      "logloss": {
        "mean": -0.29510494028410733,
        "CI95": [
          -0.3757539204310282,
          -0.21928802741655598
        ],
        "parents": 128
      },
      "recall": {
        "mean": 0.23013209172301233,
        "CI95": [
          0.16961379066631602,
          0.2927867360156132
        ],
        "parents": 128
      }
    },
    "ranking": {
      "equal_node_length_pair_accuracy": {
        "mean": 0.22211988257108206,
        "CI95": [
          0.15698891123188177,
          0.28954118273272755
        ],
        "parents": 127
      },
      "beneficial_vs_harmful_AUC": {
        "mean": 0.22953183208858727,
        "CI95": [
          0.17099009117495118,
          0.2885274983851179
        ],
        "parents": 105
      }
    }
  },
  "cost": {
    "CPU_seconds_export": 0.22131163906306028,
    "CPU_seconds_score": 11.885705698048696,
    "total_main_and_addendum_CPU_seconds": 71.04113919916563,
    "GPU_seconds": 0,
    "new_model_expert_or_updates": 0
  }
}
```


## 最终裁决与测量边界

Raw Action在此固定错父映射下保留对应视频信息：dense AUROC差+.229033、同节点长度pair差+.222120（127定义父）、有益/有害AUC差+.229532（105定义父）。供体循环引入共享曲线依赖，区间只作固定映射下描述，非独立确认。主128父均保留；没有新增最终时间区间或OPD收益。只获得一次匹配旧R7学生/Direct/fresh Fixed/Live的反馈转移资格，不承诺2–3pp。

整项辅助rank correlation因理论Flat的FP64伪tie无效，已null/定义父数0；原字节及修订记录保留，不能解释原rho。其余主统计和旧字段/能量exact不变。EOS未增加同长度信息且损害Brier/纠错排序，不再沿该校准配置延伸。已有structured_temporal包含Bernoulli mask代数，本诊断未发明新语义源。主导出/评分与补充合计71.041139秒CPU；独立审计另计，GPU/新专家/模型训练/空间/学生更新均0。
