# 对象条件时间memory：HC128能力筛查

日期2026-10-19，128已曝光开发父视频，每父一query。原冻结空间Live41/42/43，父内seed均值；E辅助。不是时间OPD，没有新专家、网络训练或空间更新。

旧Object_temporal只重建共享itq。本轮保持itq、第一pass路由、时间self-attention、全部原memory values与时间头，第二pass每层cross-attention在运动token上加逐帧对象几何log prior。文本prior1，运动prior=.5+.5×fractional覆盖密度，空交集noop，原padding保留。Refer轨迹只作为时间条件，未替换最终空间坐标。

| 时间条件 | 完整vIoU% | tIoU% | 对原生完整v增量pp [95%CI] |
|---|---:|---:|---|
| Native | 27.928474 | 51.439514 | 0 |
| E_ROI | 27.439828 | 50.598455 | -0.488646 [-1.218623,+0.267417] |
| LiveOwnROI | 27.517324 | 50.793545 | -0.411149 [-1.135480,+0.332682] |
| ReferROI | 27.697358 | 50.291758 | -0.231116 [-1.453188,+1.231326] |
| ShiftedReferROI | 27.484384 | 50.239925 | -0.444090 [-1.151184,+0.218543] |

主候选预定ReferROI，不从E/Live/Shifted事后选择赢家。正确ReferROI−ShiftedReferROI完整v：{"mean": 0.2129742514899315, "CI95": [-1.2736880578106076, 1.8937648514189964], "parents": 128}

主候选全部尾部：
```json
{
  "positive": 42,
  "negative": 51,
  "flat": 35,
  "below_minus3": 24,
  "below_minus5": 15,
  "positive_contribution_pp": 1.4815626712581074,
  "negative_contribution_pp": -1.7126785834979092,
  "worst": [
    {
      "alias": "RC028",
      "delta_pp": -18.40958879802162
    },
    {
      "alias": "RC122",
      "delta_pp": -17.06586450053797
    },
    {
      "alias": "RC107",
      "delta_pp": -16.560447706261158
    },
    {
      "alias": "RC065",
      "delta_pp": -11.766447407067492
    },
    {
      "alias": "RC096",
      "delta_pp": -10.953237462919374
    },
    {
      "alias": "RC061",
      "delta_pp": -10.354209241178495
    },
    {
      "alias": "RC015",
      "delta_pp": -10.106640576852644
    },
    {
      "alias": "RC067",
      "delta_pp": -8.721856700523102
    }
  ]
}
```

## 工程与成本

首条GT前身份核验失败：缓存克隆将原跨帧expand共享query实体化，值相同但logits最大差1.430511e-6；原native stride恢复后exact。原失败/源码/旧plan/debug保存，未通过放宽容差或改变模型解决；模型/目标/名单/几何公式不变，失败及排查GPU计入总成本。准备阶段还纠正一次加载外推重复乘128的工程公式，均在任何GT前。
```json
{
  "GPU_seconds": 690.9891937382054,
  "new_Source_queries": 128,
  "Source_offset_forwards": 256,
  "new_DINO": 0,
  "new_ReferDINO": 0,
  "temporal_conditioned_offset_decodes": 1536,
  "network_weight_updates": 0,
  "spatial_updates": 0,
  "attempts": [
    {
      "GPU_seconds": 14.706519389059395,
      "new_Source_queries": 1,
      "Source_offset_forwards": 2,
      "new_DINO": 0,
      "new_ReferDINO": 0,
      "temporal_conditioned_offset_decodes": 0,
      "network_weight_updates": 0,
      "spatial_updates": 0
    },
    {
      "GPU_seconds": 7.087161415023729,
      "new_Source_queries": 0,
      "Source_offset_forwards": 0,
      "new_DINO": 0,
      "new_ReferDINO": 0,
      "network_weight_updates": 0,
      "role": "native-stride identity engineering diagnosis"
    },
    {
      "GPU_seconds": 669.1955129341222,
      "new_Source_queries": 127,
      "Source_offset_forwards": 254,
      "new_DINO": 0,
      "new_ReferDINO": 0,
      "temporal_conditioned_offset_decodes": 1536,
      "network_weight_updates": 0,
      "spatial_updates": 0
    }
  ]
}
```

## 核验与裁决

{
  "status": "pass",
  "spatial_arrays_exact": 512,
  "prior_mass_checks": 1536,
  "brute_MBR_checks": 16128,
  "max_independent_CDF_error": 2.8892802050612537e-16,
  "Source_and_original_temporal_logits_exact": true,
  "all_source_spatial_teacher_prior_inputs_unchanged": true,
  "network_weights_frozen": true,
  "new_spatial_updates": 0,
  "engineering_recovery": "native expanded-query zero stride restored; original failure/debug preserved, cost included",
  "primary_decision": "no_meaningful_primary_increment_in_this_configuration"
}

主候选决策：no_meaningful_primary_increment_in_this_configuration。本轮开发，不宣称独立确认或OPD增量；不改变冻结空间/研究/生产登记。若没有足够主候选收益，关闭此固定prior配置，不追加强度/层/温度扫描；自主目标可转到实质不同机制，旧负结果保留。

全部128父、全部七原始时间arm×四固定空间、各seed和signed/unknown标量在RESULTS_PUBLIC.json；GT只在全输出seal后评价。源特征和原生八点专家缓存重用与新Source前向成本分别披露，专家缓存不等于原部署专家免费。
