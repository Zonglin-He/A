# 保留原分布的事件重加权：冻结lambda0.5的HC109确认

日期：2026-10-19。HC128选择后冻结lambda0.5；全部109个父与本轮lambda开发集不交叉。历史Source/旧方法及RH/RF/NH曝光逐项公开；不是完全未使用过的方法测试集。每父一个预锁query，三seed41/42/43先父内平均，109父等权。确认数据不改配置。

## 主结果：空间完全固定

| 面板 | 原MBR完整vIoU% | 重加权完整vIoU% | 增量pp及95%配对区间 |
|---|---:|---:|---|
| HC128开发 | 27.928474 | 28.413946 | +0.485472 [-0.294352, +1.463846] |
| HC109确认 | 28.896783 | 29.054291 | +0.157508 [-0.199236, +0.514893] |

| HC109固定空间 | lambda | vIoU% | tIoU% | 开始误差frame | 结束误差frame | 平均边界误差frame |
|---|---:|---:|---:|---:|---:|---:|
| Live_seedmean | 0 | 28.896783 | 51.340449 | 73.577982 | 80.788991 | 77.183486 |
| Live_seedmean | 0.5 | 29.054291 | 51.400929 | 74.437309 | 80.562691 | 77.500000 |
| E | 0 | 28.645258 | 51.340449 | 73.577982 | 80.788991 | 77.183486 |
| E | 0.5 | 28.767738 | 51.317004 | 74.541284 | 80.651376 | 77.596330 |

tIoU增量及区间：{"mean": 0.0604796469705195, "CI95": [-0.5280395027716983, 0.6518927116651719], "parents": 109}

全部正负/零变化和严重退化：
```json
{
  "positive": 32,
  "negative": 38,
  "unchanged": 39,
  "below_minus3": 3,
  "below_minus5": 2,
  "positive_mean_contribution_pp": 0.6651455284393448,
  "negative_mean_contribution_pp": -0.5076379307065251,
  "worst": [
    {
      "alias": "TC078",
      "delta_pp": -5.911301156328241
    },
    {
      "alias": "TC033",
      "delta_pp": -5.127550825185234
    },
    {
      "alias": "TC003",
      "delta_pp": -3.3099495062719644
    },
    {
      "alias": "TC057",
      "delta_pp": -2.915258180910396
    },
    {
      "alias": "TC087",
      "delta_pp": -2.3450719311315233
    },
    {
      "alias": "TC108",
      "delta_pp": -2.268250870171035
    },
    {
      "alias": "TC083",
      "delta_pp": -2.029011069312431
    },
    {
      "alias": "TC096",
      "delta_pp": -1.9255132784323559
    }
  ]
}
```

## 事后错误分解

GT只在全部109预测全局封存后使用，分组来自原区间；不进入在线选择、门控、参数或回退。每行组内父等权。原时间损失=同空间GT时间−原时间；容量=原MBR全2080决策支持GT最优−原时间；不能当作教师可兑现成绩。

| 原错误 | 父数 | 原时间损失pp | 原MBR支持容量pp | 冻结lambda实际恢复pp |
|---|---:|---:|---:|---:|
| wrong_segment | 9 | +57.538876 | +56.122511 | +0.989794 |
| missing_event_part | 8 | +9.668588 | +9.194385 | -0.005034 |
| excess_outside_time | 26 | +24.149692 | +22.486197 | -0.390891 |
| other_boundary_or_exact | 66 | +24.673067 | +23.511854 | +0.279752 |

## 执行、成本与核验

用户接受109而非原要求至少128，允许确认最多128次八帧ReferDINO。本轮实际109次视频前向/872图，同一原八位置。每次同时保存原生选中轨迹和token-max前语义，32条旧原生选中输出bitexact；109条max-token与原生logits精确。没有额外DINO、条件teacher查询或新模型训练。

固定空间算法tau0.2/尺度0.05,0.25/40步/lr0.01/32候选；32父×3seed精确复用旧RF32，77父×3seed只生成已冻结空间基线，9240次临时state更新。时间对照共享完全相同空间数组，不将这些更新当作新temporal OPD。新增Source分布/特征导出、ReferDINO和空间基线成本分别计数：
```json
{
  "status": "pass",
  "checks": {
    "prior_CDF": 109,
    "feedback_integrals": 906880,
    "posterior": 872,
    "lambda0_exact": 436,
    "space_exact": 436,
    "token_max": 109,
    "legacy_selected_exact": 32,
    "old_inputs_unchanged": 109
  },
  "max_prior_CDF_error": 3.0270924655795284e-16,
  "max_reweight_error": 1.7069679003611782e-15,
  "old_spatial_reused": 32,
  "new_spatial_baselines": 77,
  "actual_temporary_state_updates": 9240,
  "new_ReferDINO_video_forwards": 109,
  "new_ReferDINO_image_frames": 872,
  "new_DINO": 0,
  "network_weight_updates": 0,
  "temporal_OPD_updates": 0,
  "GPU_lease_seconds": 896.814724690048,
  "selected_lambda_unchanged": 0.5,
  "all109_sealed_before_new_GT": true,
  "all_parent_disjoint_from_development": true
}
```

完整stage成本：
```json
[
  {
    "GPU_seconds": 352.47909580217674,
    "new_Source": 107,
    "Source_offset_forwards": 214,
    "new_ReferDINO": 0,
    "new_DINO": 0
  },
  {
    "GPU_seconds": 368.59968266496435,
    "new_spatial_queries": 77,
    "actual_temporary_state_updates": 9240,
    "new_ReferDINO": 0,
    "new_Source": 0,
    "new_DINO": 0,
    "network_weight_updates": 0
  },
  {
    "GPU_seconds": 175.73594622290693,
    "new_ReferDINO": 109,
    "new_ReferDINO_image_frames": 872,
    "new_Source": 0,
    "new_DINO": 0
  }
]
```

p0、候选支持、finalMBR及反馈定义与开发一致。65物理端点/2080strict-index决策包括零mass决策；零质量不复活，lambda0精确原矩阵与原MBR。R为同旧八帧事件词组/全局单轨关联/duration内外contrast，视频内全legal定义支持标准化，fullspan无outside增量0，平坦缺失不变。缓存八点不是独立逐帧观测，token分数不是校准事件概率。

## 决策

本次仅检验原分布与既定事件反馈的互补性，不是temporal OPD。确认增量应连同区间、尾部和集中度判断；不得因开发选中或单个正均值宣称2–3pp已建立。 冻结的是比较，不因结果更改空间配置或选择新lambda。研究/生产登记保持不变。

全量逐父×seed signed/unknown数值在RESULTS_PUBLIC.json，曝光在EXPOSURE_PUBLIC.json；无删样本、按错误类型挑参数或事后换配置。新增接口/下一轮不自动启动。

## 最终裁决补充（核验后）

确认完整v增量+0.157508pp，95%区间[-0.199236,+0.514893]；未建立稳定2–3pp增益，当前固定反馈不进入temporal OPD，不再追加lambda/温度。独立从每seed逐父数值重算父内平均及10000次bootstrap与结果exact。301/327个Live/E关联同一teacher track，语义曲线与重加权区间exact相同，不能称空间状态改变后教师语义重读。所有109词组状态：{'valid': 106, 'no_event_predicate': 3}。权重与后处理源码pins不变。
