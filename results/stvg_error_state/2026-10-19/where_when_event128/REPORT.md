# ReferDINO对象条件事件反馈：HC128能力检查

日期：2026-10-19。128个已暴露开发父视频，冻结空间配置、原八帧、原生ReferDINO；没有运行时间OPD、空间重适应或参数搜索。

## 固定E空间坐标的时间读出

| 读出 | 完整vIoU % | tIoU % |
|---|---:|---:|
| E_MBR | 27.2103 | 51.4395 |
| E_NativeConfidence | 19.8855 | 35.5243 |
| E_EventNative | 20.7499 | 38.1601 |
| E_EventE | 21.0454 | 39.0239 |
| E_EventLive_seedmean | 20.8768 | 38.7681 |

全部方法使用同一E空间坐标，因此actual-node sIoU按构造相同；tIoU变化与vIoU变化分别报告。三seed先父内平均指标，不平均框或挑seed。

| 比较，pp [父视频配对95%区间] | 完整vIoU | tIoU |
|---|---:|---:|
| E_EventE_minus_E_NativeConfidence | +1.1599 [-0.6480, +3.1367] | +3.4996 [-0.8816, +8.0454] |
| E_EventE_minus_E_EventNative | +0.2955 [-0.4249, +1.1223] | +0.8638 [-1.1135, +3.0415] |
| E_EventLive_seedmean_minus_E_EventE | -0.1686 [-0.7926, +0.3449] | -0.2558 [-1.4893, +0.9382] |
| E_EventLive_seedmean_minus_E_MBR | -6.3335 [-9.3230, -3.2880] | -12.6715 [-17.8445, -7.4359] |
| E_EventE_minus_E_MBR | -6.1649 [-9.1260, -3.2075] | -12.4156 [-17.5864, -7.1719] |
| Live_NewTime_seedmean_minus_Live_OriginalTime_seedmean | -6.4118 [-9.4214, -3.3346] | -12.6715 [-17.8445, -7.4359] |
| Live_NewTime_seedmean_minus_E_MBR | -5.6936 [-8.7915, -2.5532] | -12.6715 [-17.8445, -7.4359] |

## 稀疏语义与候选排序

| 响应 | 事件内外AUC（父均值） | 区间tIoU成对排序准确率（父均值） |
|---|---:|---:|
| NativeConfidence | 66.538% [61.726, 71.241], n=127 | 59.609% [56.252, 62.938], n=128 |
| EventNative | 70.482% [66.183, 74.753], n=127 | 61.798% [58.599, 64.957], n=128 |
| EventE | 70.306% [65.998, 74.617], n=127 | 61.550% [58.349, 64.738], n=128 |
| EventLive_seedmean | 70.460% [66.195, 74.729], n=127 | 61.624% [58.469, 64.821], n=128 |

有效词组数、时间覆盖、对象切换和仅离线候选容量：

```json
{
  "event_valid": 128,
  "inside_and_outside_actual_evidence": 127,
  "event_inactive_verified_visibility": "not established: HC boxes are event annotations, no independent outside-event visibility truth",
  "query_switched_from_E": {
    "41": 8,
    "42": 9,
    "43": 9
  },
  "oracle_capacity_diagnostic": {
    "t_gain_pp": {
      "mean": 30.862833836147885,
      "CI95": [
        26.785457807696144,
        34.91593430933
      ],
      "parents": 128
    },
    "v_gain_pp": {
      "mean": 14.668161186010058,
      "CI95": [
        12.018461813711042,
        17.443229573831253
      ],
      "parents": 128
    }
  }
}
```

## 负例与成本

完整主评价保留全部128。以下是最终联合输出相对E的严重尾部；不是筛样或回退依据。

```json
{
  "positive": 42,
  "negative": 81,
  "unchanged": 5,
  "below_minus5": 55,
  "worst": [
    {
      "alias": "RC111",
      "delta_pp": -57.03582717078164
    },
    {
      "alias": "RC076",
      "delta_pp": -48.32338279053241
    },
    {
      "alias": "RC051",
      "delta_pp": -46.90268160696012
    },
    {
      "alias": "RC112",
      "delta_pp": -46.61860998542353
    },
    {
      "alias": "RC128",
      "delta_pp": -43.87925730599501
    },
    {
      "alias": "RC080",
      "delta_pp": -42.9319676247462
    },
    {
      "alias": "RC016",
      "delta_pp": -42.73972962391793
    },
    {
      "alias": "RC115",
      "delta_pp": -42.431968710944425
    }
  ]
}
```

```json
{
  "GPU_seconds": 209.95029943506233,
  "new_teacher_video_queries": 128,
  "new_teacher_image_frames": 1024,
  "new_spatial_updates": 0,
  "new_Source": 0,
  "new_DINO_T": 0
}
```

成本为新增教师租约，包含一次模型加载及持锁期间解码/写出；没有新Source/DINO-T或空间更新。既有E/空间OPD缓存的历史成本没有计作免费总推理。逐query解码/前向成本另见COST_BREAKDOWN.json。

## 结论边界

这项实验只检验固定的词组/全轨迹关联/稀疏时间contrast接口。事件分数不是校准事件概率；全上下文融合响应不是独立帧观测。描述性分词和交互对象名词也可能进入固定词组。

同一教师track的语义响应不依赖学生框；空间作用若存在，首先归于选择不同track，不能称同一对象的语义被重读。一个track index也不保证物理身份从未漂移。

时间曲线只来自八个实际位置；Voronoi单元仅定义积分。fullspan没有outside，contrast未知，不能借人为0分宣称能排序全视频。无事件/恒定响应按协议保留MBR。

HC事件框不提供独立事件外可见性真值。本轮不能完成“目标可见但事件不成立”的可靠负例验证；不拿teacher高分或缺GT框循环定义这类标签。

候选oracle在全预测封存后离线计算，仅衡量有限支持容量，不是可部署收益。直接读出失败也不等于所有OPD失败；候选排序和任务结果分别判断。没有运行新时间策略采样/蒸馏，不把本轮成绩归为OPD增量。

现空间版本继续封存，CURRENT/MAIN/PAPER不改；新一轮模型、扫参、扩大观察或OPD均不自动启动。所有query/seed signed结果与unknown见RESULTS_PUBLIC.json；数值与源缓存核验见INDEPENDENT_AUDIT.json、ROOT_AUDIT.json。

## 本轮裁决

事件读出包含可测时间信息，但本轮没有建立WHERE→WHEN增益：E关联相对原生confidence的事件内外AUC与候选排序均值提高，但这两个配对差值区间均跨零；Live关联相对E关联两项变化的配对区间仍跨零。具体差值见RESULTS_PUBLIC.json/event_paired。当前联合输出显著低于原时间方案，本配置不接入正式推理，不进入时间OPD或追加参数搜索。

直接区间读出的失败不等于OPD失败，也不能推成token反馈没有事件信息。本轮仅有同一原生语义缓存上的对象关联变化；多数父视频关联对象不变，语义响应按构造相同，未建立“空间变化使同一对象语义重读更好”的机制。

有限候选的GT容量仍较大，但只是封存后的离线诊断，不证明教师已能兑现2–3pp。实际时长统计为原MBR占视频42.08%，EventE为58.41%，Live事件读出约57.36%；当前读出平均更长，因此不能用“只追逐短峰”解释退化。这只确定了时长分布变化，没有定位退化的唯一原因。所有时长统计与轨迹切换见POSTSEAL_READOUT_DIAGNOSIS.json。

冻结空间与原时间对照保持；保留事件反馈作为尚未兑现的候选证据，不自动恢复旧队列、不引入新模块或扩大预算。
