# R115 — 输入特定单门适应与固定/打乱门校准

本轮只补上R114单门正信号的关键控制，不增加loss、source训练、专家或更新层。历史暴露DEV768/384表达、六方向、44保护别名回退，原A256/坐标/decoder位置不变；FINAL未访问。既有动态单门输出全部复用，新增6读出共4608任务值，4344非别名CPU端点与全体双几何/独立聚合审计完成。

## 机制和对照

R114证明仅gateI可复现45k桥更新的全部Top1，但门归零有害。此处比较实际A-only梯度给出的动态标量更新，source内全部开发观测的平均更新量，以及固定哈希非自身置换的同一组更新量。均值和置换先读取既有当前A梯度，在任何C隐藏状态读取前固定；没有任务标签筛选/选强度。均值与打乱需要其他输入的梯度，属于转导开发诊断，不冒称同一在线信息预算。部署参数没有利用GT，开发架构/主张选择已有历史标签曝光。

## 总体与平台

| 范围 | 方法 | mIoU% | Acc25% | Acc50% |
|---|---|---:|---:|---:|
| all_six_equal | native_real_fusedABC | 16.974318 | 31.250000 | 14.843750 |
| all_six_equal | neither_correct_fusedABC | 17.265658 | 31.901042 | 14.322917 |
| all_six_equal | anchor_noLate_fusedABC | 17.120908 | 31.510417 | 14.843750 |
| all_six_equal | native_gate_only_fusedABC | 17.383951 | 32.031250 | 14.453125 |
| all_six_equal | native_bridge_no_gate_fusedABC | 17.265658 | 31.901042 | 14.322917 |
| all_six_equal | sourcekey_frozen_fusedABC | 17.271698 | 31.901042 | 14.322917 |
| all_six_equal | gateI0_frozen_fusedABC | 16.953391 | 30.729167 | 14.192708 |
| all_six_equal | fixed_source_mean_gate_fusedABC | 17.384108 | 32.031250 | 14.453125 |
| all_six_equal | shuffled_gate_fusedABC | 17.394825 | 32.161458 | 14.453125 |
| waymo | native_real_fusedABC | 11.535583 | 23.437500 | 7.421875 |
| waymo | neither_correct_fusedABC | 12.304763 | 25.390625 | 7.812500 |
| waymo | anchor_noLate_fusedABC | 11.484142 | 23.437500 | 7.421875 |
| waymo | native_gate_only_fusedABC | 12.385622 | 25.390625 | 7.812500 |
| waymo | native_bridge_no_gate_fusedABC | 12.304763 | 25.390625 | 7.812500 |
| waymo | sourcekey_frozen_fusedABC | 12.288986 | 25.390625 | 7.812500 |
| waymo | gateI0_frozen_fusedABC | 11.893641 | 24.218750 | 7.421875 |
| waymo | fixed_source_mean_gate_fusedABC | 12.385622 | 25.390625 | 7.812500 |
| waymo | shuffled_gate_fusedABC | 12.305557 | 25.390625 | 7.812500 |
| drone | native_real_fusedABC | 14.710633 | 26.171875 | 12.890625 |
| drone | neither_correct_fusedABC | 14.503438 | 26.171875 | 11.718750 |
| drone | anchor_noLate_fusedABC | 14.996071 | 26.953125 | 12.890625 |
| drone | native_gate_only_fusedABC | 14.777456 | 26.562500 | 12.109375 |
| drone | native_bridge_no_gate_fusedABC | 14.503438 | 26.171875 | 11.718750 |
| drone | sourcekey_frozen_fusedABC | 14.537333 | 26.171875 | 11.718750 |
| drone | gateI0_frozen_fusedABC | 14.018964 | 24.218750 | 11.718750 |
| drone | fixed_source_mean_gate_fusedABC | 14.777926 | 26.562500 | 12.109375 |
| drone | shuffled_gate_fusedABC | 14.890145 | 26.953125 | 12.109375 |
| quad | native_real_fusedABC | 24.676737 | 44.140625 | 24.218750 |
| quad | neither_correct_fusedABC | 24.988774 | 44.140625 | 23.437500 |
| quad | anchor_noLate_fusedABC | 24.882511 | 44.140625 | 24.218750 |
| quad | native_gate_only_fusedABC | 24.988774 | 44.140625 | 23.437500 |
| quad | native_bridge_no_gate_fusedABC | 24.988774 | 44.140625 | 23.437500 |
| quad | sourcekey_frozen_fusedABC | 24.988774 | 44.140625 | 23.437500 |
| quad | gateI0_frozen_fusedABC | 24.947567 | 43.750000 | 23.437500 |
| quad | fixed_source_mean_gate_fusedABC | 24.988774 | 44.140625 | 23.437500 |
| quad | shuffled_gate_fusedABC | 24.988774 | 44.140625 | 23.437500 |

动态17.383951%，固定source平均17.384108%，打乱17.394825%，均高于零步17.265658%。动态没有优于固定或打乱，不支持“当前输入的恢复误差提供有用的特定门调整”。保留小幅静态/均值校准信号，但不升级为已成立的输入适应或跨模态身份修复。按预锁定规则结束该逐输入gate配方，不进行扩展TTA或强度网格；当前结果也不能排除不同协议下的域级校准，但本轮没有验证它。

## 全体差异与负例

```json
[
  {
    "control": "neither_correct_fusedABC",
    "n": 768,
    "gain_pp": 0.11829271742670008,
    "better": 5,
    "worse": 2,
    "same": 761,
    "largest_positive_single_input_pp": 0.08687719749109822,
    "largest_negative_single_input_pp": -0.03478109382039299
  },
  {
    "control": "fixed_source_mean_gate_fusedABC",
    "n": 768,
    "gain_pp": -0.00015668272094423256,
    "better": 0,
    "worse": 2,
    "same": 766,
    "largest_positive_single_input_pp": 0.0,
    "largest_negative_single_input_pp": -0.0001566449270827498
  },
  {
    "control": "shuffled_gate_fusedABC",
    "n": 768,
    "gain_pp": -0.010874423440540675,
    "better": 1,
    "worse": 4,
    "same": 763,
    "largest_positive_single_input_pp": 0.02668853354421027,
    "largest_negative_single_input_pp": -0.03478109382039299
  }
]
```

```json
[
  {
    "control": "neither_correct_fusedABC",
    "same_query_as_primary": 754,
    "n": 768,
    "identity_not_certified": true
  },
  {
    "control": "fixed_source_mean_gate_fusedABC",
    "same_query_as_primary": 761,
    "n": 768,
    "identity_not_certified": true
  },
  {
    "control": "shuffled_gate_fusedABC",
    "same_query_as_primary": 758,
    "n": 768,
    "identity_not_certified": true
  }
]
```

原主项与全部负例保留；没有看新图，query一致或IoU改变量均不认证真实物体身份。没有按平台/个案选择更新量。

## 工程与审计

12真实输入验证动态概率逐值复现R114、两个新控制实际完整模型和冻结基础缓存重放位等、源状态恢复。第一版复制审计的24臂门槛未替换成实际6臂而失败，在任何任务GT前修复为实际契约并将参考预测authority改为R114，隐藏基础仍来自R113；预测/封存完全未改。失败audit代码与日志留存。全部新CPU概率误差<5e-6，无近平赢家歧义。

下一项只考虑已有深度监督确实经过的RGB参数，依据R113深度对桥零梯度事实，不重新解释门控制阴性为整个TTA方向失败。
