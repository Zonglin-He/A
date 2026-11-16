# R116 — 原生grounding基础上的RGB GroupNorm稀疏深度TTA

本轮不增加source训练或专家，不改变原A256/六层位置/文字视角。沿用既有R91/R76图像编码器及其源域训练的I-only深度头，更新RGB三处GroupNorm的6个仿射张量、224参数；卷积、Adapter、桥、门、late、匹配头、竞争接口和3EED原网络冻结。Point Adapter与早期RGB→point固定旁路，原生点—文字h/u/r不变。

## 为什么换这个更新位置

R113真实反向检查证明，原深度项对point_to_image/gateI精确零梯度。因此本轮只取已经存在的深度恢复组件，并把它接到其实际经过的RGB参数；不是再次假定辅助loss下降就等于任务改善。GN不含运行BN统计，不将本实验冒称TENT复现或第一次尝试任何归一化。

## 监督与预算

当前A的可见前表面LiDAR经既有标定得到32×48格的log1p(depth)目标；RGB深度头只读图像，不读取点或深度feature，防止直接复制目标。固定哈希75%有效格拟合、25%物理格留出，两遮挡76001/76002保持原定义。拟合前不使用留出目标loss；所有7个参数状态固定后才读取C/融合观测。C是已有对齐LiDAR，当前RGB不变，不称新RGB视角或独立场景。主项一次梯度按原GN参数L2的1%信赖半径归一化；自然clip1/0.5提议为控制。同L2错配train深度目标与随机GN方向为负控制；raw/9尺度回溯均保留，主项不按结果切换。没有目标grounding标签进入梯度/接受。

## 总体与平台结果

| 范围 | 方法 | mIoU% | Acc25% | Acc50% |
|---|---|---:|---:|---:|
| all_six_equal | native_real_fusedABC | 16.974318 | 31.250000 | 14.843750 |
| all_six_equal | neither_correct_fusedABC | 17.265658 | 31.901042 | 14.322917 |
| all_six_equal | anchor_noLate_fusedABC | 17.120908 | 31.510417 | 14.843750 |
| all_six_equal | native_gate_only_fusedABC | 17.383951 | 32.031250 | 14.453125 |
| all_six_equal | gateI0_frozen_fusedABC | 16.953391 | 30.729167 | 14.192708 |
| all_six_equal | depth_trust_raw_fusedABC | 17.286984 | 31.901042 | 14.322917 |
| all_six_equal | depth_trust_bt_fusedABC | 17.286984 | 31.901042 | 14.322917 |
| all_six_equal | depth_natural_raw_fusedABC | 17.373861 | 32.031250 | 14.453125 |
| all_six_equal | depth_natural_bt_fusedABC | 17.373861 | 32.031250 | 14.453125 |
| all_six_equal | wrong_depth_trust_raw_fusedABC | 17.352536 | 32.031250 | 14.453125 |
| all_six_equal | wrong_depth_trust_bt_fusedABC | 17.352536 | 32.031250 | 14.453125 |
| all_six_equal | random_GN_trust_raw_fusedABC | 17.265658 | 31.901042 | 14.322917 |
| waymo | native_real_fusedABC | 11.535583 | 23.437500 | 7.421875 |
| waymo | neither_correct_fusedABC | 12.304763 | 25.390625 | 7.812500 |
| waymo | anchor_noLate_fusedABC | 11.484142 | 23.437500 | 7.421875 |
| waymo | native_gate_only_fusedABC | 12.385622 | 25.390625 | 7.812500 |
| waymo | gateI0_frozen_fusedABC | 11.893641 | 24.218750 | 7.421875 |
| waymo | depth_trust_raw_fusedABC | 12.304763 | 25.390625 | 7.812500 |
| waymo | depth_trust_bt_fusedABC | 12.304763 | 25.390625 | 7.812500 |
| waymo | depth_natural_raw_fusedABC | 12.304763 | 25.390625 | 7.812500 |
| waymo | depth_natural_bt_fusedABC | 12.304763 | 25.390625 | 7.812500 |
| waymo | wrong_depth_trust_raw_fusedABC | 12.304763 | 25.390625 | 7.812500 |
| waymo | wrong_depth_trust_bt_fusedABC | 12.304763 | 25.390625 | 7.812500 |
| waymo | random_GN_trust_raw_fusedABC | 12.304763 | 25.390625 | 7.812500 |
| drone | native_real_fusedABC | 14.710633 | 26.171875 | 12.890625 |
| drone | neither_correct_fusedABC | 14.503438 | 26.171875 | 11.718750 |
| drone | anchor_noLate_fusedABC | 14.996071 | 26.953125 | 12.890625 |
| drone | native_gate_only_fusedABC | 14.777456 | 26.562500 | 12.109375 |
| drone | gateI0_frozen_fusedABC | 14.018964 | 24.218750 | 11.718750 |
| drone | depth_trust_raw_fusedABC | 14.567414 | 26.171875 | 11.718750 |
| drone | depth_trust_bt_fusedABC | 14.567414 | 26.171875 | 11.718750 |
| drone | depth_natural_raw_fusedABC | 14.828045 | 26.562500 | 12.109375 |
| drone | depth_natural_bt_fusedABC | 14.828045 | 26.562500 | 12.109375 |
| drone | wrong_depth_trust_raw_fusedABC | 14.764070 | 26.562500 | 12.109375 |
| drone | wrong_depth_trust_bt_fusedABC | 14.764070 | 26.562500 | 12.109375 |
| drone | random_GN_trust_raw_fusedABC | 14.503438 | 26.171875 | 11.718750 |
| quad | native_real_fusedABC | 24.676737 | 44.140625 | 24.218750 |
| quad | neither_correct_fusedABC | 24.988774 | 44.140625 | 23.437500 |
| quad | anchor_noLate_fusedABC | 24.882511 | 44.140625 | 24.218750 |
| quad | native_gate_only_fusedABC | 24.988774 | 44.140625 | 23.437500 |
| quad | gateI0_frozen_fusedABC | 24.947567 | 43.750000 | 23.437500 |
| quad | depth_trust_raw_fusedABC | 24.988774 | 44.140625 | 23.437500 |
| quad | depth_trust_bt_fusedABC | 24.988774 | 44.140625 | 23.437500 |
| quad | depth_natural_raw_fusedABC | 24.988774 | 44.140625 | 23.437500 |
| quad | depth_natural_bt_fusedABC | 24.988774 | 44.140625 | 23.437500 |
| quad | wrong_depth_trust_raw_fusedABC | 24.988774 | 44.140625 | 23.437500 |
| quad | wrong_depth_trust_bt_fusedABC | 24.988774 | 44.140625 | 23.437500 |
| quad | random_GN_trust_raw_fusedABC | 24.988774 | 44.140625 | 23.437500 |

主项17.286984%，较同信息零步17.265658高0.021325pp，但低于错配深度17.352536和旧单门17.383951，不通过锁定门槛。自然一步17.373861的开发正结果完整保留，不改为主项、不追加步长网格。它也尚未超过已有单门对照。停止当前uniform-depth/GN配方，不据此否定全部物理监督或归一化适应。

## 恢复和任务分别评价

```json
[
  {
    "state": "depth_trust_raw",
    "n": 724,
    "frozen_holdout_depth_mean": 0.23363303444065278,
    "updated_holdout_depth_mean": 0.2027495051090971,
    "improved": 705,
    "worsened": 19,
    "same": 0
  },
  {
    "state": "depth_natural_raw",
    "n": 724,
    "frozen_holdout_depth_mean": 0.23363303444065278,
    "updated_holdout_depth_mean": 0.17516609914703757,
    "improved": 717,
    "worsened": 7,
    "same": 0
  },
  {
    "state": "depth_trust_bt",
    "n": 724,
    "frozen_holdout_depth_mean": 0.23363303444065278,
    "updated_holdout_depth_mean": 0.20274953234899604,
    "improved": 706,
    "worsened": 18,
    "same": 0
  },
  {
    "state": "wrong_depth_trust_raw",
    "n": 724,
    "frozen_holdout_depth_mean": 0.23363303444065278,
    "updated_holdout_depth_mean": 0.203237635376786,
    "improved": 697,
    "worsened": 27,
    "same": 0
  },
  {
    "state": "random_GN_trust_raw",
    "n": 724,
    "frozen_holdout_depth_mean": 0.23363303444065278,
    "updated_holdout_depth_mean": 0.23353584646890507,
    "improved": 368,
    "worsened": 356,
    "same": 0
  }
]
```

```json
[
  {
    "control": "neither_correct_fusedABC",
    "n": 768,
    "gain_pp": 0.02132527628593644,
    "better": 1,
    "worse": 0,
    "same": 767,
    "largest_positive_single_input_pp": 0.021325276285936442,
    "largest_negative_single_input_pp": 0.0
  },
  {
    "control": "native_gate_only_fusedABC",
    "n": 768,
    "gain_pp": -0.09696744114076365,
    "better": 3,
    "worse": 5,
    "same": 760,
    "largest_positive_single_input_pp": 0.03478109382039299,
    "largest_negative_single_input_pp": -0.08687719749109822
  },
  {
    "control": "wrong_depth_trust_raw_fusedABC",
    "n": 768,
    "gain_pp": -0.06555192120516179,
    "better": 1,
    "worse": 1,
    "same": 766,
    "largest_positive_single_input_pp": 0.021325276285936442,
    "largest_negative_single_input_pp": -0.08687719749109822
  },
  {
    "control": "random_GN_trust_raw_fusedABC",
    "n": 768,
    "gain_pp": 0.02132527628593644,
    "better": 1,
    "worse": 0,
    "same": 767,
    "largest_positive_single_input_pp": 0.021325276285936442,
    "largest_negative_single_input_pp": 0.0
  }
]
```

恢复误差变化与任务选框收益分开：224参数的更新接口实际有效，不能再归因为深度梯度未到达该scope；但这份恢复监督的任务增量弱。正负IoU尾部均保留，没有新看原图，实例身份UNKNOWN；不把微小同物体框切换自动叫纠错。

## 额外的loss信息检查

后验仅CPU前向/反向，未做参数更新或新任务评分：724条中平均15.47%残差处于Huber截断区，正确/错配Huber梯度余弦0.9584；改成MSE后的正确梯度与原Huber余弦0.9958，正确/错配MSE余弦0.9541。因此“多数梯度被Huber截断抹平”的解释不成立，没有继续开MSE loss一轮的依据。保留该阴性分析，不为延长实验换loss。

## 工程与独立审计

12真实输入、7状态/输入的完整原生模型与基础缓存重放位等，深度→GN梯度非零；全768含44保护媒体别名冻结回退。31读出共23808任务值，21新16128、10旧7680复用；15204新CPU读出端点、5068实际参数状态核验。CPU还复现了更新后训练深度误差和未拟合物理格误差（最大约1.79e-7），实际写回范数误差5.55e-17、提议误差7.46e-9；概率误差<5e-6、近平0。全部预测先封存，后评分/独立几何和聚合；FINAL未访问。

## 下一步

不继续扫描GN层或回归loss。R112静态架构撤回相对原prepared的较大DEV增益尚需扩展验证；下一项固定配置检查原生基础与RGB/竞争接口各自贡献。R115均值/逆向门仅作冻结校准对照，不在扩展数据重新做TTA或据GT选择参数。
