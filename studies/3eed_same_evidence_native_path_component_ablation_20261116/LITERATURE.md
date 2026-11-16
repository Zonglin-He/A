# 机制与实现依据

本轮是既有系统的受控撤回实验，不增加新算法或引用一个分类TTA配方作为有效性保证。

- 3EED官方前向同时维护原生点云—语言cross-encoder、逐层decoder和query/text匹配投影。单次A的query演化可以用六层原位置跟踪；不同采样不能直接假定query编号保持同一实例。因此本轮保留source A全部位置/256框，只改变送到相同A1024种子位置的PointNet特征。
  - https://github.com/worldbench/3EED/blob/main/models/bdetr.py
- 本地既有 `research/round66/interface66.py::replay` 从融合前点特征重新经过原生语言交互与query初始化；它不把旧的最终query当作新输入。R106 `interpolation106/prefix106` 提供固定的几何支持及未知回退，本轮不修改其阈值或样本选择。
- R76/R91是已完成源域准备的轻量RGB↔LiDAR与实例竞争模块。本轮使用其原冻结状态作为对照，不重新训练，也不把撤回该模块的诊断当成三模态TTA创新。

改变匹配投影而固定对应h/u，只能检测该投影在当前输入系统上的作用。跨系统的表示基、上下文文字都可能改变，所以不能由一个互换成绩单独宣布哪组参数“错误”。最终位置指标、同信息无更新均值、真实/随机/错配对照与全部回退共同约束结论。
