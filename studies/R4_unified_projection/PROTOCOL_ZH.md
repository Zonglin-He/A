# R4 · 7-DoF Observation Projection 协议

正式顶缩为 A，固定 yaw 三轴、水平中心加尺寸、中心加 yaw、7D Sobol、7D 局部模式改进、可见子空间搜索及同池无先验均共享同一 Seal。对最多 192 条 Selection 射线的预测首交用正负差分构造归一化 Jacobian，只计活动面恒定的观测。按奇异值秩分出弱方向，先验只读取现有 Qwen 尺寸和轴向角。区分相对于原框的严格 E，与正支持和实际测量定义的 F。独立 Audit 单独测更改成本。对 E 中去重的有限候选另验证全部 Selection，再追加全部 Audit。GT 兼容性和候选 Oracle 仅离线报告。 [面板锁](../3eed_five_branches_20261128/PROTOCOL_ZH.md)
