# R8 全局信任尺度的最终登记方向

root 在真实 R8 执行前决定使用目标空间收缩，而不是把全局信任乘进 normalized loss 或再次乘学习率：

`pseudo_center = anchor_q1 + r_global × (raw_consensus_center − anchor_q1)`

仅对通过几何位移界限的轴应用；`r_global=semantic/PC-margin × PCsupport × multirepresentative`，各因子在[0,1]。Direct G、参数更新和等价输出步接收同一个最终伪框。`w_center` 仅决定有效轴及相对 dispersion；尺寸保持 q1 source prior，尺寸权重0。学习率仍 `0.25×0.1/max(q1_center_feature_norm²+1,q1_size_feature_norm²+1)`，不再次乘 global trust。

这修复了“统一 scalar 在 sum(wloss)/sum(w) 中抵消”的解释问题；并不证明信任度能估计正确性。独立无标签审计须分别记录 q0→q1 关联变化、raw consensus 非零、最终 float64 target 非零、实际 float32 target 非零、真实执行步骤和非零参数变化，不能只凭 q1 改变或 w>0 声称有效更新。必须保存 raw consensus、三个信任因子/r_global、未量化与实际伪框和q1 anchor。

这是对最初协议草案中“若 driver 选择额外 LR 门控”的最终替代；原草案作为实现前问题记录保留，不作为实际采用的 LR 规则。
