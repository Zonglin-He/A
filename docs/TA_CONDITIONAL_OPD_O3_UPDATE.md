# O3 Conditional Online OPD：四臂已完成

**这版 Conditional Pairwise / Reverse-KL 在60个非专家位置都没有改变最终选择，task增益均为0。训练确实发生了，但还没有转化为未来query收益。**

本轮严格沿用O2五条homogeneous16-source流、25%预算、原候选/hidden/UniversalVTG。新两臂固定768→128→1 ReLU，98561参数、相同初始化、零输出头；SGD .001每专家一次、lambda1、tauE=tauS=1、最多4个prior expert候选集replay。Replay为current+mean past，两臂只有loss不同。四次专家到达只用到最多3个历史集，未检验buffer淘汰。

附件前段提及拼g/score，末段明确首版768→128→1：本次按后者，用已经过query-conditioned多模态编码的768D特征，native score加在输出端，没有擅自扩为1025D。此配置的局限应保留，不能声称已验证所有context-conditioning方案。

| Arm | 非专家macro tIoU (%) | 非专家macro vIoU (%) | Δt vs Budgeted (pp) | Δv vs Budgeted (pp) | 改选择 |
|---|---:|---:|---:|---:|---:|
| Budgeted Rerank | 28.8343 | 17.1304 | — | — | 0/60 |
| 原O2 Linear Slow | 28.9637 | 17.1758 | +0.1293 | +0.0454 | 4/60 |
| Conditional Pairwise | 28.8343 | 17.1304 | 0 | 0 | 0/60 |
| Conditional Reverse-KL | 28.8343 | 17.1304 | 0 | 0 | 0/60 |

五类分别的两MLP增益全部为0。源内五条件平均后bootstrap12个非专家父源，条件区间[0,0]来自这些预测逐条完全相同，不是“总体等价”的统计证明。All-arrival指标也与Budgeted完全相同。旧Linear两条正例Q09/drop与Q16/occlusion保留，两MLP没有获得这两处改善。

## 更新与读出诊断

- 两种loss的总current+replay目标均20/20下降；当前样本项各19/20下降，说明replay会与当前fit发生取舍。
- 40次SGD均独立重建。零输出头导致每流首步隐藏层梯度为0，这是初始化的数学结果；后续30/30次隐藏层梯度非零。
- Pairwise非专家residual score range：中位0.00045055，最大0.00460400。
- Reverse-KL：中位0.00008901，最大0.00111187。
- 没有做LR、alpha、温度、宽度或步数搜索，没有额外当前query H更新。

因此本轮只能说固定小步配方未产生task transfer，不能归因为“conditional representation无用”或“reverse-KL不行”。同理，reverse-KL本身仍依赖teacher score scale与温度，不能以mode-seeking叙述替代校准证据。

按用户“两种loss效果相近时优先OPD”的规则，**Reverse-KL保留为后续conditional实现候选**；这一选择不等于在线收益已证实，不改生产注册。Fast sparse reranking继续作为有实际效果证据的temporal参照。

## 核验与成本

80到达、5流重置、20专家读取，非专家当前teacher读取0；40次新CPU SGD，0新GPU、0backbone、0expert计算。在线循环约0.735秒，独立NumPy约0.953秒，不含旧缓存创建及开发/报告。

梯度重建最大误差6.08e−16，状态8.67e−19，loss7.22e−16；540项公开均值/区间与所有选择/状态链/replay顺序核验通过，3项CPU合同测试通过。先seal预测再读O2旧cached GT-derived指标，无新GT文件读取。16唯一source在五类条件下重复，不当80独立样本。

## 新指令覆盖与接续

新指令到达前，旧O3 KNN memory已经CPU完成：不同的O1混合32源流上，24非专家t+2.0778/v+0.4862pp，CI跨0，23改选择、v9改善11退化4不变、1条>5pp损害。它作为superseded历史单独归档，未继续、未与新O3择优，不删除结果。

最新要求只temporal，S0暂缓。Sa2VA-4B官方源码及隔离依赖已准备，权重下载在partial阶段停止，S0没有进行任何模型推理。原文件保留，下一次S0仍按用户选择优先RVOS specialist。
