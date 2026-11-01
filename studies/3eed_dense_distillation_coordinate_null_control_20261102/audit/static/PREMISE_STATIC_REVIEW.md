# R43 独立只读审查：表达前提与必要竞争解释

状态：**前提代码/协议/汇总静态审查通过；本文件不是独立数组重放或任务质量审计，也不是训练授权。** 审查未运行模型、优化器、数组诊断或任务评分，未读取 FINAL。

审查对象为 `premise/PLAN_LOCK.json`、`premise/IMPLEMENTATION_LOCK.json`、`crossplatform/round43/premise43.py` 和 `premise/SUMMARY.json`。算法作者已运行的 CPU 前提测量与本次静态审查分开。

原 R24 的 teacher 构造逐项一致：先按原表达式计算 `sourcep.astype(float)+2*vision+2*position+2*relation`，再转成实际 R24 使用的 float32。完整 A208 和每条 256 候选保留，原正值、负值、零值未筛选。数组输入逐文件核对既有 seal SHA，新增完整前提向量先 seal，之后做无标签汇总；输入为 source 缓存与旧 G 缓存，没有模型加载、GT 容器、媒体或任务评分入口。使用 torch CPU 不等于使用 GPU，代码显式检查 CUDA 未初始化。

teacher/源概率的候选 posterior 使用稳定 LSE 的 CPU64 log-probability，KL 使用有限 log-probability 的差值，避免对下溢后零概率再取 log。额外 CPU torch32 softmax 只度量数值差异，没有宣称复现原 GPU kernel 的逐位结果。source z 来自已缓存归一化 query/token 的 CPU64 点积与 LSE，不是原模型重新前向，也不等于原 GPU32 的逐位 z。这个边界在协议和汇总中已明确。

作者汇总显示：94/208 条 teacher 分数跨度大于 1，20/208 条 teacher 峰值超过理想质量坐标后验上界 0.1763516676。因而不只是纸面上可能，旧 G 的确存在超出自由质量坐标表达族的行。这里不把 CPU float32 的阈值端点比较当任意精度证明，也不把 peak 边界当完整分布可表示性的充要条件。rank_adapter 无界残差的历史反例继续保留。

**更强的竞争解释也得到支持。** 109/208 条原始外部增量为零；这些行的 teacher 对原 source-probability posterior KL 为 0，对 source-z posterior 的平均 KL 却为 22.798292。全体 source 概率候选分布平均熵 5.281370 nats，source-z 同数值温度下为 1.090218 nats。因此 s→z 的单因素改动会产生很强的非 G 校准压力。不能把其后任何变化全部称为吸收外部 G 知识。零数值不自动意味着语义 UNKNOWN，不能为消除该压力而删除这些行。

本前提测量只证明表达限制及坐标初始化失配存在，**没有测量 teacher 身份是否正确、跨观测迁移是否有效或任务指标是否改善**。训练损失变小仍可能学到错误监督。

后续训练草案须满足以下条件才算隔离当前因素：

- full-G 两臂使用同一个原 F32 teacher 分布；仅 teacher KL 的 student 候选分布由 softmax(s/0.25) 换为 softmax(z/0.25)。
- source KL 仍使用原 source-p target 和 native-p student，hinge 仍用 native-p、原 0.05 margin；参数正则、原 source 初始化、查询 MLP、AdamW5e-4、16 步、最终 native-p 读出全部保持。
- R30 只固定更新位置/预算并作为结果参考；不能把 22 条局部偏好混成原稠密 G 的复现。
- 保留同 source-probability teacher 的 null-G 两坐标对照。不能悄悄给 z-null 换成 source-z teacher，从而掩盖分布坐标失配。null-G 仍有原 source-top hinge，故不能承诺它一定零更新。
- full/null 的差与交互只解释这个固定训练配方中的外部信号增量，不是外部 G 普遍有效或一般因果保证。原 G 同信息直接重排、相同历史信息的无更新缓存不能丢失。
- 不增加层、不新增 RGB、不筛反馈或按 GT 选温度/预算；所有拟合与预测在单独授权后执行。A/DEV 已暴露，不能称独立确认。

当前裁决：**前提足以支持提出一个含 null-G 的最小方案，但不足以支持直接采用 s→z 修复。** 训练草案仍需按实际代码逐项核查与 root 单独授权。
