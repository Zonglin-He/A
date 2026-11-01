# R44 phase2 计划与损失静态审查

状态：**提案静态通过，未运行模型、损失函数、优化器或任务评分；不是 GPU 执行授权。** 对照原 R30 fit/predict/memory_all 源码核查，不更改算法代理文件。

范围为原 R24 A208→原 V64，source 初始化查询投影 MLP 184960 参数、两平台各16步。全部 E32 保留，其中 Drone14、Quad18；保持集合为精确补集 U176，其中90/86。每平台偏好分母固定14/18，loss 为 group-max exact token-z RankNet（0.1）加 β1 的全部 U 相对 frozen q0 的 z 输出保持。没有混入 dense teacher KL、R24 hinge、参数锚定或 RGB 新训练。

A 使用所有 R37 完整组的 native 概率组内最大值，再返回原 source 最高分固定代表；未映射 query 保留。V 使用原生 float32 概率 argmax，不伪造 V 的完整组表。组 max 偏好满足、最终固定代表和第三组胜出是不同事件，第三组不能自动补负。固定代表也不保证 IoU 收益都来自真实实例身份纠错。

两个 memory 均沿用原 R30 概率单位规则：对 P/N 全组写入正负 `max(0.1-gap_probability,0)/2`，source query×归一化 noun-key，再做 L2、每 A 行最近 query、CPU float32 top4、exp((cos−1)/0.1)、alpha1。E-only 与 all-A 零 U 库分别保留；修正仅加入 source native probability，不能加到 z。实际邻居索引、相似度和权重保存，独立审计不能通过另算 float64 top-k 替换近并列赢家。零修正端点严格 source。full V current-G 只是额外系统信息对照，与同 current-G direct 配比，不能把它全部算成参数收益。

当前不同于原 R30 的因素包括反馈生成路径、E及偏好分母、完整组覆盖和 U 保持集合，不是只改 teacher 的单因果实验；RankNet/输出保持也不是新算法。A/V均已开发暴露，不能把净正阶段一门或未来 V 收益称为独立确认。

实际 runner 锁前/运行时仍需校验：phase1反馈与组逐值绑定、32唯一观测、U精确补集；原 fit24 loader 无 fit29 import hook；source参数/输入原样、q0冻结、z0停止梯度；最终 p/z、更新前后loss/gap/keep、global候选及 memory 真正索引均持久化。上述是实现与审计守卫，不改变已提案配方。完整模型闭环与进一步确认需另锁，不因为缓存拟合成功就宣称部署收益。
