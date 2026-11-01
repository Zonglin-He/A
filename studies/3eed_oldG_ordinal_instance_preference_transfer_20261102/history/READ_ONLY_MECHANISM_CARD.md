# R44 只读候选卡：旧 G 提名两个不同实例组的局部偏好

## 当前判断与执行边界

R43 的直接 z 稠密蒸馏在 full/null 两路都崩坏，当前固定配置已停止。本候选不加 z 温度或步数，而是改为只表达旧 G 的**局部次序证据**，避免要求学生拟合整个 teacher posterior。

这是待验证候选，不是新算法或效果承诺。旧 G 的总体直接收益不能保证经实例组资格约束后的反馈仍有净收益，因此先做同信息直接使用，不启动 TTA。

本卡完成时没有生成新的 G/ranking 预测或任务评分。root 另行授权的唯一执行项是原 R24 A208 与 R37 准备数组的 source256 boxes/scores exact 兼容门；它不使用任务 GT，任何不精确即停止原组复用。FINAL、U、T 和新 GPU 均不涉及。

## 历史审核：哪些已经做过

| 历史 | 已有机制 | 与本候选的区别 |
|---|---|---|
| R16/R17 | 不同 G 的正负 query 集，末端 Linear 上 mean log-prob 排序＋源 KL | 不等于原 R24 全局 F 提名、当前完整 MLP、exact token-z、完整 mask 组 |
| R19 | 历史 q0-negative 案例的成对接口和预算诊断 | 不等于全 A208 旧 G 的新资格反馈；不能当总体有效证据 |
| R24 | 旧 G 的完整 dense teacher KL；另有 qG 对 max-other 的 native 概率 hinge | **已经隐含训练 qG>q0**，而且把同物体其他 query 也纳入 max-other；不能称从未使用该次序信息 |
| R29 | 另一条严格全文 A/B 核验路径获得 22 条偏好，分组 RankNet；先概率后 exact-z | 不是直接从原旧 G 的全 256 分数提名 qG；组表还仅写入原 eligible A/B 的 mask |
| R30/R31/R33/R35/R36 | 沿用同 22 条偏好，增加保持或换接口/模型 | 不隔离同旧 G 的 expanded ordinal 反馈 |
| R37 | 复用 R17 几何，补全全部独立 mask→唯一 query 组 | 新 G 为 point/ID 直接读出，未用旧 G 的 qG/q0 次序训练 |
| R43 | 原旧 G 的 full/null dense probability/z 配对，原全候选 hinge保留 | 不是仅两个不同实例组的局部 RankNet；该 z 配置阴性结果保留 |

有界检索未找到“固定旧 G 全量 F 提名 qG 与 frozen q0、两独立完整组、局部 exact-z RankNet＋保持”的完整已执行组合。它组合了历史已有机制；不能把去掉 dense KL、换反馈集合、改变分组和 keep 覆盖一起归成一个纯因子，也不作新颖性主张。

## 源兼容是先决条件，不能靠近似相同跳过

R24 特征来自 `round24/capture336_attempt001`。R37 的 A 行经 common30→common29→common28→common27→common26 读取 `round25/source_features_attempt002`。路径不同，但 R25 `capture25.py` 对已存在的 R24 样本明确原值读取重存，为 exact-reuse 提供了源码依据。

这仍不是逐元素验证。独立 CPU 门必须对原 D208/A208 每条的 **256×6 boxes 和 256 scores** 比较形状、dtype、全部数值和字节；不能只看 q0、mIoU、同一 checkpoint 或 1e−6 误差。R37 的几何组与 fixed source representatives 只在此门通过后复用。若任一不 exact，停止；不自行放宽容差、换 source、重映射或重算 G。

该门只检查兼容性，不证明 2D mask 身份正确，也不产生新 rank 预测。

## 第一阶段：仅 eligible 直接使用，不更新

待 root 审核另锁后，固定原旧 G：

\[
F=s_0+2v+2p+2r,
\quad q_G=\arg\max F,
\quad q_0=\arg\max s_0.
\]

F 使用原直接 G 的 float64 计算/argmax，不借本轮换成 F32 排名。对每行：

1. qG=q0：没有纠错提名，记录 no-change，不当正确标签。
2. 必须 F(qG)>F(q0)；精确分数并列不产生严格偏好。
3. qG、q0 各自必须落入一个非空、唯一的 R37 完整 mask→query 组；缺失、歧义一律 UNKNOWN。
4. 两组 mask ID 必须不同；同组多个 query 属于同一实例假设，不作互斥正负。不同 mask 是现有检测/几何身份假设，不是实际物理身份已认证的 gold；重复检测仍是风险。
5. 原 source 分数最高、同分取最小 query 的组代表固定。直接输出正组这个固定代表，UNKNOWN 返回 q0。不会通过几何更好的组内 query 冒充实例选择收益。

原始 source、原不受资格限制的 G 直接选框和新的 eligible-group direct 必须共同保留。固定 source 组代表避免用 GT 挑框，但任务 IoU 增益仍不能单独证明语义实例纠错；重复 mask、组代表质量和几何差异也可能贡献。记录 qG、q0、两 mask、两组成员、固定代表、全部拒绝原因、G 分数差和映射覆盖；它们不由 GT 决定。每条最多一条组间偏好，不把一个组里的多个 query 当独立反馈样本。

**先封存全 A208 新直接预测，再独立评分总体与平台指标、改善／恶化和场景表。** 资格接受子集只能解释机制，不能从其任务收益反选门。原全 G 的正收益不能替代这个受约束版本的实际结果。

如果总体净收益不明确、为非正或覆盖太低不足以建立反馈依据，停止该候选；不是继续放宽映射直到挑出有利子集。A208 已暴露，即使净正也只是开发依据，需要 root 再决定是否授权下一阶段。

## 第二阶段：只有第一阶段支持后才另授权

保留原查询 MLP/Waymo source/冻结框，不重启 RGB 或教师。沿用 R30 的局部损失结构：

\[
L_{pair}=\frac1{|E|}\sum_{x\in E}
\operatorname{softplus}\left(-\frac{\max_{i\in P_x}z_i-\max_{j\in N_x}z_j}{0.1}\right),
\]

加相对 frozen q0 的 z 输出保持；不得加入全候选绝对 target cross-entropy，也不得把所有未提名实例都设为负。保留 group-max 与最终固定 source 代表的区别，检查偏好落实和最终第一名；出现第三个未监督组时只记录，不自动补负。

E 为新 G 的可用偏好观测。保持集合若定义为 A\E，则数量和内容会与 R30 U189 改变；pref 分母也不再是原 10/12。这些和监督内容是耦合因素，不能称为只改 teacher。建议未来锁中明确新集合及固定分母，不按训练后损失或测试结果改变；旧 22 条/R30 作为不同系统参照，而不是逐项单因果对照。

新方法至少保留同 A 信息的直接组纠错与无参数 memory。memory 要使用同一可用偏好和固定代表；如果构造 source-z 的最小组间修正，单位须与 z 一致，不能把旧概率增量直接加 z。最终读出及 source 端点要单独验证、保存真实检索索引，不靠 CPU/GPU 近并列随意替换旧赢家。具体 memory/读出与预算由第二阶段另锁，本阶段不执行或预选其超参数。

这个局部次序损失不要求整条 teacher posterior 的数值校准，所以不会因为 ΔG=0 就要求学生向 source-prob 的近均匀分布靠拢。但它仍有学生 gap/温度、梯度、伪偏好错误和跨观测干扰风险。若没有 E，返回 source；不能把缺失反馈当 0.5 相等标签。

## 论文机制与限度

RankNet 在每个查询内部，用两个实值评分的差构造 logistic 成对概率并训练交叉熵；共同平移不会改变该差。它不要求完整全局排序，更不自动认证当前伪偏好为真。[原论文](https://www.microsoft.com/en-us/research/wp-content/uploads/2005/08/icml_ranking.pdf)。

应用到这里，避免的是**拟合 teacher 的绝对 posterior 形状**，不是消除所有尺度问题：学生 z 差和固定温度仍决定梯度。RankNet 原任务有相关性标注，本项目是可能出错的外部 G 伪监督；不能据论文保证有效。

输出保持沿用本项目 R30 已测公式，其思想与旧输出蒸馏有关；不是把 R30 公式宣称为 RankNet 原机制。本候选的研究问题是原 G 的增量次序能否在独立实例组上提供净有效信号，当前先验证第一阶段，而非直接再次训练一套 R29。
