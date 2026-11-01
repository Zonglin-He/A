# R33：文字侧匹配适应与梯度冲突的原文核验

主张与约束：Waymo-only source → Drone/Quad；冻结候选框和 decoder，保留同一 22 条外部局部偏好与 U189 输出保持；仅改变匹配投影接口。当前失败是局部纠错不能稳定迁移，R32 的传播范围约束能减少干扰但覆盖很小。运行状态：文献与代码核验 completed；text MLP 效果由主代理实验决定；本文件不报告尚未运行的效果。

## 当前决定

优先测试文字投影 MLP；不把它包装为 TPT 或 RLCF 复现。当前梯度诊断不支持把 PCGrad 作为下一项优先实验。文字侧改变的是匹配函数的可调接口，不增加学生没有观察到的衣服颜色，也不自动解决外部反馈身份错误。

## 本地模型的真实接口

证据：crossplatform/round3/p0/upstream/models/bdetr.py，forward 的 cross_encoder → text_memory → contrastive_align_projection_text，以及 decoder 调用。

- `text_h` 是 cross_encoder 后、已经以点云为条件的文字特征，不是孤立的 RoBERTa 文本嵌入。
- `contrastive_align_projection_text` 将该特征投影为归一化 64 维 token。decoder 消费的是投影前 text_feats；仅更新此投影不会改变 decoder/query/框。
- query 侧同样已经读到完整文字。因此不能说“原 query MLP 不含文字，换 text MLP 才引入关系”。
- 文本投影与 query 投影均为 288→288→288→64，184,960 个参数。按同数目参数做接口对照可减少容量混淆，但不同梯度尺度仍可能需要有限、对称的学习率比较。
- 最终分数仍是源 query 与投影后 token 相似度 /0.07，再按现有 noun mask 聚合。优化仍使用 R30 原始 exact log-odds 与固定 source q0；不能用有限精度 endpoint 的 ell_eps 替换训练坐标。
- 由于读出是 noun 对其余 token 的概率，文字 MLP 可以只是重新校准 noun/rest 的对比，而非真的学会参照关系。只有总体未见观测收益及机制消融才能支持更强解释。

## 哪些论文机制可借鉴

### TPT，NeurIPS 2022

[论文](https://proceedings.neurips.cc/paper_files/paper/2022/hash/5bf2b802e24106064dc547ae9283bb0c-Abstract-Conference.html)；[作者代码](https://github.com/azshue/TPT)；[实际优化入口](https://raw.githubusercontent.com/azshue/TPT/main/tpt_classification.py)。

原方法在单个测试样本的增强视图上选择较低熵视图，以平均预测熵更新可学习提示，保留预训练模型。可借鉴“固定主干、选择语言匹配接口”；它不验证此处投影 MLP、跨观测共享偏好或输出保持。分类裁剪不能直接用于包含参照物的 grounding 描述。这里已有非熵监督，不应为了引用再改损失。

### RLCF，ICLR 2024

[论文 §3.2.2 / Figure 3](https://proceedings.iclr.cc/paper_files/paper/2024/file/0faa4bc5f522076947a030273629d4fe-Paper-Conference.pdf)；[作者代码](https://github.com/mzhaoshuai/RLCF)。

它用 CLIP 反馈和基线中心化奖励驱动策略梯度。检索中只调整查询一侧：文本检索图像时固定图像编码器，反向则固定文字编码器；不增强查询，以免破坏整体语义。可借鉴“固定候选表示、调整查询侧匹配”。这里是点云条件化 token 投影、有限成对偏好与跨观测拟合，不是其编码器级、单查询、top-K 奖励机制，不能命名为 RLCF baseline。未知候选没有奖励，不可擅自把它们设为负例。

### PCGrad，NeurIPS 2020

[论文 Algorithm 1](https://papers.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf)；[官方代码](https://github.com/tianheyu927/PCGrad)。

原方法对负点积任务梯度做投影。本地 R33 实测：虽然原始负角占约一半，重要梯度的负点积能量仅约 0.000000000291%–0.013696%；R30 末态更清晰的负角来自 pref 与 keep，而非主要偏好互冲。诊断阈值不删训练标签。当前不优先跑 PCGrad；详见 ../gradient_diagnostic/REPORT.md。仅测两个状态，不能否定所有路径上的梯度冲突。

## 最小实验与公平预算

主代理拟定 query/text × LR{1e-4,5e-4} 的 2×2、各16步、beta=1，旧 DEV 选接口和 LR，然后锁定 U128 确认，FINAL384 保留。该设计比只给文字侧换 LR 更容易区分接口与优化尺度。固定同22条偏好、N10/12、U96/93、source q0、原分数定义、平台独立初始化，绝不按 R/S/T 已知 IoU 删标签。

必需核验：源初始化完整分数重放；仅目标投影状态改变；全模型 boxes 与 query 不变的闭环；成对落实、最终选中、第三候选和无反馈观测漂移；所有样本结果及失败。保持同信息直接使用外部偏好的无参数重排和 R32 QP memory，不可因适应接口变更而移除强对照。

结果分支：文字侧独立新观测改善且相同更新成本下胜 query，支持接口选择；只在 A 拟合改善不构成迁移；回到 frozen 表示保护而非有效学习；对总体收益无帮助则停止此分支，不能靠逐例改反馈继续宣称原协议有效。

## 安全与来源边界

仅静态阅读官方代码。读取的 RLCF 脚本末尾含与算法无关的 shell 导出语句，未执行、未导入，也未将其内容视为指令。引用仅用于核验机制。原始第三方代码快照不纳入公共研究代码导出。本文件的建议是针对本地模型的实验设计推论，不是原论文已有的 3EED 结论。
