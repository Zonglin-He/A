# S0完成：真实RVOS有局部信息，当前轨迹只小幅扩展整条tube支持

本轮按固定16个原C3曝光VidSTG来源、clean＋五类5% transient corruption完成96cell。Temporal研究方案冻结为TA-STVG原生候选＋UniversalVTG critic rerank；Hard/current-query OPD/conditional OPD保留历史消融，没有继续救援。

主要结果（sIoU，全GT有效采样帧，整条tube选一个候选）：

| 条件 | Native | 匹配六层oracle | RVOS轨迹oracle | 六层gain | RVOS扩展gain |
|---|---:|---:|---:|---:|---:|
| 五类corruption，源内平均后16源macro |45.3509%|46.0474%|46.2502%|+0.6965pp|+0.8993pp|
| Clean16 |45.8489%|46.6194%|46.7438%|+0.7704pp|+0.8949pp|

Corruption扩展gain CI95%[+0.5453,+1.3098]pp；相对匹配六层只多+0.2028pp，CI[−0.5857,+0.9146]跨零。六层和轨迹并集相对native+1.3886pp。结果有小正信号，但远未达到附件期待的5–10pp；本轮不进入S1 selector/空间OPD，也不以此宣称整个RVOS或空间接口无效。

实际配置：官方Sa2VA-4B固定revision3fee777d49ee9276eac51ea3e5f9b69e81d09be6，BF16/eager、greedy256token、原query、固定均匀5帧、first SEG mask转框。该官方接口让VLM看前5帧，因此这里明确是5帧稀疏证据。31次新专家推理、65次同源同采样像素复用；87/96非空、391/480帧非空、9空证据cell不丢弃。仅H_app按原visual总norm每步.004固定3步，text/motion/参数冻结，动态TTS/ASA/query/decoder，B0始终保留，完整四候选先seal再读取旧16键GT评分。没有GT更新、择步训练、backtracking、selector或online state。

最有信息的诊断是传播范围。对同样12个同时有观察与未观察GT帧的来源，s-oracle候选在观察位置改善+11.9481pp，未观察位置仅+0.3145pp；两者差CI[+5.0596,+21.1175]pp。全16源未观察位置为+0.3654pp，不能把两个分母混起来。对有非空且GT有效观察的同样11源，Sa2VA空间框71.6326%，native45.5328%，差+26.0998pp。它说明该子集上的专家平均有信息；不能称全16源或完整tube的专家得分。当前loss从3.6766降至2.2987，86/96下降（87非空），实际更新有效；整体小gain更符合这版稀疏对齐向整条tube传播有限，尚未唯一定位采样、objective、预算或interface。

正负例均保留：corruption源均值Q09/Q12 oracle +3.0132/+2.2152pp；Q03/Q06 oracle +1.0676/+0.5002pp，但第三步−2.4869/−2.4656pp。14/16源有正oracle gain。第三步源均值11好/3差/2中性（0.1pp），均值+0.4344pp且CI跨零；native保留使oracle不受损是构造性质，不是在线选择安全性。Clean与corruption收益相近，未建立corruption-specific优势。

核验：96历史native/完整suffix逐值一致，4完整回插中2条实际编辑轨迹精确，所有H_app-only与半径断言通过；960双实现task计分、576独立loss数值检查、96mask读回、1728公开scalar重构、3CPU测试通过。主288梯度步＋6验证重放。累计GPU进程276.7632秒（4.61分钟，含加载、smoke和回插），首次15.16GB权重续传准备77.84分钟另计。没有生产晋升或后续新实验。

执行决定：保存真实RVOS接入、expert与H缓存及全部正负证据；当前S0不足以启动S1，不提前实现OPD selector。下一步仍需针对已观察到的空间传播限制另定最小实验，而不是继续救temporal OPD。本轮未运行该后续实验。
