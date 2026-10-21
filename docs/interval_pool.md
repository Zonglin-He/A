# 最后两轮时间区间推断：预注册协议

主张：区分源模型缺失事件信息、后验不确定性与外包算子、MBR决策的作用。空间锁定原BoxFixed，原readout/观察位置/Source224两offset/六层头与原精确tIoU MBR不变。不继续失败的空间反馈调权重，不跑OPD或训练。整体基线Source，时间对照Global6旧hull+MBR，Strong仅历史参考。

## 第一轮

HC64原缓存原冻结小头恢复，封存12个单层单offset原网格MBR与Global6（严格13候选）及各view/hull概率后才读取GT。统计signed起终点误差秒及GT时长归一化、过/欠覆盖、长度比、tIoU/vIoU、m.5/.7质量、期望长度恒等；事件<5/5-10/>=10秒，无标签端点分歧<.25/.25-1/>=1秒分组，近乎不交tIoU<.1仅描述。oracle只在13个预封存候选离线，不搜索GT全合法区间、不GT选层/offset/温度。没有完整448后验则未测，不新增448。

## 第二轮唯一候选

各view先在自身a<b区间归一化，然后按真实起点fid[a]、排他终点fid[b]+1 scatter-add到并集U，各层两offset12个完整分布等权混合，不做区间配对、不插值、不均logits/温度/裁尾。与旧hull各用MAP及原MBR共2×2；主比较Mix_MBR−Hull_MBR。MBR决策保留U上所有a<b，包括q0；最后只一次原interval BoxFixedDP。相同冻结空间轨迹另测纯时间。

配置scripts/interval_pool/config.json及32×2名单在新GT之前锁定。第一轮诊断不触发扫参或改变第二轮方法。确认候选先按当前128父排除、固定parent/query序取32每域；历史full曝光无法排除，因此名称为冻结父隔离开发复核，不称独立确认。64条新域样本缺六层/空间/mask缓存，各一次共享224捕获及同ReferDINO八观察，Source/旧MBR/新MBR共用，不补448/条件decoder，不恢复formalVid/CG。

## 定版和成本

目标共同要求轻量且涨点，不能单报缓存零encoder。记录读入/头/后验构造/MBR/DP/共享捕获/专家/搬运与冷启动；概率表构造操作量不等于端到端倍率。两方向复核平均tIoU上升、固定空间及完整vIoU不降、无新增网络计算才选择Mix_MBR；区间跨零只称初步，正式独立确认仍缺。若不一致或无收益，保留Global6旧hull+MBR并结束本改进，不扫第三组权重；只提速则工程备选，不声称非劣或时间涨点。全面瓶颈审计可提出待验假设，不能掩盖本轮负结果。生产当前Strong未自动变更；最终明确研究定版与生产默认的边界，保留旧版恢复及所有结果。


## 已执行结果及恢复

本协议已完整执行，最终[分析与定版](../results/interval_pool/ANALYSIS.md)、[机器可读决定](../results/interval_pool/decision.json)、[工程证据](../results/interval_pool/engineering.json)、[暴露/隔离证明](../results/interval_pool/cohort_provenance.json)已保存。HC64、HC32、Vid32均不支持新mixture优于旧hull+MBR，按预定规则保留Global6旧读出，停止本候选，不调权重。生产Strong和所有旧源码/缓存/结果未改；原Git13a681a及更早历史完整保留。

旧固定归因/干预和新32×2各自源快照/输入哈希/receipt/seal分目录保存，评分只在seal之后，不能放宽旧验证器接受新运行代码。HC32/Vid30已有历史Source及观察位置均逐位恢复；无旧预测的Vid2只做同轮共享配对，不伪造历史恢复证明。768视图独立物理scatter验证最大误差1.39e−17，105工程测试通过。新64各仅一次224捕获和一次原八帧专家，无448/条件decoder/更新。

实际冷/热边界：首条capture包含Source与ReferDINO延迟加载；均值包含首条，余31条单列。Python/import/setup/资产哈希不在query计时中，不能称应用完整冷启动。Source进程显存与专家子进程没有合并实测，不能把前者当总峰值。所有嵌套成本不重复相加，三臂共同执行的query wall不是某一单独方法的端到端对比。
