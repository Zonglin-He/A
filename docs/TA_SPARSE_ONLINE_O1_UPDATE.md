# 2026-09-29 O1: Sparse-Critic Online Transfer — completed

本轮只问：过去8次专家反馈能否帮助随后24个没有专家的query？

原32历史曝光VidSTG源，每源一次；source-only SHA选一个旧15种正式transient条件，另一SHA排单一32序列。专家位置1/5/9/13/17/21/25/29。Vid-source TA-STVG冻结，candidate相同；UniversalVTG best+PE为critic。持久状态仅768维线性w和公共bias，原始final temporal hidden的start/end/mean特征，SGD .001每专家一次，不调参/不加gate/memory/空间。原生两offset-envelope的精确最大legal-span分数确保零状态32/32为native；没有native bonus或校准。bias在pairwise loss中抵消，保持0。

| Arm | logical expert calls | nonexpert tIoU | nonexpert vIoU | all32 tIoU | all32 vIoU |
|---|---:|---:|---:|---:|---:|
| Frozen | 0 | 46.2867 | 17.9538 | 45.0910 | 16.2558 |
| Budgeted Rerank | 8 | 46.2867 | 17.9538 | 43.9290 | 16.8264 |
| Online Slow-Fast | 8 | 46.2867 | 17.9538 | 43.9290 | 16.8264 |
| Full Rerank | 32 | 50.0614 | 18.6286 | 46.7600 | 17.3325 |

单位%。主比较24 nonexpert Online−Budgeted：Δt=0、Δv=0，逐条指标都相同。这是本流的精确观察，不是一般等价证明。只有position6/Q21/frame_freeze_5改变候选，前后t/v均为0；其余23选择不变。8专家输出三臂完全相同，8次写入loss均下降，最终||w||=.01139664，因此不是状态丢失或更新未发生，但尚无任务层面的future transfer。

补充只读诊断（封存后、无额外推理/更新）：24 nonexpert中Full critic有23次偏好非native，继承残差在其中15次向该候选有利方向移动；native领先分数中位数.04384351，残差对该候选的相对优势中位数仅.00211967。对当前expert排序的pair agreement均值61.9961%→62.4714%，只作描述性信号、不称可迁移表征成立。现证据更直接说明本次学习幅度/读出几乎没改变决策，不能据此决定“representation无信号、必须prototype”。也没有看到逐步变坏的任务表现，不能据此加state-pollution gate。附READOUT_DIAGNOSTIC.json与复现脚本。

32到达状态/前一记录hash完整，复用旧O-all arrival语义；NumPy重算8次SGD与全部到达状态最大误差6.51e−19，loss2.23e−16；专家读取恰8次、非专家0次，256候选双实现评分、32critic标量重算和81公开bootstrap汇总核验通过。在线输出先seal，才读/算Full参考的24个额外expert；全部输出seal后仅读原32授权GT评分。Full Rerank为高预算参考，非保证上界或oracle。

GPU进程累计112.706096s（含capture smoke/加载/解码），CPU在线循环.049839s；online专家证据4新/4复用，full额外12新/12复用，逻辑budget仍8与32。无运行失败。旧媒体、GT、raw hidden/state/predictions留本地，代码/协议/匿名逐条结果公开。

一条历史开发流不能代表任意顺序或长期稳定性；bootstrap只对已封存结果做条件描述，不重模拟在线轨迹。本轮不追加LR搜索、gate、prototype、memory、多序列、空间或C0.6，production CURRENT保持不变。旧C3“单query H更新不及rerank”与本轮“持久rank adapter未产生nonexpert任务增益”分别保留，均不扩展成online TTA普遍失败。

[Full report](../results/tastvg_sparse_online_o1/2026-09-29/REPORT.md), [all rows](../results/tastvg_sparse_online_o1/2026-09-29/ROWS.json), [protocol](../protocols/tastvg_sparse_online_o1_v1.md).
