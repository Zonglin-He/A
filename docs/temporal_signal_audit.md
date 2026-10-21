# 空间信息与时间边界的缓存验证

本轮用户明确授权离线探索：原骨干及完整框空间分支是否已包含事件/起止相关信息，注意力是否在起止点附近增大。结论来自已封存HC64、父隔离HC32和Vid32缓存，不改变正在执行的完整框全量。

完整[结果、定义与限制](../results/temporal_signal_audit_v1/report.md)、[全部数值](../results/temporal_signal_audit_v1/report.json)、[固定协议](../results/temporal_signal_audit_v1/protocol.json)、[匿名逐query正负关联](../results/temporal_signal_audit_v1/signed_queries.tsv)、[明确事后敏感性](../results/temporal_signal_audit_v1/sensitivity.json)、[独立核验](../results/temporal_signal_audit_v1/verification.json)。

三项结论分别成立于其测量范围：

1. 原时间decoder确有事件与边界相关信息；不能先验断言它不能利用目标空间信息。
2. 原始目标范围注意力不是普遍的边界峰；同面积真实/置换ROI对比在HC事件内外有信号，但Vid未复现，共变量残差也未建立独立增量。
3. hidden变化的边界关联在三组均有，排除裁剪边缘后仍可见；但位置/观察/面积及原时间头控制后的残差未建立额外信息，不能由此登记新时间方法或承诺涨点。

八项主问题、全部128无标签信号先封存后读GT。后续敏感性明确在看到主结果后提出，新曲线再次先封存后GT，未隐藏为事前确认。全部数据曾被开发或历史评估曝光，父隔离不等于未见测试集。每query宏平均/bootstrap，不能把10,017帧当作独立样本。

仅CPU重放同权重TimeDecoder小子图256次，1536层hidden检查最大3.33786e-6；不构建完整Source、不用GPU、不读RGB、不调用encoder或专家。没有用特征余弦替代attention。434正式冻结源逐项不变，真实worker PID/startticks及进度前进核验通过；正式profile/default/旧结果不改。

当前BoxFixed是原生时间预测之后的空间推理；其aux轨迹可作为已有attention的空间读出范围，但本轮关联不能归因为新增空间分支已经反向改变时间网络。若开发新读出，需要额外证明它纠正原时间分布的错误，而不是重复同一信息；本轮不执行该后续方法实验。
