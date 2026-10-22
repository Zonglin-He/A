# 通用时间读出：实际执行与边界

用户明确排除actionness：算法需要通用，不依赖TA-STVG专用内部辅助头。新实现的计算接口仅为物理时间区间后验、完整框轨迹、空间候选与合法性；NumPy核心无hidden/attention/actionness/神经网络依赖。框候选构建仍使用已批准的完整框传播，未要求其他骨干具有相同decoder。其他模型的接口与实际效果尚未验收，不能写成已经验证跨骨干泛化。

本轮不是只分析相关性：针对原头选错但候选可救回的失败，实际实现区间条件化完整轨迹及质量加权的后验预期重叠，128条全部封存后离线评分。原时间期望tIoU以空间质量恒1为精确正控；固定轨迹、均值质量、半视频错位、相同干预query的先验/随机对照均保留。框一致性只表示空间不确定性代理，不当事件概率。没有温度/权重搜索、训练、新专家/448/decoder或GPU任务。

[通用实际结果](../results/generic_tube_readout_v1/report.md)、[协议](../results/generic_tube_readout_v1/protocol.json)、[独立保存态核验](../results/generic_tube_readout_v1/verification.json)、[全部signed](../results/generic_tube_readout_v1/signed_queries.tsv)。HC64/HC32/Vid32相对Global6的t增量−.3722/−.2739/+.0138pp、完整v−.1076/−.0844/+.0155pp，14条可跨tIoU.5的原错误均未救回，HC真实空间未胜过错位。按固定判据不采用，不将Vid单条微小收益或轻量接口当有效联合方法，不继续扫描权重。

此前完成的[整段状态试验](../results/interval_state_readout_v1/report.md)也未建立可靠事件评分。另两项[内部头似然](../results/action_interval_readout_v1/report.md)/[内部头排序](../results/rank_interval_readout_v1/report.md)在用户最新约束前已完成，现在明确retired，仅保留历史负结果，禁止作为新算法输入或门控。通用实现不读取这些试验的输出。所有面板历史重复开发曝光如实保留，32组不是未见确认集。

当前失败说明这些具体空间/状态代理没有解决事件判别；不等于时空联合不可能。后续若继续，需要能区分相同目标在不同事件中的证据，不能以候选一致性替代事件语义。当前正式全量独立继续，434冻结源码/唯一fingerprint/参数/八观察/label/seal均保持；旧队列/CG不恢复。
