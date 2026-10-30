# 无标签竞争候选观测重合诊断（仅准备，不改G）

选择锁：只取full5912已封存trace中rgb_text理由为
`same_evidence_competing_candidate`的记录。分别在Drone/Quad按
`SHA256('R7_same_component_diagnostic_v1|'+platform+'|'+id)`升序选最多8例，合计≤16。
不使用GT、IoU或任务成败选择。执行计划先绑定full seal与manifest/code hashes；
SELECTED_PRIVATE在打开选中P/RGB/camera/cache前写出。

重放严格同采样/点hash、实际JPEG、camera白名单、已锁projection、全点像素zbuffer、
HSV/source crop/components及G拒绝altq，要求G诊断与封存trace完全一致。
无真实模型/GPU/scorer/GT接口。

逐例私有表：source与alt无GT AABB IoU；source最大local color component点数；
alt颜色支持点数；两集合交集；sourcecomponent被alt覆盖比例；alt支持来自sourcecomponent
比例；alt在source local component外点数；以及在完整visible color graph中与任何source
component点均不连通的几何component内，alt点数/component数/≥12支持component数。
“独立”仅指冻结radius=.35m图上的观测分离，不指真实独立对象。

描述分组在真实执行前固定：两重合比例均≥.8；另外统计该分组中没有额外≥12点支持的
全局独立component的数量。该阈值不是G新gate，也不改变实验臂、候选或任何预测。
报告PUBLIC_AGGREGATES仅含总数/平台聚合/均值/hash，不含ID、原caption或坐标。
PERCASE_PRIVATE/SELECTED_PRIVATE保留本地，不公开。结论最多为候选歧义启发式是否使用
高度重复观测作为替代支持；不能由此断言GT同实例或声称V2改进。
