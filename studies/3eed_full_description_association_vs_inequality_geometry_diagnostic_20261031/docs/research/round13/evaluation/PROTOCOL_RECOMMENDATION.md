# R13：先分开验证关联与几何

状态：协议准备；未执行R13模型、预测或目标标注评分。旧R12代码、预测和结果不修改。

R12已经证明部分独立实例与跨观测支持可构造，但没有证明完整文字目标身份，也没有证明可见局部面能充当完整框外表面。单帧与多视图直接框在Quad均降低IoU，主多视图在Drone没有实际改框。R13因此只检验两个互不混杂的干预；不是给旧G增加优化步骤。

## 输入与复用

采用root确定的新108内容锚点：Drone32（1场景），Quad76（32/12/32，3场景）。按既有JPEG+点文件hash排除prior700与R12全部501前端观测，排除并集914组；剩余5场景440/0/297/12/90。第二Drone场景没有剩余新内容，不能补旧数据称确认；四场景也不是新场景，目标VAL历史研究暴露。新108在配置锁定、所有A/B预测完成后仅一次最终任务评分。

R12独立单帧7个改框与主多视图2个改框的并集8例，按已封存changed flags选出，单列exposed regression。不得与108合并为116个独立确认样本，也不按8例的已知IoU改动本轮阈值。

旧8可复用同一次forward的原256候选、q0、源特征、同image/config的独立mask。新108必须重新获得同checkpoint、同预处理的strictFP32源缓存。旧缓存没有完整token/query语义投影或GPU原生Top10；更换grounder文本/采样/精度须重新forward。完整文字曾输入grounder不等于当前firstnoun读出已经具有完整文字关联能力。

## A：只改完整文字关联

候选为同一源池256个原6D框，预先完成无GT去重并保存每个组的原slot成员、代表规则和映射。完整文字VLM、noun-only和预锁deranged-language共用同一候选/ROI/图像/候选呈现顺序、预算和输出约束。VLM是额外冻结的语义关联助手，不是目标框真值或3D教师；首版不训练它。

选中qA后必须逐bit输出原pool[qA]，不做组内中心聚合、尺寸修改或几何优化。source输出原pool[q0]；NA/无解/多解保留q0且显式记录。去重如丢弃候选，应保存未裁剪原pool和候选覆盖，以区分关联错误与候选缺失。不要把跨进程相同slot编号当作同一实例。

必须分别记录q改变、原框改变、候选组改变、辅助模型拒绝和完整文字解析失败。主比较fullcaption相对source、noun-only和deranged-language，保持所有锚点分母。只赢被破坏的shuffled对照不够；完整文字还须相对源读出及noun-only有正任务增量，才支持本配置的关联价值。

q0框与qA框取一半不是对象身份对照。若另报它，只能命名为generic coordinate interpolation；它可能位于两个物体之间，不能证明“半关联”。A本身已经是无更新direct-use基线。

## B：固定q0，只换观测约束

字面臂：source / direct_full / direct_no_rgb / half_full。所有臂q0固定，所有支持绑定同一源缓存、原图、原点文件、标定、mask和物理去重点。full/noRGB共用输入gate、点集、变量、sourceprior及bounds，只关闭RGB坐标项；half为source + float32(0.5)*(full-source)。不把其差异叫纯点云部署效果，因为点mask本身仍可来自RGB。

前景点在框内、前景像素由投影框包含只能提供单侧约束。禁止可见面=完整外表面的等式；禁止把点极值/均值直接当完整框位置；无传感origin与solid-object证书时不引入free-space。source已经满足全部有效不等式时，先验最优应精确source，不为提高非零率强行修框。

中心有界、size只能从源值扩张到预锁1.2倍是有限先验选择，不能称观测恢复了完整amodal尺寸上界；不能用局部前景缩小框。逐坐标保存observational lower-bound/containment证据与sourceprior消歧。raw非正尺寸、无mask、无支持、数值异常显式fallback，不能暗改source；有限非收敛结果按预锁策略保留状态。

主比较full相对source、half和noRGB，报告全部样本和GT-free共同eligible/changed并集；各臂own-changed只能描述，不能直接跨不同分母比较。source相对full的IoU与相关坐标误差均需改善，且full不能只表现为通用缩步收益。尺寸从未变化则不作尺寸能力结论。

## 评分与身份边界

所有A/B臂、新旧两个数据层分别完整seal，代码/配置/源权重/原图与点hash/候选映射全部绑定后，root逐阶段CPU_SCORE_RELEASE才可打开评分index和标注。主task为nativeyaw0，使用已审计7D GT坐标转换和float64 3D IoU，不加PCA、GTyaw、GT候选选择。严格IoU>.25/.5；同时报告micro、场景等权、center/size误差、逐坐标绝对误差与signed gain/harm/tie、signed sums和全部负尾。Drone新内容仅1场景，scene-bootstrap CI应NA，不能用帧数制造精度。

A的IoU增量不自动证明精确身份。仅封存后：先按原metadata的稳定instance ID核schema；若没有可核ID，可对同帧重复annotation盒按已锁几何规范去重，保留冲突/歧义。比较q0/qA对目标及其他标注物体的匹配；沿用已有诊断IoU>.25、target-over-other margin>.1才称annotated-spatially-confirmed，不称语义身份真值。未确认不是已证错目标。不同描述分数、VLM自信、框内支持比例都不是身份GT。原pool的最大IoU只是candidate headroom；不等于某个proposal对应正确instance。

B按source q0的封存后空间匹配状态拆分center/size误差，可以发现“疑似正确对象但完整框错误”；不得以该标签筛选G或改变primary分母。A→B组合前必须先保留A关联原框效果与B固定q0几何效果。

## 资源裁决与条件下一步

新108为唯一当前配置效用检验，旧8仅回归说明。分别判定A与B是否在Drone/Quad都相对相应强对照获得正净任务增量，收益不能由少量有利指标、各平台挑臂或通用half缩步解释；报场景差异和负尾，不设凭空总体显著性阈值。

A/B任一未满足其主张所需证据，不构造Cx组合，也不进入神经TTA。即便二者通过，Cx与参数更新仍需另行锁定：更新只用观测A，任务读出在未用于构造伪目标/优化/门控的另一观测B验证；关联和静态对应要独立成立，运动/歧义返回NA，same-information direct-transfer/no-update必保留。同输入拟合伪目标下降、exactfit到伪框或当前参数梯度非零不是可迁移监督收益。当前不执行该阶段。
