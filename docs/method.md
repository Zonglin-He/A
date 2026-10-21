# 当前方法：完整框固定推理

用户在2026-10-21明确将空间主线从OPD切换到原BoxFixed。当前空间为同一ReferDINO的ERef初始化、完整原生框传播、原固定评分和精确DP；不使用committed_policy的新评分，不执行OPD/Direct/KD更新。`configs/default.json`只保留当前必需配置。旧OPD数值配置在`configs/legacy_opd.json`，旧源码和结果保留，不重写旧评价。

一次原生224 Source流程保留原双offset、FP16视觉编码和FP32两遍原生解码顺序，去除为适应创建的可训练残差/LN/优化器。原生预测按既定规则选择八个观察位置。同一次ReferDINO前向选择一条跨帧query，导出其原生八框和软mask logits；仍使用原Source域校准的鲁棒ERef初始化和HC=.5、Vid=1初始化强度。

软mask只加权已有Source特征对应与原型，不重新决定框边界。双向top20/.1对应以固定WLS估计逐轴平移/缩放，将完整原生框传播到各物理节点，保留前/后两个来源和原始ERef三槽。ESS、方差、尺度和图像交集检验不变，失败保持回退，不将缺失解释为事件不存在。范围RBF与关联余弦构成原u；原加性score为.5*mean_obs(u)+.5*mean_unobs(c*u)+.25*mean_event_edges(cprev*c*L)−.1*mean_all_nodes(G)，由精确DP读出，无训练或GT选取。

几何、对应、来源、可靠性在`stvg.evidence`/`stvg.spatial`中独立于时间区间构建。interval视图显式读取给定物理区间并恢复原BoxFixed分母/边支持；full_timeline视图不使用Strong event/active/区间内外身份，作为明确的新方法扩展：观察/未观察分母为8/T−8，相邻边支持全时间，保留原置信下权及fallback。全时间辅助轨迹不保证与原裁剪版相同。

Strong仍为已知时间参考：448原生后验与224六层平均各占一半，保持原物理MBR。空间切换的HC64工程回归核验动作、完整框、score和Strong区间；这不是新精度实验。原BoxFixed HC64完整vIoU31.3985%、未观测sIoU50.9672%，属于已曝光开发面板，不是独立全量确认。其他数据方向/骨干未因迁移自动获得新方法有效性证据。

实验时间模式`native224`仅使用原生224最终层后验与原MBR，不运行448，也不构建Strong专用六层融合后验。`tube_replay`使用全时间辅助BoxFixed轨迹，把框读取范围放大1.5倍后构造连续网格覆盖M；r为已有对应可靠性（不是事件/身份概率），缺有效mask/对应时r0。固定alpha=.5，w=1−alpha*r*(1−M)，在原时间decoder每head/query的同帧合法motion视觉token组中保持注意力总量重分配；文本概率不直接修改。实际原生时间memory/query/text/pos/mask/offset完整缓存，最多每offset重放一次必要时间子图，不使用appearance ROI替换motion memory。alpha0走原注意力算术，必须逐位恢复原生输出。

新时间区间出来后只执行一次最终interval BoxFixed DP，停止，不做交替迭代。`tube_replay`是experimental，不预先声明有效或等精度。协议/输入/缓存命中和实际成本在HC64固定A/B/C/D试验中记录；全部预测封存后离线GT。正式Vid仍暂停于8405/10303，CG继续暂停。

第三方来源/权重版本见[third_party.json](third_party.json)与[assets.json](assets.json)。历史OPD/Strong定义和负结果可由Git历史及[history.md](history.md)恢复，不删除旧源码、数据、权重或缓存。
