# TubeDETR 原生指代mask信号资格检查

2026-10-21，按已授权的OPD/Direct机制研究实施。原ROI与两个机制探针均已完成并发布。本阶段不把mask Dice当作已解决STVG事件/身份问题的新方法，而是检查现有一个ReferDINO是否提供比几何和共享特征ROI更可靠的候选评价。

## 冻结范围

同一已曝光Vid128/128父视频、原seed41清单、原八个物理帧。每query最多一次原ReferDINO调用，模型冻结，只补存原infer本已计算的选中轨迹与最多3个替代query的软mask和逐帧文本分数。选中框evidence SHA、RGB SHA、物理帧、mask/pixel_decoder画布大小均须匹配旧证据；不能新增专家、观测帧、学生适应、参数候选或全量。替代query按原全视频分数选，不依GT。

控制器按masks(GPU)→bank(CPU)→score(CPU)执行，不自动重试。GPU独占runtime/gpu.lock；mask与reward bank各128完整seal后，独立CPU才读固定标注。新tools独立，原生产模型、ROI冻结代码/计划/旧结果不改。

## 两个候选信号

对同帧候选矩形b与原生query mask概率m，令I为矩形与软mask的连续面积积分，A为完整矩形面积，M为有效图像内mask总量。第一个分数是 `2I/(A+M)`。面积同时约束覆盖和紧致，图像外矩形面积仍在分母；不阈值化mask、不裁掉候选框的越界面积。用带padding校正的积分图和连续角点双线性插值，精确积分分片常值像素密度。

第二个分数为选中轨迹的同类分数减去可用替代query分数的最大值。选中/替代框IoU≥0.5视为可能重复而不作竞争项；全无可用项则退化为第一个分数。0.5事前固定不搜索，框去重不等于身份标注，替代query不能称已知错误对象。几何代价不加入两个反馈分数。

两奖励均可向Direct提供同样的连续面积梯度，已有CPU有限差分检查；不通过detach或阈值化剥夺Direct梯度。共享特征裁ROI与原生query专属mask的区别来自[ReferDINO原论文4.1/4.2](https://arxiv.org/html/2501.14607v2)及本地原infer；这不证明本奖励或OPD优越。软mask依然可能复制错误教师，没有解决“八帧未观测事件”的保证。

## 同候选比较与停止条件

原完整diagnostic bank不变，主比较只用原policy32候选，Geometry/Layout3/MaskDice/MaskContrast在同框、同有效GT观测帧上比较。先帧内、再query内、再有效query等权；无GT的query单列，不能从奖励排序推断其质量。报告成对排序准确率、Spearman和top1 IoU regret；最高分并列时按所有并列候选的平均IoU计算regret，避免候选顺序偏置。

开发信号资格事前规定为：成对排序准确率不低于Geometry、top1 regret不高于Geometry，且至少一项严格改善。配对父bootstrap区间也报告，但此门槛不是独立显著性或适应性能证明。新旧教师质量分组仅用于诊断，不作为推理门控。

逐帧原生文本分数另按真实事件区间计算query内正负帧AUC，只在存在两类帧的query上统计并披露分母。该分数不预先当作事件概率或遮挡标签，也不用于两个候选奖励的GT门控。

任一资格通过后仍需主线程冻结有限同奖励OPD/Direct公平验证，不自动运行。若两项未通过，保留负结果，停止本阶段，不扩大候选/参数/样本搜索。无论结果如何均保留全部128。候选/原始mask/视频/标注留本地，公开plan、汇总、报告及final seal。

## 完整终态解释

本阶段已完成，MaskDice与MaskContrast均未通过，不启动其学生适应。候选排序有93条query可计算（其余部分GT位置的候选IoU完全并列），top1 regret有102条、无GT观测26条不评分；这两个分母不能混写。MaskDice相对几何的regret增加2.5700pp，配对区间[0.7475,4.3963]；竞争分数增加10.9502pp，区间[8.3572,13.6771]。逐帧文本分数AUC0.6512、区间约[0.5772,0.7219]，仅81条query同时有事件内外观测可计算，不是对26条缺失事件query的成功率，也不是可直接部署的事件概率。
