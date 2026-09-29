# J0整合结果：Fast temporal correction + Slow spatial Rank-OPD

本轮最终方法通过并冻结的条件满足：False。组件选择与实现已锁定，但该状态不等于全流效用通过。四臂采用原16曝光来源、六条件、同25%到达位置。时间固定native候选＋UniversalVTG rerank；空间固定1792维持久参数＋9个antithetic probes＋rank teacher＋reverse-KL一步SGD.005。当前空间输出始终为更新前central policy，更新只用于之后样本。

## 四臂全流主表

单位%，corruption先五条件源内平均再16source macro。

| Arm | sIoU | tIoU | vIoU |
|---|---:|---:|---:|
| Frozen | 45.350867 | 35.040795 | 15.845411 |
| Fast-only | 45.350867 | 35.374475 | 15.773407 |
| Slow-only | 45.422650 | 35.040795 | 15.881107 |
| Final | 45.422650 | 35.374475 | 15.809093 |

## 两个主要比较

- 当前expert的Fast-only−Frozen：ΔtIoU +1.334720 [-1.314152, +5.712840]pp；ΔvIoU -0.288018 [-0.621280, +0.001670]pp。专家子集只有4来源/20corruption cell。
- 未来nonexpert的Final−Fast-only：ΔsIoU +0.078608 [+0.022241, +0.143997]pp；ΔvIoU +0.047181 [+0.007487, +0.095654]pp。12来源/60corruption cell。
- 全流Final−Frozen：ΔvIoU -0.036318 [-0.164640, +0.056689]pp。16来源/80corruption cell。
- 全流Final−Fast-only：ΔvIoU +0.035686 [+0.005630, +0.073389]pp。
- Clean future Final−Fast：Δs +0.084098 [+0.026113, +0.152190]pp、Δv +0.051306 [+0.010724, +0.101671]pp。

区间为固定轨迹下的source bootstrap95% CI；不同corruption不作为独立来源。Clean收益相近，仍属于一般修正证据，不能说已独立验证corruption专项鲁棒性。

## 集成确实按online语义工作

新Final完整跑96到达/18更新。192个pre/post状态、32256梯度坐标与旧Slow-only逐项精确一致；当前temporal候选从已适应参数产生，0/24组与Frozen候选不同，最终选择区间0/24与Fast-only不同。所有96个当前空间输出均使用pre-update，18个更新后的空间预测保留作审计、不进入当前Final成绩。4次学习后完整回插＋2次全部六层temporal回插、384双metric、3CPU及8550公开scalar检查通过。

这里有一个必须明确的比较关系：Fast只在expert到达修改输出，不改后续参数，所以nonexpert上Final等于Slow-only，Fast-only等于Frozen。J0的future增量因此精确复现S1.1；这轮新增的是组合实现正确性和全流效果，不是独立新来源上的再次确认，也不是强协同效应。

## 调用预算与资源

同25%availability：Frozen 0；Fast-only 24次时间expert；Slow-only 24次空间expert；Final 24次时间＋24次空间。总specialist调用数不相等。Final−Fast隔离的是额外历史空间反馈经参数传递的效益，不是相同总专家成本下的比较。本轮全部复用像素/query一致的冻结expert缓存，新expert调用0。

GPU进程累计59.100831秒，包含首个工程校验失败尝试；216新spatial候选/18反向、0新encoder capture。首次因CPU/GPU预测直接比较而报错，0到达保存/0有效更新；原日志与runner保留，仅修校验设备并追加版本记录，方法和评分不变。

## 决定

按本轮预定结果分支，研究方法冻结状态为False。保留非对称设计：**Select temporally, probe spatially, and distill ranked on-policy evidence.** 不再做Norm、temporal OPD、parameter-space OPD、LR/temperature/multiple-step或candidate扩展。正式研究recipe单独保存配置和代码hash，历史生产CURRENT_METHOD不被静默替换。

之后工作进入固定方法评价阶段：更大/未曝光来源、same-domain corruption主setting、clean对照、cross-domain辅助及online stream/order稳定性。本轮没有启动这些额外评估，也没有把这批已曝光短流当作fresh结果。历史O2等也有弱online信号，不能把J0写成所有online实验的首次正迁移。

## 为什么旧时间正结果没有出现在J0全流

只读回查同16来源、同五种5%corruption的旧C3缓存成绩：若每次到达都允许时间专家，Rerank−Frozen为Δt+6.130562/Δv+3.156181pp。按J0固定schedule分组，实际可用的4来源为Δt+1.334720/Δv−0.288018pp；不可用的12来源旧参考为Δt+7.729176/Δv+4.304248pp。四来源的负均值乘25%得到J0 Fast全流Δv−0.072005pp。它解释了与历史全专家结果的差异，不能据此重选更有利的schedule。

具体保留四个来源：Q01的Δt+7.965456pp但Δv0；Q05 Δv−0.828373pp；Q09 −0.327039pp；Q13 +0.003339pp。可用来源的candidate vIoU oracle仍有+3.585205pp，但这是离线GT上限，不能拿来替代实际critic选择。当前负结果不能唯一归因schedule运气，也不能普遍否定Temporal critic。

**本轮结论：future transfer仍为正，然而corruption全流Final−Frozen为−0.036318pp，CI[−0.164640,+0.056689]，没有达到用户预设的正向条件。** 不能宣布J0成功或最终方法效果已验证。按用户意图停止机制救援，封存当前非对称组件与整合代码；没有扩大专家覆盖率、按GT更换schedule、删除负例或追加模块。是否在更大/不同独立流上确认稀疏Fast效应属于之后的固定方法评价范围，本轮未自行启动。
