# 已封存最佳配置的深入 GT pipeline 诊断

2026-10-02。依据用户明确授权，在已完成并公开的 `tastvg_best_quick_v1` 上分析 GT、原始候选/奖励/梯度与逐步日志，并阅读原论文和官方代码。本次模型 forward、参数更新、新预测、GPU 初始化均为 **0**；没有修改 frozen 或在线方法，没有重选参数，没有恢复保存暂停的全 query 队列。

## 结论和范围

当前瓶颈至少包含 **证据与计分支持错配、排名丢失偏好强度、偏好到参数更新的代理目标错配、时间 critic 定位质量失准，以及跨 query 状态的正负迁移抵消**。不能用“专家不行”或“再调小学习率”概括。Vid 的已验证正收益应保留；HC 尚未建立稳定收益，不能将其负尾归为单一原因。

基础结果为 Vid 670 来源/一 query/两序/六条件 8040 到达，HC-STVG-v2 128 来源/一 query/一序/六条件 768 到达；官方各自同域 TA-STVG checkpoint、原 Paper48 采样、clean 与五类 5% transient physical corruption、25% 专家，全部来源有历史项目曝光。Vid 封存配置 lr=.033761698432507946、teacher temperature=.34902548789596055、K1；HC lr=.006097133675874025、teacher temperature=1、K8；rho=.05、student temperature=1、D4、1792 参数固定。

corruption 来源等权 dense vIoU：Vid Frozen21.0217%→Ours22.3684%，+1.3468pp、95%来源配对bootstrap CI[+.9485,+1.7582]；HC29.1984%→29.4118%，+.2134pp、CI[−1.0299,+1.4268]。outside-tuning 来源590/48的差值分别+1.4135/−.6474pp；仍有历史曝光，不称 fresh。这些是完成的快速面板结果，不是全 query 结果。本轮 GT 分析只用于事后解释，不进入线上选择。

当前输出先于本次空间更新封存；时间 rerank 可以影响当前输出，空间更新的当前 post 预测仅作诊断，正式效用主要由后续 query 继承状态体现。下文单次更新的 fixed-time GT 差值不能当作 online 最终收益。

## 1. 稀疏空间参考帧与 GT 计分事件不同

实际 Sa2VA 接口在整个观察 clip 上均匀取五个位置，仅在这些位置保存 mask-to-box 与 valid。空间 reward 是候选框和这些 valid 参考框的平均 IoU，没有限制在预测的事件区间；几何 student 距离则平均整个采样轨迹。GT 指标计分的是 query 对应事件的 dense tube。

下表仅 corruption 的专家到达。参考框质量只在有 GT 空间标注重叠的采样帧上计算，不能当作整条专家 tube 的指标。

| 测量 | Vid | HC2 |
|---|---:|---:|
| 专家到达 | 1680 | 160 |
| 空参考 | 211 | 20 |
| 非空但没有任何参考帧落在 GT 计分支持上 | 400 | 15 |
| 至少一参考帧与 GT 重叠 | 1069 | 125 |
| 这些到达的参考框 GT-IoU 中位数 | .7259 | .7226 |
| 第一更新中上述无 GT 参考重叠：受损 / 非空到达 | 94/400 | 3/15 |

Vid 非空参考的 400/1469（27.23%）没有 GT 计分帧重叠。能看到正确对象，不等于在正确事件上监督；但事件外参考也可能学到可迁移的对象几何，不能把400次全部判错。其当前更新平均Δv只有−.0589pp，HC对应+.0259pp；这些是按事后GT分组的局部描述，不是筛掉后重跑的结果。

参考质量也确实有作用：Vid 第一步参考GT-IoU<.5的390次平均当前Δv−.5650pp，>.7的587次为+1.0361pp；HC对应35/66次为−.1606/+.0199pp。它们存在 query、难度、时间支持等混杂；这些GT阈值不得用作线上 gate。

较直接的待测改动是让空间比较的时间支持与无GT的事件证据对齐。可以从 native/temporal候选或可靠专家时间证据得到权重；native 时间自己可能错误，单独裁到 native 区间可能强化错误，不能预先认定更好。不必先换 backbone 或新增重型专家。

## 2. rank teacher 把弱偏好转为高置信度

实际映射为：

\[
q_i=\operatorname{softmax}(-\operatorname{rank}(r_i)/T_E),\qquad
p_i=\operatorname{softmax}(-d(B_\theta,\operatorname{sg}(B_i))/T_S).
\]

候选9个，rank按降序、平均处理数值差≤1e−12的tie。只要排序不变，reward差距几乎消失或很大，q都可以一样。第一步非空更新的 corruption 分布：

| 测量（中位数） | Vid | HC2 |
|---|---:|---:|
| max reward−min reward | .01443 | .01704 |
| 第一名−第二名 reward | .001815 | .001756 |
| q 最大概率 | .9430 | .6322 |

reward是IoU量纲，但排名生成的q并不是“专家正确概率”。Vid的低温度将中位约.0018的第一、二名差压成约94.3%的top质量。当前量级下确实发生强弱不分；不过小差距也可能方向正确，不能仅据此说所有弱偏好无用。

完整偏好也有误排序：GT有差别且专家reward非tie的36对统计中，Vid排序一致率66.33%（26652正确/13529错误），HC61.02%（2746/1754）；另有专家tie1690/180对。重复condition/order共享来源，候选对也相关，这不是独立样本准确率或置信区间。

更合理的候选改动是**保留排名方向，保留证据强弱**：由真实reward差距或无GT的一致性控制loss权重/teacher温度。仅降低全局lr不能区分可靠与不可靠到达；单纯加温度也不能恢复被rank丢掉的绝对差距。是否有收益需要匹配对照，不能用本次GT挑threshold。

## 3. 无偏好时，当前 loss 仍然更新

所有9个reward完全相同时，q=Uniform，而当前目标为

\[
\mathrm{KL}(p\Vert U)=\log9-H(p).
\]

最小化它是在增大几何分布熵，并不是学习专家区分。这是代码和公式直接成立的事实。Vid corruption109次flat全部有非零梯度，其中20次当前GT受损；HC第一步5次、K8合计40次flat均非零梯度，但本轮40次没有GT受损。flat的平均当前Δv为Vid+.00648pp、HC+.0000146pp；**不能据此声称跳过flat已证明改善，更不能将HC失败归因于flat**。

明确“专家无可区分偏好时no-op”可以修正学习信号语义，但它只是小范围控制，不能替代弱偏好和后续迁移诊断。

## 4. 正确偏好没有被梯度转成正确 tube

第一步检查：专家**唯一第一名**的候选GT-vIoU高于当前native，但实际更新后的fixed-time GT-vIoU下降，Vid **159次**、HC **11次**。原宽口径top集合含更好成员是179/11；Vid其中20次flat并非专家的明确偏好，故主结论采用159/11。

匿名Vid源70、occlusion_5、order2、arrival272：

| 项 | 值 |
|---|---:|
| 更新前固定时间GT-vIoU | .44727 |
| 专家唯一第一名、同时oracle候选的GT-vIoU | .46489 |
| 实际梯度更新后的GT-vIoU | .21614 |
| KL loss前→后 | 9.40289→8.92701 |
| 全局梯度L2 / 实际单步参数L2 | 27.0221 / .91231 |

这里候选有改善空间、专家选到了该候选、loss也下降；失败发生在代理目标和参数执行层。当前p是对整条框轨迹的L1+GIoU距离softmax，**不是TA-STVG原生生成某条tube的概率**。每次第一步 native 都是p的最大项（Vid1469/1469、HC140/140），这是几何构造的性质，不能当作模型语义置信度。目标改善会通过整个decoder、LayerNorm和共享query residual作用于其他时刻及后续query，不保证逼近最高reward候选的实际GT效用。

rho=.05规定的是固定探测方向的candidate半径，Vid/HC绝对L2半径1.57658/1.56435；实际参数更新是普通SGD `theta -= lr * gradient`，没有candidate-radius投影或累计状态trust region。不能把rho误称“更新最多5%”。Vid记录最大单步L2为2.18659，已经超过其探测半径；这不独自证明过冲导致所有负例，但说明梯度可以走出被专家评分过的支持。

反例也保留：Vid匿名源240、frame_drop_5、order1、arrival560，当前GT-vIoU .19262→.37467，空间更新有明显正例，正式继承空间净收益+1.2481pp。不能因负例把R-OPD路线整体否定。

## 5. 时间 critic 需要定位质量，raw confidence不等价

实际评分不是加和，而是

\[
C(c_i)=\max_j[\operatorname{tIoU}(c_i,u_j)\,s_j],\quad\hat i=\arg\max_iC(c_i).
\]

空proposal或全tie时native-first回退；其余直接替换区间。一个错误但高confidence的proposal就能支配max；存在好proposal不保证其贡献最大。实际UniversalVTG代码以分类head的sigmoid产生raw_scores，训练区分binarypoint labels并单独回归边界；本接口没有另作“confidence=GT定位IoU”的校准。

| corruption、joint vIoU>阈值 | 有正确候选 / 漏选 | 原正确被破坏 | 有good teacher proposal（tIoU>.5）却由bad proposal支配的破坏 |
|---|---:|---:|---:|
| Vid、.3 | 698 /159 | 57 | 50/57 |
| Vid、.5 | 403 /104 | 33 | 20/33 |
| HC、.3 | 77 /4 | 0 | 0 |
| HC、.5 | 35 /14 | 8 | 3/8 |

匿名Vid源63、occlusion_5、order2、arrival300：当前vIoU .55562、现有oracle候选 .61212，但critic选中 .01820。专家最佳proposal tIoU .87352，实际支配选择的proposal tIoU只有.02607。critic native score .28820、选中项.30904。另一失败源594的score差约.0905，仍把vIoU .67461变成.24338，故单靠“小score margin就拒绝”不能解决全部误选。

时间候选必须与空间一起解释：有tIoU>.5候选却所有joint-vIoU≤.3，Vid311/1680、HC38/160。时间支持足够但框质量仍可限制joint指标；不能把“无正确joint候选”全归于temporal候选生成。

可借鉴的方向是定位质量校准与native保留/替换证据，而不是直接假设多加proposal或把max改sum会改善。sum可能放大重复错误proposal，过强native prior也可能失去已观测纠正。若学习IoU-quality critic，需要另一个合法训练集；不可在当前test GT上训练后称无监督TTA。

## 6. HC 更突出的是正负迁移抵消，而非已经证明长流崩溃

继承空间框的corruption gross gain/loss：Vid1.8825/.6344pp，HC1.2886/1.2720pp；HC净效用只有+.0166pp。HC当前完整K8更新的全到达净fixed-time空间效用+.0465pp，CI[−.0129,+.1126]；nonexpert到达平均Δv−.0659pp也CI跨零。

当前1792参数是共享三组LayerNorm与一个256维query residual，跨query持久，没有参数anchor、replay memory或恢复机制。局部loss降低不等于对下一query有用。不过按arrival四分位，HC继承空间平均Δv为−1.7838、+.8801、+.5725、+.3976pp，**没有单调越跑越差**；固定顺序中的不同来源难度也混杂。因此这里只支持正负迁移和代理目标风险，不足以宣称“灾难性漂移已被证明”，也不能把两数据集差异单归于K8。

## 原论文与官方实现如何处理相关问题

以下是实际查阅的一手材料。它们的成功场景与当前STVG不同；迁移到本方法的内容属于推断/待测方案。

| 文献 | 原方法的处理 | 对当前管线的启发和边界 |
|---|---|---|
| [IoU-Net, ECCV2018](https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Borui_Jiang_Acquisition_of_Localization_ECCV_2018_paper.php) | 训练预测定位IoU，用定位质量决定排序/抑制与框refinement | 分类或匹配confidence不能替代定位质量。原方法用训练GT，不能拿本轮test GT直接建critic。 |
| [EATA, ICML2022](https://proceedings.mlr.press/v162/niu22a.html)；[官方eata.py](https://github.com/mr-eggplant/EATA/blob/main/eata.py) | 过滤不可靠/冗余样本，熵损失加权；Fisher加权参数约束减少遗忘 | 借鉴“信息决定是否更新/更新多强”和initial anchor。其softmax类别熵不是我们的几何p，不能照搬阈值；Fisher需要额外准备。 |
| [SAR, ICLR2023](https://arxiv.org/abs/2302.12400)；[官方sar.py](https://github.com/mr-eggplant/SAR/blob/main/sar.py) | 两次可靠性过滤、sharpness-aware更新，监控与恢复模型；代码支持LN/GN | LN可更新不代表稳定。参考更新稳定性诊断，但SAM需额外反向，现无证据必须引入。 |
| [CoTTA, CVPR2022](https://arxiv.org/abs/2203.13591)；[官方imagenet/cotta.py](https://github.com/qinenergy/cotta/blob/main/imagenet/cotta.py) | EMA teacher与augmentation平均降低伪标噪声，随机恢复初始参数缓解累积遗忘 | 说明跨样本状态需要约束。参考轻量anchor/recovery；本方法独立Sa2VA/UniversalVTG并非CoTTA的学生EMA teacher，不能混称。 |
| [GKD, ICLR2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5be69a584901a26c521c2b51e40a4c20-Abstract-Conference.html) | 学生自生成序列上，教师提供同prefix的token分布反馈，比较原生teacher/student概率；不同任务适合不同divergence | 支持student-proposes的思想，不证明当前geometry-softmax就是native policy probability，也没有保证reverse-KL普遍更好。 |
| [Reward Model Overoptimization, ICML2023](https://proceedings.mlr.press/v202/gao23h.html) | 用gold/proxy reward的实验区分代理指标提升与真实目标下降，并研究优化强度 | 可用来解释为何loss下降不等于GT提升；原工作是合成RLHF环境，不能把其曲线当作STVG定律，也不支持“加KL约束必定解决”。 |

当前实际时间专家是 [UniversalVTG](https://arxiv.org/abs/2604.08522)，不能用名字相似的UniVTG当作同一个teacher。PE-Core-L14-336视觉/文本encoder已在本地实际worker核验；此前缓存说明误写为CLIP，本轮已更正，不涉及teacher变更或重跑。

## 最小下一步，尚未执行

优先处理 **reward→学习信号**：保留当前native候选、R-OPD、各自封存lr/K与在线pre-update读出，只改变偏好强度的表达，做到flat不作专家更新、弱证据不过度尖锐。不能简单删除全部低margin样本，它们也有正例；需要保留这些成功情况。使用原调参开发来源上的小规模匹配对照，不使用本次全快速面板GT选阈值。若采用loss权重，还应与相同平均梯度尺度的统一缩放对照，区分可靠性信息和有效lr下降。

第二项才是单独处理时间critic的quality/native保留；第三项在HC用online/episodic或单独anchor对照分开状态迁移与局部更新。不同时加入SAM、memory、多个expert或复杂网络，也不凭这些诊断先跑全量。旧raw-reward S1结果仍保留：本候选保留rank方向、只补强弱信息，不把历史失败的原始score版换名当新成功。

本报告提供原因定位与待测方案，**不报告任何新方法收益**。GT可指出错误，不可用来线上拒绝样本、挑最优step或修改已封存结果。新方法全量前，应保持输入/专家缓存合同，重新生成依赖新适应历史的候选、奖励、梯度、状态和输出。

## 可检查证据

CPU分析目录 `artifacts/tastvg_quick_deep_diagnosis_cpu_v1`：两组全部匿名DEEP_STEPS、REFERENCE_ROWS、TEMPORAL_ROWS、SUMMARY、SUPPLEMENTARY、正负CASES与输入哈希/CPU完成凭据。分析实际回读2016/192份scheduled封存payload与匹配expert cache，基础root已核对所有8808预测和状态链；额外匿名标量重建5843/1776项通过。额外标量重建不等于独立再读GT。

源码：`scripts/deep_analyze_tastvg_quick_cpu_v1.py` 与 `scripts/summarize_tastvg_quick_deep_cpu_v1.py`。计算参考IoU采用GT计分支持内的cxcywh框；保留官方HC端点合同，不事后替换评价器。quantile顺序min/25%/median/75%/max。SUMMARY的flat `net_GT_delta_pp` 是逐步Δ的 **100倍总和**，不是全组均值；正确flat均值见SUPPLEMENTARY。所有GT条件分组、案例选择都是事后描述，重复来源/顺序/候选对不作为独立统计样本。公开材料不含媒体、caption/source真实身份、GT坐标、权重或原始状态。
