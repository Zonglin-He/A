# 三方向 TTA 第二轮：原论文配套条件、结构反馈与新样本验证

2026-10-30，Asia/Shanghai。三个方向均已执行新的开发与确认实验，不是只做方法建议。全部模型/GPU工作已结束；预测先封存，随后独立 CPU 评分。上一轮源代码、计划、预测和旧公开包保留；本轮没有修改旧 STVG 方法或恢复旧暂停实验。

**总判断。** GREAT 获得了确认集 aIoU 的正向信号，但新增局部支持项不成立，其他指标有退步；AVIS 的早期适应和独立音频活动适应均未胜过 source；3EED 的优化约束显著减少了破坏，仍未将几何反馈变成有效的网络参数适应。没有候选被晋升为部署方法。下面区分参数更新、潜变量优化、无更新信息融合和工程修复。

## 数据、样本与测量

AVIS 完整原始档案现已完成：10,473,330,966 字节，SHA256 `fe5ccdffe5756aaf33f44bf107b8b7e5c6359bdfff93ec1dc3f437618d745ec0`，完整 gzip CRC/size 通过；顺序解包 test.json 与此前恢复版本 SHA256 一致。完整测试输入为205视频、12,544帧。GREAT 完整 Unseen-aff 和3EED官方公开档案也已校验并解压；其他GREAT分区及冗余训练状态包没有被混称全部下载完。

| 方向 | 新开发 | 预先锁定的新确认 | 独立性与限制 |
|---|---:|---:|---|
| AVIS | 8视频、480帧 | 12视频、768帧 | 视频级不重叠，均排除旧8；metadata hash选取，确认方法冻结后才推理；完整标注容器早先用于完整性检查，不能声称从未加载其元数据 |
| GREAT | 16 geometry/image组件 | 32 geometry/image组件 | 排除旧12；XYZ规范hash、PCA/近形状和图像复用连接隔离。它不是已验证的真实物理物体ID；确认隔离只针对本次TTA开发，原checkpoint选模来源仍不完全明确 |
| 3EED | 18帧、8场景 | 36帧、15场景 | 开发/确认场景不交叉，排除旧81帧；Waymo新场景，Drone/Quad场景已历史曝光，仅新帧；三平台每平台确认12帧，场景数量不均 |

确认集没有替换样本或逐例GT回滚。GREAT适应缓存仅XYZ/image/CoT；3EED仅XYZ/caption；AVIS不使用目标标注构造对应。准备器会打开包含GT的原始容器，但不把数值GT放入适应数据流；不能将这一点夸称从未读过含标注的文件。3EED仍用union-trained checkpoint和固定纯文本首名词读出，不是source-only跨平台迁移，也不是官方GT类别辅助跨度读出的完整benchmark复现。

## GREAT：任务式损失有aIoU信号，局部支持机制未成立

冻结原模型，保持原点序，只改变FPS起点形成K5视图。新增129参数零初始化残差头，输入64维多模态hidden及kNN局部对比，使用原论文literal focal+双侧Dice作leave-one-view-out软教师拟合；完整机制再加稳定局部支持的单侧logit下界。Adam .01、3步、残差上限 .5，逐样本重置。零头与每个真实CUDA native概率、median读出逐元素相同。

开发阶段按预锁规则选择Q75为主无更新比较，确认前冻结；确认保留mean、median、Q75和HM-only全部对照。四指标有效分母均32。

| 确认32 | AUC % ↑ | aIoU % ↑ | SIM ↑ | MAE ↓ |
|---|---:|---:|---:|---:|
| Source | 75.2785 | 17.6737 | .373384 | .117450 |
| Mean K5，无更新 | 75.3240 | 18.0917 | .381466 | .115134 |
| Median K5，无更新 | 75.2305 | 18.2234 | .380368 | .115034 |
| Q75 K5，预选主对照 | 75.3651 | 18.4687 | .383694 | .117845 |
| HM-only，真实残差参数更新 | 75.1688 | 19.4858 | .377697 | .123024 |
| HM + local support，真实残差参数更新 | 74.8474 | 19.0839 | .379100 | .119544 |

完整机制相对Q75的aIoU为+0.6151pp，21正/8负/3同，父样本描述性配对bootstrap区间[+0.3271,+0.9460]pp；但其余三指标全差。支持项相对HM-only的aIoU为−0.4019pp，7正/22负/3同，区间[−0.8425,−0.0947]pp。因此不采用局部支持项作为核心机制，HM-only保留为有指标权衡的候选证据，不能改写为全面改善。

确认后有限诊断没有新推理或重评新校准预测：median→full的Spearman均值 .98945；shift+温度投影解释full残差能量54.08%，HM-only66.40%，仍有空间残差，不能说全部只是全局偏置。相对Q75，full平均TP+0.78、FP+5.5，目标/非目标MAE均变差，支持覆盖/概率读出权衡，未支持语义排序修复。

根代理亲看固定最大正/负私图：两例均为Kettle/pour，壶嘴GT与壶把假阳性并存。正例相对Q75 aIoU+3.4125pp；负例−.4744pp、TP56→54，AUC94.83→92.56。图像与点云来自不同实例，不作像素投影解释。图片不公开，未用于再调方法。

## AVIS：早期可控性不等于活动信息充分

第一条机制是在9层局部decoder之前加入25,600维逐视频query latent，利用原冻结clip轨迹为frame预测提供class/mask伪监督，并补原训练identity度量。真实source零桥、梯度、后更新读出和episodic reset均通过。反馈不再是对称反转，但同源teacher依然保持了多数错误活动集合。

开发8视频中，计数错误90帧、类别组成错误6帧，约占19阈值失败的61.6%；边界变化无法解决这些阻断。身份伪负对另做无GT共现审计，实际6对均8/8 anchor同时出现，没有为改名重跑。

第二条机制引入冻结LAION CLAP音频语义证据，48kHz物理归一化PCM、3秒非因果窗口、公开26类固定描述。新增逐类逐时activity latent，优化置信加权Bernoulli目标、时间TV和source anchor，直接控制晚输出活动，避免再次被tracker衰减；这属于输出潜变量优化，不是预训练网络权重TTA。

初版相对最强类别的参考仍强加类竞争，开发中误拒真实共现guitar/violin。只移除该竞争项，保留背景参考，模型/prompts/窗口/温度/步数完全不变；作为单因素配套条件修复。开发多类共现误拒由129/216降到0，但背景参考仍会误拒某些机器/交通声音。没有按确认结果继续调阈值。

确认前冻结early6、competitive7、independent7共20个输出配置；source/early在study间逐hash复用，全部20封存后才允许评分，不把重复引用当独立实验。

| 确认12视频、768帧 | 独立frame exact-set % ↑ |
|---|---:|
| Source | 46.2993 |
| 同信息frame/clip直接融合，无更新 | 46.5803 |
| 早期latent + identity | 46.2856 |
| 早期latent去identity | 46.2856 |
| 旧late LN/反转对照 | 46.2925 |
| Competitive audio activity优化 | 17.1875 |
| Background-independent直接音频gate | 44.5792 |
| Background-independent直接平滑gate | 44.5929 |
| Background-independent activity + TV | 44.9630 |
| Background-independent activity，无TV | 44.9630 |
| Background-independent解析source-prior gate，无优化 | 44.8808 |

该独立指标要求逐帧非空实例数量、类别和阈值可行完美匹配同时正确，平均IoU>.05:.05:.95；不是原生FSLA。根独立DFS匹配复核全部结果。早期latent比source少2/14592个正确判断；直接融合多41个，但12视频配对区间跨零。Independent activity比source少195个判断，0视频改善、3退步、9相同；HOTA-area34.2413低于source34.8220，mAP49.29低于50.88。

no-TV目标的静态最优硬gate阈值可由导数算出q*=.5−sqrt(.03)。开发时它与no-TV逐byte相同；确认时有限20步造成12个判断差异，不能把数值未收敛视为新信息。TV与no-TV确认任务结果相同，未验证时间项收益。修复类竞争恢复了大量性能，却仍未超过source，不能将恢复幅度当作TTA有效证据。

无任何AVIS更新方案被采用。类级音频不能独立判定同类实例身份、目标是否可见或精准发声起止；背景语义也不是已校准的目标无声证据。原生FSLA、area-only敏感性及独立exact-set始终分列；子集HOTA macro绝对值不与论文全测试集比较。

## 3EED：约束减少破坏，但反馈支持不足

新教师对source top8候选构造部分可见面/法向多模态似然，将观测点分fit/check两半并独立估法向，保留yaw模π歧义。新1445参数共享query残差头学习center/yaw/semantic-logscore，原模型冻结；直接latent读出为同信息强对照。新增yaw是明确的6D→7D输出扩展，原head自身没有heading。

开发发现共享Adam会改变未通过几何筛选的候选。一次有限重构对teacher覆盖与优化器做2×2：补有实测法向和边界支持的屋顶分支；两优化器均严格mask未接受query，并以输出空间步长/总位移和无标签目标回溯约束新solver。原无mask坏例保留。屋顶分支在dev18和confirm36均没有新增合格候选，结论是配套资格未在当前数据成立，不是屋顶方法理论被推翻；没有降低门槛强造成功。

Primary在确认前固定为roof_constrained3。V1/V2全部确认预测封存、216个共用baseline数组逐byte核对后才评分。

| 确认36帧 | 平均IoU % ↑ | Acc>.25 | Acc>.50 |
|---|---:|---:|---:|
| Source | 44.0171 | 25/36 | 18/36 |
| 固定PCA yaw，无网络更新 | 49.2250 | 26/36 | 23/36 |
| 固定surface选择，无网络更新 | 48.8526 | 26/36 | 22/36 |
| 结构teacher直接latent | 43.7624 | 25/36 | 18/36 |
| V1共享未mask Adam，3步 | 38.3234 | 24/36 | 14/36 |
| V2 masked Adam，3步 | 39.0670 | 24/36 | 14/36 |
| 预选输出约束adapter，3步 | 43.6937 | 25/36 | 18/36 |

1步约束结果44.0371%不替换预选3步。优化器修复降低了破坏，仍低于source和直接几何；全部参数适应方案不采纳。PCA再次提供正均值，但保持正负尾且不称网络TTA。

原始数据范围：开发Waymo16.5k–22.8k、Drone5.4k–8.3k、Quad9.7k–13.2k；确认Drone最低约2.3k。小点云的16,384点输入来自replacement，不是新增独立观测；也不是存在几十万高密度原点被偷偷截去。缺少足够可见面、同类错误实例稳定与语义身份未验证，是反馈的主要剩余限制。

评分精度另有独立更正：新scorer曾对float32 yaw直接取cos/sin，与3D交集紧容差不匹配，14/144开发IoU失效。双精度旋转修复后，独立2D多边形×高度和3D凸体交集对54例全部臂一致，最大差约5.39e-8。原预测不变，旧错分/日志/锁留档并明确INVALID。旧round1受影响路径经有限复核已使用float64，不受此新bug影响。

## 原论文关键tricks的裁决

- GREAT：恢复native float64-normalize→float32顺序，实际旧/新source差很小；literal focal+正负Dice及soft-target自梯度确实影响适应。采样起点、邻域索引截断和decoder索引融合分开检查。训练配对loss求和、epoch/crop文字与代码差异记录为当前源码事实，不推断checkpoint一定训练错。
- AVIS：原frame/clip mask-class损失与identity约束被核查并测试；当前checkout similarity权重键不匹配、训练schedule debug值和论文不同，但权重训练版本未知。保留作者VGGish缓存幅度域，不把单改PCM缩放称作有效修复。
- 3EED：MSS固定半径、top256与1024seed区别、GT类别辅助span、Drone/Waymo坐标和真实scale aggregation均核到执行代码；不能将没有quality head的noun概率当正确性。优化尺度、函数空间界与精度问题经控制修复，但它们不能制造可靠语义监督。

## 独立审计、成本与研究决定

GREAT用Mann–Whitney排名公式独立复算AUC及其余指标，dev16/confirm32最大差2.22e-16。AVIS使用独立增广路DFS替代评分侧匹配；3EED使用不同的2D裁剪×高度算法。源零步桥、RNG/reset、冻结buffers和真实参数/latent变化均有单独receipt。失败工程尝试、序列化修复和评分失效链全部保留。

计时只按实际窗口报告，不把cached replay当部署速度：GREAT K5 forward、adapter和离线CoT边界分开；AVIS source/early/CLAP/CPU activity与加载分开；3EED sourceGPU、CPU几何与solver分开。GPU串行，不累加互斥进程峰值，不声称完整部署延迟已经优化。

研究决定：GREAT保留任务损失TTA的aIoU信号，下一步核心应区分覆盖/校准和语义定位，局部支持项被当前消融削弱；AVIS需要经校准且能区分类与实例活动的独立证据，同源轨迹教师与粗类音频都不足；3EED在本次cohort中PCA朝向扩展有正均值和负尾，但保守资格没有提供充分反馈，安全优化也没有额外任务收益。后续新机制尚未执行，不计为本轮成果。本轮不继续网格搜索、不替换确认集、不更改已封存结果。


证据入口：[独立审计索引](audits/INDEPENDENT_AUDIT_INDEX.json)、[AVIS完整结果](avis/README.md)、[GREAT完整结果](affordance/RESULTS.md)、[3EED完整结果](crossplatform/REPORT.md)、[档案完整校验](ACQUISITION_COMPLETED.json)。
