# BoxFixed空间损伤诊断与仅参数128验证

2026-10-22用户要求解释Vid下降、HC几乎无提升，并先128验证；随后明确“只能调整参数，不能改别的”。所有候选构造/传播/评分/特征/时间/模型实现保持原hash。尚未进行数据验证的机制草稿已移出源码至outputs/maintenance/boxfixed_followup_discarded_mechanisms，未执行其数据预测。只保留离线诊断和调用原worker/scorer的参数调度脚本；没有新GPU、教师、学生前向或全量。

## 完整旧结果与数据集差异

完整Vid Full−Temporal −.294599pp，HC +.036692pp、区间跨零。Vid疑问句5693条为−.812273pp，陈述句4610条为+.344690pp。Vid无观察落入GT事件1912条，损失−.035550pp；有观察8391条反而−.353626pp。因此“八观察没覆盖事件”不是总体损伤的唯一或主要解释。HC几乎为929胜、914负的幅度抵消；不是所有query都没变化。所有GT分组仅诊断，不用于推理门控、参数分组或过滤。

VidSTG原论文明确区分显式目标的陈述句与未知目标的疑问句，任务包含关系推理；HC-STVG原作者说明其聚焦人、复杂多人场景。这支持检查跨帧身份/关系歧义，而非把两域当同一分布；但本次没有身份GT，不把分组损失断言成已验证身份切换。当前正式HC口径仍是作者清单1902/227，不能用外部其他v1/v2样本数替代。

- VidSTG原论文：https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Where_Does_It_Exist_Spatio-Temporal_Video_Grounding_for_Multi-Form_Sentences_CVPR_2020_paper.html
- HC-STVG作者数据说明：https://github.com/tzhhhh123/HC-STVG

## TubeDETR接口与已测空间问题

官方TubeDETR使用稀疏慢多模态分支、逐帧快视觉分支及空间时间decoder。当前两个k2/res224检查点均fast=true/pool_size2、ResNet101非dilation；慢特征stride复制后与快分支融合，并非逐帧完全相同。接口核对为典型7×10×256真实空间前缀、文本槽已排除，Source/Native/八观察和教师框逐位一致，未发现offset、坐标或权重漂移。它接受框/时间等监督，不是训练过的像素对应网络，不能把融合token余弦当作已验证对应。

- TubeDETR论文：https://openaccess.thecvf.com/content/CVPR2022/html/Yang_TubeDETR_Spatio-Temporal_Video_Grounding_With_Transformers_CVPR_2022_paper.html
- 官方实现：https://github.com/antoyang/TubeDETR
- 本地固定版本：external/TubeDETR/models/transformer.py的stride复制/fast融合、models/tubedetr.py的损失列表，旧plan绑定upstream_revision。

在同一固定128，空间token非对角平均余弦约.61，相邻同位置约.97。这是共同成分强的描述，不能单独证明错误对应。小于3网格单元的GT帧Vid844/4480、HC212/4643，准入率仅约2.49%/13.68%；它们候选机会少，但实际修改主要发生在较大目标，不能把总体下降全部归因小目标。

旧配置改变的未观察GT节点中，Vid470个、HC537个（含标注终边界，与区间内诊断口径不同），面积收缩比例96.81%/94.04%。平均宽比Full/Source=.711/.823、高比=.803/.875。Vid中心归一化误差由.415降至.304，但尺寸log绝对误差由.265升至.395；HC尺寸误差.143升至.199，中心误差也略升。说明中心对准与框范围正确并不等价；几何损伤不能仅用匹配中心解释。

软对应WLS有一个明确风险：若正确恒等对应混入比例ε的独立均匀对应，协方差回归的尺度为1−ε，即使真实物体未缩小也会收缩。旧已接受拟合平均尺度约.83/.82，与此机制相容，但不是对真实误差成因的完整因果证明。教师框本身、姿态与遮挡也会影响尺度。当前评分的候选自身参考框extent项为1，cycle Dice又不是正确概率；存在自参考偏置风险。同一候选集的GT oracle明显更好，说明除了候选缺失还有选择损失，不能把问题归为DP没求到最优：固定加性评分的DP已精确回归。

诊断文件：results/tubedetr_boxfixed_followup/diagnosis.json、extent_diagnosis.json、old_full_strata.json。节点IoU与完整dense/vIoU范围分别披露；HC节点诊断已对齐原官方下界裁剪，原未对齐草稿保存在私有archive，不参与选参。

## 参数验证：原实现不变

唯一执行函数仍为tools/boxfixed_tube/profiles.py的predict；新脚本只隔离输出/plan目录，调用原tools/tubedetr_boxfixed_development.py的profile_worker与score。旧所有code_sha逐项校验。原配置128预测逐条、逐数组回归后才能比较新参数；原始缓存及旧全量一字节不改。

沿用原Vid/HC各128query/128独立来源，不抽换样本。参数范围事前冻结在parameter_plan.json，每域最多10不同配置：

1. 准入 .4/.7/.9：检验提高原cycle阈值能否减少错误替代，原confidence定义不变。
2. 保护 .4/1/2：原最佳落在旧网格上边界，检验提高原guard权重是否抑制大幅错误框替换。
3. 连接 .5/1：只调整原link项权重，不添加新关联规则。
4. 初始化gamma 0/.0125/.025：包括零初始化控制，其他规则不变。
5. 所选参数下最多两个相邻准入复查，不扩搜索。

原锁定配置一直候选，仍按完整128等query Full m_vIoU最大值选择，严格并列保留原锁；同时报告dense/观察/未观察空间、尾损和改动数量。不是按query或句型选参，不改变选择目标，不通过GT构造推理门控。Source+Global6只作公开对照，不偷偷替换完整方法。

每profile全部128预测封存后，独立CPU scorer才读GT。推理guard在任何校验前安装，只有原始输入及代码hash可在预测阶段读取；注释hash也只在score阶段检查。原生教师/Source缓存精确复用，本轮新增前向0。没有full命令或自动新全量；即使验证有改善也只报告128结果，等待另行用户指令。

此轮假设来自已曝光旧128/全量分析，仍用同128验证，不能称独立确认。参数只能控制现有算法的风险，不能据此声称已修复特征对应能力或尺度估计结构；正负结果均保留。
