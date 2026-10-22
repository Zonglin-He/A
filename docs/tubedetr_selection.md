# 固定候选的评分与准入：两域128验证

用户2026-10-22要求继续定位候选选择问题，小改评分/准入，先验证原128来源，时间分支不动，不跑全量。本轮仅此范围；原模型、教师、观察、候选坐标、特征、对应、时间和旧结果保持不变。

## 两阶段证据

先冻结qualification_plan.json，用上轮固定候选配置精确重放并捕获两个真实评分bank（Native/Global6）。Vid沿用relative_residual alpha1/beta.5、gamma0、准入.4/保护2/连接1；HC沿用原候选gamma.025、准入.7/保护.4/连接1。只观察原执行的bank对象，不改源码和执行数值；128条两种时间输出与旧封存数组逐位一致。bank完整seal后才独立CPU读GT作诊断。随后冻结plan.json再执行新评分profile；诊断可用于形成开发假设，但不是未曝光确认。

诊断范围是冻结Global6预测区间内、有GT标注的实际物理节点。它不是等query dense指标，也不代表所有GT事件帧。Vid原选择的292个受损未观察节点中，132个自身参考优势为正、另一侧优势非正；478个受益节点中也有89个满足这个模式，说明排除自参考既可能去掉损伤，也会去掉收益。受损节点原cycle均值约.626，高cycle不等于定位正确。HC现有高准入下，选中的未观察标注节点仅10个且均改善；全部可用未观察候选574个中370个反而更差，不能靠扩大准入保证改善。完整分解见diagnosis.json。

## 小改动

候选集合及坐标完全固定。对由前锚点产生的候选，只使用后锚点的原extent/association评分作交叉验证；后锚点候选反之。比较对象始终是在同一参考下的候选与ERef分数差：

`A(t,k) = .5*(extent(candidate,other)-extent(ERef,other)) + .5*(association(candidate,other)-association(ERef,other))`。

这里的“独立”只表示锚点来源不同，同一教师两帧错误仍可能相关，不能当统计独立或身份真值。两侧都必须有可用原候选、不同来源且原cycle的较小值至少.1；原来的候选准入集合不能扩大。只有A严格高于margin才允许该候选参与DP。

观察位置的两个参考来自同一教师观察，不能互相佐证。因此新模式保留该位置ERef，不用自身教师框证明自身应被替换。教师仍是全部传播证据来源。单独observe_protection控制只关闭这些观察候选，用来区分收益是否来自取消观察更新；它并不证明所有教师观察都错误。HC的ERef仍含原gamma.025初始化，不把它冒充完全Source。

- cross_gate：只做上述准入，原评分不变。
- cross_score：未观察unary改为 `.5*min_cycle*A/N_unobserved - guard*G/T`；连接项、分母、精确DP保持原样。
- cross_agreement：在cross_score上增加两个已有候选的对称extent一致性门槛；不是新增候选或语义识别。

每域最多12配置：原控制、观察保护控制；cross_score/cross_gate分别margin0/.02/.05；所选cross_score的一致性.5/.8；所选可调交叉评分最后仅guard .4/1/2。若最优是原/观察保护/只准入控制，不执行无作用的guard更改。不扩网格、不换query、不修改时间。

## 选择与审计

可行配置须整体dense及未观察sIoU均高于Temporal，m_vIoU不低于Temporal及固定候选原评分；其中最大化等query整体dense。另预先登记整体和未观察均至少+.1pp为实用增量目标，这只是工程标准，不是统计显著性。选择之后报告父bootstrap区间和胜负/尾损，不能将同一开发集上的选择后区间当独立确认。

每profile128完整预测seal后独立CPU评分；推理先装文件访问guard，连标签hash也不读取。原控制输出逐数组核验；每条新预测必须来自同一封存候选bank，Native/Global6区间、Source、帧ID和八位置均逐位不变。既有mean_admitted_frames字段仅描述原候选bank可用节点，实际新准入通过独立审核另行给出，不能混用。

实现位于tools/boxfixed_followup/selection*.py，结果在results/tubedetr_selection，私有bank/预测在outputs/tubedetr_selection。只有CPU缓存评分和DP，没有新教师、学生前向、参数训练、全量或自动续跑。
