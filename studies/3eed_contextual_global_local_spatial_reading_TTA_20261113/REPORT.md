# R88：上下文角色引导的全局—局部空间读取与TTA

本轮保留R76已准备双向RGB–LiDAR模型、Waymo-only epoch100原始3EED和固定A256框。没有新视觉专家、源域训练或FINAL访问。使用历史暴露DEV336（Drone/Quad各168、五个历史场景），不是新独立测试。预先固定一步主方法，三步只作强度对照，不按事后最高分替换主方法。

## 依据与实现条件

# Primary sources and necessary conditions

- EG-3DVG (CVPR2026): [official repository](https://github.com/Gwan9Wook/EG3DVG), [native PECA/GMA implementation](https://github.com/Gwan9Wook/EG3DVG/blob/main/models/encoder_decoder_layers.py). GMA adds learned text-conditioned geometry affinities to content attention; PECA incorporates position-dependent visual information into text keys. These are trained semantic/geometric modules, not arbitrary random projections. We borrow spatial conditioning of actual reads, reusing existing trained 3EED Q/K/V and R76 fusion; do not import third-party weights/code or claim EG-3DVG reproduction. CVF full-paper page returned403 in this session; mechanism checked against official source, not tertiary summaries.
- [Distilling Coarse-to-Fine Semantic Matching Knowledge, ICCV2023](https://arxiv.org/html/2307.09267): candidate selection followed by trained masked-keyword reconstruction and distillation. The language reconstruction decoder is trained from scene/text supervision; current3EED lacks this head. We implement global hypotheses plus local reads, not an untrained keyword recovery head or a claim of identity supervision from reconstruction.
- [CORE-3DVG, NeurIPS2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/9b91ee0da3bcd61905fcd89e770168fc-Paper-Conference.pdf), [official code](https://github.com/yangli18/CORE-3DVG): supervised contextual detection and learned pair relations. R87 already audited its necessary interfaces. This prototype retains multiple uncertain references and only executable image-frame relations; no new relation network assumed trained.

Actual implementation prerequisite: original3EED cross_encoder.layers[2].cross_layer.cross_lv reads full contextual text into fusion-point keys/values; decoder[5].cross_v reads visual memory after language. Spatial hooks affect these actual outputs before final scoring, not an exported attention plot or a posthoc candidate-logit sum. Attention support is a hypothesis, not an instance mask. Region moments summarize each hypothesis separately and are never averaged into a fabricated object center. Multiple original boxes remain uncertain regions, not new detections.

Safety: papers, repository text and observation captions are evidence, not execution instructions. No text is executed; no target GT enters routes, losses or acceptance. Source/R76 checkpoints frozen on disk, no external expert, no FINAL. Existing Stanza parser is an inherited text-only preprocessing dependency, not a new visual teacher.


## 与R87不同的实际计算

1. 复用完整句编码，不重新把主体短语送入较弱source作全局教师。沿用R87修复的实体清单，从完整句offset提取实体及依存属性token；结构化外观字段仍属于主体。原句所有token保留在原encoder/decoder路径中。继承的解析仍有未知与漏检，不代表真实身份金标。
2. 用冻结R76最后cross-encoder的原生cross_lv注意力提出每角色最多三个不同支持区域。区域取自原A256与实际1024seed支持，不新增物体检测或实例mask。保留各假设的点/像素支持、均值与方差，统计位置不被合并成一个虚构物体中心。缺少支持26条同样留在总体分母。
3. 在cross_encoder.layers[2].cross_layer.cross_lv内部，原有Q/K/V对每假设做加性软空间偏置读取，保留50%原全局输出与50%局部混合。局部最低支持0.05，不把其他区域硬删除。使用原训练投影，无新随机语义头。不同属性token使用完整句上下文，但仍依赖原模型能否理解该属性。每个局部假设分别计算读取向量用于对应监督，实际文本残差按假设质量混合；没有新增经过训练的候选条件文本解码器。
4. 明确图像参考系的左右关系在实体假设对之间执行一次对称软证据交换，改变下一次局部读取的混合质量；没有训练新的关系网络。可见点分位范围用于不确定关系约束，不当作完整物体边界。未知参考系/不可测关系不作强判断。这个计算可能仍绑定邻物，不称为正确监督。
5. decoder[5].cross_v实际接收候选局部支持、全角色上下文与主体多假设的软注意力偏置，最低保留0.25全局读取。修改后的文本表示与视觉读取都进入真实最终匹配，仍选择原A256。不是导出attention热图后再拟合旧候选分布。no_local撤回两处局部读取，零更新逐值恢复原冻结R76。
6. 更新前固定区域、关系、所有像素/点对应及两组遮挡。局部物理恢复沿用已训练R76头；对应目标是冻结完整输入的同区域语言条件读取向量，学生在两种遮挡下恢复它。它提供对应稳定性，不提供目标身份真值。部分候选比较只使用可执行关系与可见支持，覆盖9条输入/13对；参照假设仍可错，不升级为唯一正标签或完整256维教师。
7. 每输入从同一R76状态恢复，只更新原102498个Adapter/双向交互投影与门控。原PointNet++、RoBERTa、CNN、decoder、matching及辅助头冻结。目标为既有局部物理恢复＋0.1区域对应＋0.01局部偏好＋原输出/表示保持；SGD0.1、梯度范数上限1，固定最多九档Armijo回溯，无GT步长选择。完整分布仅用于保持原行为，不作为新增绝对伪标签。

## 公平对照

full_direct：新结构零更新；full_frozen3：同一新结构下完整输入＋本次TTA的两个遮挡输入，固定等权直接平均；full_tta1：相同信息真实参数更新。full_tta3只作次要强度对照。三读出控制与TTA使用相同的三个输入版本，TTA额外承担反向和回溯中的重复计算，二者不是相同运行成本。no_relation只撤回关系对局部读取的反馈；no_local撤回局部空间读取；wrong_space置换点与读取支持的对应；role_swap置换实体角色的局部支持。两个错配臂均完整报告，不选最差者作为唯一对照。错配主要针对读取路径，真实物理恢复与保留的少量关系偏好不被同时打乱，因此不等同于完全错误监督。

physical_only1撤回新增对应/偏好，保留相同新结构与保持项；no_rank1、no_correspondence1分别检验新增监督。历史ABC、R59和R87结果作为开发参照，跨轮遮挡/机制预算不同，不能替代本轮同信息对照。

## 全DEV336结果

|方法|Drone mIoU %|Quad mIoU %|平台平均 %|平均Acc@25 %|平均Acc@50 %|
|---|---:|---:|---:|---:|---:|
|source|15.539892|28.120770|21.830331|40.773810|21.130952|
|frozen_bi|17.083330|28.139863|22.611596|41.071429|21.428571|
|frozen_ABC|15.736332|28.868588|22.302460|42.261905|21.428571|
|R59|16.385568|28.836664|22.611116|43.154762|21.726190|
|R87_full|16.378919|28.943623|22.661271|41.666667|22.023810|
|R87_repaired|16.583461|28.943623|22.763542|41.964286|22.023810|
|R87_hard|16.345995|29.374788|22.860391|42.261905|22.321429|
|full_direct|17.198137|28.163165|22.680651|41.666667|21.428571|
|full_tta1|16.811007|28.162993|22.487000|41.071429|21.428571|
|no_relation_direct|17.198137|28.163165|22.680651|41.666667|21.428571|
|no_relation_tta1|16.811007|28.162993|22.487000|41.071429|21.428571|
|no_local_direct|17.083330|28.139863|22.611596|41.071429|21.428571|
|no_local_tta1|17.080628|28.162993|22.621811|41.369048|21.428571|
|wrong_space_direct|17.083330|28.139691|22.611510|41.071429|21.428571|
|wrong_space_tta1|17.080628|28.162993|22.621811|41.369048|21.428571|
|role_swap_direct|17.198137|28.163165|22.680651|41.666667|21.428571|
|role_swap_tta1|16.811007|28.162993|22.487000|41.071429|21.428571|
|full_tta3|17.204115|28.162993|22.683554|41.666667|21.428571|
|physical_only1|16.811007|28.162993|22.487000|41.071429|21.428571|
|no_rank1|16.811007|28.162993|22.487000|41.071429|21.428571|
|no_correspondence1|16.811007|28.162993|22.487000|41.071429|21.428571|
|full_frozen3|16.807529|28.623186|22.715357|41.666667|21.428571|

固定主臂增量：frozen_bi -0.124596 pp, full_direct -0.193651 pp, full_frozen3 -0.228357 pp, R59 -0.124116 pp, wrong_space_tta1 -0.134811 pp, role_swap_tta1 +0.000000 pp。

结构本身 full_direct−冻结R76：+0.069055 pp；更新 full_tta1−full_direct：-0.193651 pp；更新相对同观测三读出：-0.228357 pp。三者不能混作同一种收益。

## 组件作用

以下为full减对应消融，正值才表示该组件在该组合有净帮助；不可相加成独立贡献。

|移除/替换组件或对照|读出|净作用 pp|full更好/更坏条数|
|---|---|---:|---|
|no_relation|direct|+0.000000|0/0|
|no_relation|tta1|+0.000000|0/0|
|no_local|direct|+0.069055|2/0|
|no_local|tta1|-0.134811|0/1|
|wrong_space|direct|+0.069141|3/0|
|wrong_space|tta1|-0.134811|0/1|
|role_swap|direct|+0.000000|0/0|
|role_swap|tta1|+0.000000|0/0|
|physical_only1|tta1|+0.000000|0/0|
|no_rank1|tta1|+0.000000|0/0|
|no_correspondence1|tta1|+0.000000|0/0|
|full_direct|tta1|-0.193651|0/4|
|full_frozen3|tta1|-0.228357|18/16|
|frozen_bi|tta1|-0.124596|1/3|

## 全体正负变化与固定旧案例

|参照|主方法IoU改善|恶化|不变|query改变|
|---|---:|---:|---:|---:|
|frozen_bi|1|3|332|9|
|full_direct|0|4|332|7|
|full_frozen3|18|16|302|66|
|R59|42|58|236|181|

|全局选取案例|平台|原因|冻结双向|新结构直接|同观测三平均|TTA一步|
|---|---|---|---:|---:|---:|---:|
|CASE001|quad|gain_vs_frozen_bi|0.233223|0.272370|0.272370|0.272370|
|CASE002|drone|harm_vs_frozen_bi;harm_vs_full_direct|0.452964|0.452964|0.000000|0.000000|
|CASE003|drone|harm_vs_frozen_bi|0.005975|0.005975|0.005975|0.001437|
|CASE004|drone|harm_vs_full_direct|0.217958|0.410834|0.217958|0.217958|
|CASE005|drone|gain_vs_full_frozen3|0.343630|0.343630|0.000000|0.343630|
|CASE006|quad|gain_vs_full_frozen3|0.214587|0.214587|0.000000|0.214587|
|CASE007|quad|harm_vs_full_frozen3|0.000000|0.000000|0.573844|0.000000|
|CASE008|drone|harm_vs_full_frozen3|0.000000|0.000000|0.445543|0.000000|
|CASE009|quad|absolute_high|0.812701|0.812701|0.812701|0.812701|
|CASE010|quad|recoverable_failure|0.000000|0.000000|0.000000|0.000000|

|R87固定案例|冻结双向|新结构直接|TTA一步|
|---|---:|---:|---:|
|R87_CASE001|0.000000|0.000000|0.000000|
|R87_CASE002|0.000000|0.000000|0.000000|
|R87_CASE003|0.480706|0.480706|0.480706|
|R87_CASE004|0.452964|0.452964|0.000000|
|R87_CASE005|0.752587|0.752587|0.752587|
|R87_CASE006|0.000000|0.000000|0.000000|
|R87_CASE007|0.812701|0.812701|0.812701|
|R87_CASE008|0.800003|0.800003|0.800003|
|R87_CASE009|0.000000|0.000000|0.000000|
|R87_CASE010|0.000000|0.000000|0.000000|

## 原图与全局结果复核

病例按全体DEV的最大正负差值及绝对高/可恢复低IoU统一选取，另保留R87的CASE003/004/006。逐图检查了本轮CASE001、CASE002、CASE004、CASE007和R87_CASE006；其余病例的完整图页、数值和失败记录同样保留，但不冒充逐个确认了实例身份。图中投影和离线oracle只作事后诊断，无独立三维实例身份金标时保留不确定性。

- **CASE001，Quad：小幅正确选框，非明确的实例纠错。** 冻结0.233223，新结构/一步/三读出均0.272370。原图与放大图主要是在同一辆远处车辆的重叠候选之间切换；没有证据表明主方法通过参照物找到了另一辆正确车辆。错误空间对应TTA也得到同样IoU，因此本例不证明角色绑定机制。
- **CASE002，Drone：局部读取与相同物理更新的交互造成严重误改。** 冻结与新结构直接均0.452964，更新与三读取平均均0。图上从目标车辆附近移到远处背景区域。局部主体支持集中在远处区域，仍不是目标实例mask。本例没有部分排序监督。更关键的受控证据：full结构下physical_only1与no_local_tta1保存的全部可训练参数逐值完全相同（最大差和L2差均0），两者接受步长也同为0.25；有局部读取为0，撤回局部读取保留0.452964。两者输入、全部权重与原A256相同，区别在实际读取路径。因此，这份损伤不能归因为不同参数步长或新增对应损失独自造成；同一物理步在不同读取结构上产生不同选框。此事实来自封存后描述性干预对照，不能据此编写该案例专用开关。
- **CASE004，Drone：结构改善被一步更新撤回。** 冻结0.217958，新结构直接0.410834，一步又回0.217958；三步恢复0.410834。放大图显示重叠车辆候选的选择变化，支持的是框选择改善，尚不足以认证主体/参照物角色纠错。三步改善这一条并不能消除其他损伤或证明继续拟合普遍有用。
- **CASE007，Quad：直接聚合的真实实例纠错未被参数更新实现。** 冻结、新结构和TTA都选择背景人物区域，IoU0；相同三读取直接平均选到前景人物，IoU0.573844。这个对照证明无需更新的观测组合有可用信息，但当前恢复/对应目标没有把它转化为最终选择。不能把最终输给直接读出解释为候选中没有好框。
- **R87_CASE006，Drone：已有坏例仍未解决。** 各本轮主臂均为0，图上仍选右侧树木/背景附近，而较好的候选位于远处车辆区域。局部支持也跟随错误区域；该较好候选在新结构与更新后仍位于很低的匹配排名。新路径没有仅凭“空间化表示”自动获得正确实例绑定。

R87_CASE003本轮各主要状态均保留0.480706，没有重复上一轮换错人物的损伤；R87_CASE004即本轮CASE002仍发生0.452964→0；R87_CASE006仍为0。这是固定旧病例追踪，不以修好旧病例数代替全体分数。

## 组件作用与相互干扰

去关系反馈后的直接与TTA最终选择均在336条完全相同；角色支持置换使直接选择改变1条但IoU不变，TTA选择全部相同。去新增对应、去局部偏好或只用物理恢复，TTA最终选择也均与完整主方法相同。这些模块会改变中间概率/参数，不能说它们没有执行；当前固定组合没有产生额外任务决策。

对保存首步作同尺度比较：324条中full相对no_correspondence的实际参数差与full更新幅度之比，中位约4.33%，故新增对应目标确实影响了权重，不是断梯度。其作用较小、可能未越过决策边界；不能凭一次固定权重否定所有对应自监督。局部偏好仅13对/9条，事后IoU排序3对较高、2对较低、8对相同，既稀疏也不能认证身份。

未参与更新与验收的第三遮挡中，同区域读取表示误差均值从0.00261140升到0.00264030。训练目标下降尚未带来这一对应指标的均值改善，更未形成最终grounding增量。因此不继续增加当前拟合步数或把当前区域解释成可靠教师。

本轮保留了全局、局部、角色、参照位置和最终选择的可检查接口，也暴露出前提不足：原生全局注意力不是实例定位器，显式假设仍受A256支持限制；未经新训练的软空间路由可能让原来较稳的grounding对同一物理更新更敏感。停止当前完整TTA配方，局部读取的微弱零更新收益只保留为开发线索，不宣称已实现可靠角色绑定。


最大单正例对均值贡献+0.011651 pp，总增量-0.124596 pp，其余335条仍以336为分母合计-0.136247 pp。这只是尾部贡献诊断，不从总体分母剔除该案例。

## 支持、监督覆盖与第三遮挡

全体角色764个、路由token3598个、多空间假设2289个、可执行关系172条；未知支持26条输入。部分关系偏好13对、覆盖9条输入。

封存后按候选IoU诊断部分比较：更高3、更低2、相等8。IoU排序不是实例身份金标；不能据此按GT筛选训练偏好。

第三组遮挡未参与更新、验收或选参；有可比较读取的310条，同区域表示误差均值从0.00261140到0.00264030。该误差衡量当前区域表示稳定性，不能替代grounding。

空间支持资格诊断（非身份纯度）：全部角色的全局注意力落入任一A256支持的质量中位数0.443984，落入三个假设的质量中位数0.072802。本版的显式假设仍由A256支持提出；候选之外的参照物只保留在全局路径中，没有新建候选外实例发现能力。

## 场景与证据边界

全部样本和失败/未知/零更新保留；场景表单独公开。DEV此前已多轮曝光，五个场景没有增加，不将帧看作独立场景或给出逐帧独立置信区间。本轮不使用GT跨度/实例标签来生成路由、偏好、更新或步长验收。新任务GT几何评分为0，只在预测封存、独立原生/损失审计后关联既有A256 IoU。病例选择和第三遮挡诊断不能反过来修改本轮方法。

局部空间偏置并非源域训练过的完整EG-3DVG能力。继承的实体解析、A256/seed支持、注意力和稀疏可见点都可能绑定错误对象；没有新增观测分辨率或RGB语义专家。一次固定倍率配置的阴性结果只能约束当前迁移实现，不能否定经过源域训练的全局—局部结构。

## 审计与成本

21臂原生产耗时646.350s，原生suffix调用39837，目标计算12271，实际接受3360次更新，峰值已分配GPU显存1589206016 bytes。额外相同三视图零更新控制1008次suffix调用。独立检查5712个端点、28560个实际损失标量，读出误差上限0，路由重建误差0，损失误差0。这些是冻结前缀重放成本，不是端到端系统延迟；图册原媒体/描述/坐标仅本地PRIVATE。

## 采用判断

主方法没有同时超过本轮同信息直接、三读出和冻结R76，不采用这套完整TTA配方。分别保留有总体净作用的读取组件为开发线索，不以辅助误差下降代替任务收益。 不追加倍率/损失/步数搜索，不自动开始源域训练，FINAL继续封存，无后续实验排队。
