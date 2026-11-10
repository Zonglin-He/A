"""Assemble a bounded R63 scientific report from saved audited artifacts."""
from common63 import *
H=ROOT/'research/round63';S=read(H/'ANALYSIS_SCREEN_VALIDATED.json');HH=read(H/'ANALYSIS_HEADS.json');E=read(H/'ANALYSIS_EDIT.json');reach=read(H/'CAUSAL_REACH_PUBLIC_SAFE.json');control=read(H/'control_direction_diagnostic_attempt001/SUMMARY_PUBLIC_SAFE.json');sm={x['arm']:x for x in S['metrics']};hm={x['arm']:x for x in HH['metrics']};em={x['arm']:x for x in E['metrics']}
def metrics_table(rows):
 text='|配置|Drone mIoU %|Quad mIoU %|两平台平均 %|Acc@25 %|Acc@50 %|相对ABC pp|改善/恶化/换query|\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
 for x in rows:text+='|'+x['arm']+'|'+'|'.join(f'{x[k]:.6f}' for k in ['Drone_mIoU_percent','Quad_mIoU_percent','balanced_mIoU_percent','balanced_Acc25_percent','balanced_Acc50_percent','gain_vs_ABC_pp'])+f"|{x['IoU_improved_vs_ABC']}/{x['IoU_worsened_vs_ABC']}/{x['query_changes_vs_ABC']}|\n"
 return text
text='''# R63：decoder 实体角色路径、单 head 定位与 V 子空间定向编辑

**发现了一条能改变局部错误排序的计算路径，但没有找到足以解释主要错误、并带来总体有效收益的主体／参照绑定故障。** 第0层head6的参照token value路径移除相对ABC提高0.085424pp，匹配输出改动规模的随机token路径移除也提高0.083196pp。保护主体V输出的实际共享V矩阵编辑仅提高0.002229pp，无Acc@25/50收益；与R59组合没有额外成绩。不采用该编辑，不追加损失、rank/强度搜索或整体文字放大。

## 协议与测量边界

固定Waymo-only epoch100、已多轮暴露DEV336（Drone168＋Quad168，2＋3场景）、原有ABC采样、原始A256最终框、R57对应和UNKNOWN质量、现有主体／参照文字跨度与最终读出。不访问FINAL384，不运行ALL5912，不改变 source 训练，不加载外部视觉模型，不新增损失、梯度、优化器或伪标签拟合。

源数据复用R62封存的decoder0前缀，无新增完整点云backbone/文字encoder前向。干预均重放真实下游decoder、每层预测头及下一层几何位置，最终仍从原始256框选择。所有方法保留全部336条，角色可明确分离的300条允许干预，其余36条UNKNOWN保留各臂既有零编辑读出与总体分母。q槽位锚保持，内部几何变化及旧传输风险保留；槽位不被冒充独立物理实例真值。

顺序是全六层角色路径筛查→最多一个正收益层的8个head→最多一个正收益head的定向编辑。阶段预测先封存、独立实现重放、再使用历史保存的原候选IoU评分，最后独立重聚合。层/head选择使用暴露开发标签，运行时配置仅有固定选中位置与出处hash，不含任务分数、GT候选或逐例GT选择。这是开发机制诊断，不是独立确认。

## 1. 主体、参照物信息如何进入候选

原生decoder的计算顺序为query self-attention→`cross_l`文字读取→`norm_l`→`cross_v`点特征读取→FFN→匹配投影。每个query的几何位置参与Q，语言提供K与V；8个head各36维，经输出投影进入query残差。文字输出改变后，点云读取的query也会变化。最终候选排序还经过匹配MLP、归一化、固定token读出和ABC传输，attention质量本身不能代替最终选框分数。

本轮分离的是**所识别token槽的计算路径**，不是将attention解释为实例mask，也不是断言上下文化token仅包含它自己的实体语义。参照跨度采用既有多个参照物的并集；未独立认证各个现实参照实例身份。GT IoU只用于封存后几何质量评分，不能替代参照身份标签。

## 2. 六层三类路径干预

在每层分别：(1)移除主体token的加权V贡献；(2)移除参照token的加权V贡献；(3)交换主体/参照attention质量，保持两角色总质量及角色内部相对分配。V路径移除不重归一化剩余attention，不删除原输出bias；不是整体文字分支增益。role swap对极小质量维持原值。原生手工路径重构经过数值检查。

|层（0起）|移除主体V mIoU %|移除参照V mIoU %|交换角色质量 mIoU %|
|---|---:|---:|---:|
'''
for i in range(6):text+=f'|{i}|'+'|'.join(f"{sm[f'L{i}_{k}']['balanced_mIoU_percent']:.6f}" for k in ['drop_subject','drop_reference','swap_roles'])+'|\n'
text+='''
ABC为22.302460%，R59直接融合为22.611116%。最好全head路径为第0层移除参照V：22.386915%，相对ABC仅+0.084455pp，2条IoU改善、0条恶化、1条Acc@25救回。全18种角色路径干预加ABC，封存后假设GT逐样本择优的有限臂上限也仅22.391933%，所有干预合计只救回同一条Acc@25错误。这个上限是离线诊断，不是可部署方法，也不限制未测试计算；它说明本组干预的可达恢复范围很小。

## 3. 进一步定位到第一层head6的参照value路径

在同一层、同一参照路径，分别只移除一个head的贡献：

'''
text+=metrics_table([hm['H'+str(i)] for i in range(8)])
text+='''
按预定正总体收益最大、同分取最小head选择head6。head5亦有1条纠错；head3有1条恶化，其余head最终IoU不变。这确认特定计算可以造成局部决策变化，未证明该head是主要跨域错误源。最高head6仍低于R59约0.223231pp。

## 4. 实际定向编辑：保护主体值，削弱参照特有子空间

为了修改实际被干预的V路径，最终采用V子空间编辑，没有预设一个固定低秩容量。设增广语言输入为x=[language;1]，将所选head的V权重与bias合为M（36×289）。先计算被保护主体token的输入跨度，再将所有参照token对该跨度的残差作SVD，得到正交基U。执行：

`M_new = M_source - M_source @ U @ U.T`。

这使主体token在该head的线性V输出保持不变，同时消去参照角色相对主体的独有输入子空间。数值秩由输入奇异值的1e−6相对阈值决定，投影系数固定1，没有按GT选rank或扫描强度。实际只改`decoder[0].cross_l.in_proj_weight[792:828]`及相应bias，最多10404个参数条目；Q/K和out_proj尺度固定，源权重每个输入视图后恢复。

实际300条输入、900个采样视图均产生非零主编辑。残差秩1—24，权重改动Frobenius范数汇总中位0.620777；被保护主体的FP32线性V输出最大变化2.623e−6，该层attention变化严格为0。主编辑分支向量相对变化汇总中位6.61%，而head6参照路径完整移除约1.38%。因此不能把结果简单解释为“权重没有改到或整体改动幅度近零”。

初始未执行预案曾考虑Q-bias；在screen评分前撤回，因为切断V路径不能单独定位Q/K logit缺陷。随后未执行的均值rank1预案，在screen评分后、head评分前改为上述完整角色残差子空间，避免预设弱低秩方向。这些开发设计修订均保留并披露；只有最终V3进行了真实工程检查和正式编辑，不能将未执行预案当作失败实验或宣称全流程盲预注册。

## 5. 定向编辑与同信息对照结果

'''
text+=metrics_table(E['metrics'])
text+='''
`physical`是真实写入V矩阵；`functional`使用完全相同的输入与计算参数，但作为功能式参数传给attention，不修改模型状态。两种方式全部1008个采样视图的最终hidden逐元素完全一致，候选分数也完全一致。它是等价执行正控，不能将写参数本身宣称为额外收益。由表示输入精确相同可得，两者接相同确定性R59读出也等价；这不是额外评分了一条functional_R59实验臂。

`opposite`反向加强同一子空间；`shuffled`以同一保护跨度、置换token的残差子空间作参数Frobenius改动规模对照。后者900个角色有效视图中834有非零矩阵改动，66个视图（22条输入）因置换残差退化而不变，全部保留，不能声称该控制逐例都有完全相同改动预算。

`random_path`对head6的随机等数量token路径作移除，并逐query匹配原参照路径移除的输出向量改动范数。它达到22.385656%，几乎复现完整参照路径移除22.387884%的净收益；两者差仅0.002229pp。这削弱“局部改善必须依赖准确参照角色”的解释。

`text_only`将该层点云读取输出恢复为source，保留文字残差编辑；`visual_only`仅保留编辑诱发的该层点云读取输出，文字残差恢复source。前者保留唯一微小IoU改善，后者回到ABC。这只说明此处微小决策变化在所测单位置干预中可通过文字残差路径保留，未显示额外视觉读取纠错；后续层仍真实重算，不能解释为全模型直接/间接效应的可加比例。

主编辑相对ABC仅+0.002229pp、1条连续IoU改善、无Acc@25/50变化；组合与R59的分数相同，没有新增已测任务收益。所有错误、退化控制与UNKNOWN均计入分母。

## 6. 一个具体实现瓶颈：共享V参数的改动不能只落在参照路径

在结果锁定后，额外对全部336条/1008视图只做源模型第一层的描述性控制核验，没有新的候选预测、参数编辑或任务评分。900个角色有效视图中，随机token集合有678个与参照token完全无交集；参照与随机路径投影后向量cosine的汇总中位约0.74996。这不是完全相同的路径，也不能把随机对照一律说成仍在删同一参照物。部分上下文化消息方向相近，但不是“完全塌缩”。未中心化V矩阵首奇异方向能量比例汇总中位约0.638，V-bias路径与参照路径的范数比中位仅0.00847，**没有证据将收益归因于V bias故障**。

主投影虽然保护了主体token的V值，却仍会修改其他非主体、非参照token的值。用原attention传播这些变化时，非参照项变化向量范数与参照项变化向量范数之比的汇总中位约9.35。参照项自身也有实质方向变化（改变量/原向量范数中位0.523），其修改后范数/原范数仍约0.976；范数保留不是语义信息保留比例。

这给出了**具体的传播范围问题**：希望改变的是query—实体边上的证据，实际修改的是所有token共享的V线性映射。即使保护主体，关系和上下文等其他token仍可能受到更大的合成改动。这个观测为编辑效果弱提供了直接竞争解释，但不能把9.35解释为任务损失贡献比例，也没有唯一证明所有失败由该传播造成。

原生源路径的另一描述性结果是：126条候选可恢复错误中109条角色跨度可用；A视角源第一名相对最近可恢复决策边界上的好框，有66条主体/参照attention比更低、43条反而更高。不能将“少读参照”当统一正确方向，更不能据此构造使用GT的运行门控。

## 测量有效性、失败保留与成本

三阶段共12768条独立实现读出，全部Top1精确一致，最大分数差小于4.4e−7、hidden差小于4.2e−5。共**14784个表内任务值**（含重复阶段对照），全部复用历史原候选IoU，无新增原始GT几何评分。全部2056条有向负比较、每条输入与所有5个场景均保留；样本与场景有历史开发曝光，没有将336帧当独立新场景。

screen的诊断receipt曾因可变字典缓存别名而让不同干预的强度字段复用最后一次值。独立模型重算确认预测、IoU、原生head角色trace及独立生成的几何记录未受影响；错误强度列被隔离，不用于选层、报告或机制判断。原生trace汇总在不重跑模型/评分的情况下重新验证，后续head与编辑使用独立复制的receipt。原日志、代码版本、无效分析及有效结果均保留，未改预测或放宽审计容差。

全层screen生产510.79秒、独立重放267.82秒；head生产226.14秒、独立重放115.76秒；编辑生产246.64秒、独立重放106.51秒。工程2样本另计。源第一层控制描述性核验约30.17秒。峰值torch allocated约0.62GB，非整机显存或完整端到端部署占用。唯一GPU实验串行，未触碰其他暂停项目与定时任务。

## 采用与停止决定

1. 保留第0层head6为已确认的**局部计算影响**，不将它等同于主要语义绑定故障。
2. 不采用共享V残差子空间投影为新TTA；主编辑没有超过R59，也没有超过同信息功能式执行。
3. 不将参照路径硬移除升级为方法；其收益小，且匹配随机路径对照几乎复现。
4. 不再加大当前投影强度、选rank、加损失或重训source。单纯限制传播也不能凭空扩大本组干预仅1条Acc@25救回的证据范围。
5. 本轮没有验证“所有decoder参数编辑无效”或“3EED所有TTA无效”。所否定的是这组主体/参照value路径到共享V投影的现有配方。前置融合、query生成或其他绑定机制仍属未测试范围，本轮不自动启动。

## 结果与复现入口

`SCREEN336/`、`HEADS336/`、`EDIT336/`完整保留所有匿名任务值、状态、场景和有向负比较。`results/`提供有效总体/排序汇总、原生角色统计、主编辑全部视图过程标量、全部编辑的A视图过程标量以及控制有效性统计；完整私有过程记录在本地封存。`audits/`保存独立重放、汇总、表示等价及无效日志隔离回执。`protocol/`保留初始方案、明确未执行预案和实际执行设计/配置。公开代码仅包括实际执行或用于验证/导出的自有程序，不包含第三方快照。

原始媒体、描述、真实标识、坐标、query/点索引、特征、权重和密钥均不公开。公开代码与标量不能替代受许可原始数据和私有前缀来完整重新推理。远端字节与旧study保留的最终公开状态由publication本地回执记录。
'''
text+='\n## 五场景主对照（mIoU %）\n\n|平台/场景|n|ABC|R59|head6参照路径移除|V编辑|V编辑＋R59|\n|---|---:|---:|---:|---:|---:|---:|\n';scene=read(H/'ALL_EDIT_SCENE_SCALARS_PUBLIC_SAFE.json')
for pl in ['drone','quad']:
 for cluster in sorted({x['scene_cluster'] for x in scene if x['platform']==pl}):
  rr={x['arm']:x for x in scene if x['platform']==pl and x['scene_cluster']==cluster};text+=f"|{pl}/S{cluster}|{rr['frozen_ABC']['n']}|"+'|'.join(f"{rr[k]['mIoU_percent']:.6f}" for k in ['frozen_ABC','r59_direct_all','head_path_knockout','physical','physical_R59'])+'|\n'
(H/'REPORT.md').write_text(text)
dump(H/'PUBLIC_FIXED_CONFIGURATION.json',{'source':'Waymo_only_epoch100','n':336,'scene_clusters':5,'all_ABC_samples_original_A256_and_UNKNOWN_fixed':True,'source_prefix_reused_no_new_backbone_forwards':True,'selected_layer':0,'selected_head':6,'selection_on_exposed_DEV':True,'runtime_metrics_or_GT_used':False,'role_mask_source':'unchanged_R62_exact_literal_text_spans','role_available_inputs':300,'UNKNOWN_inputs':36,'selected_V_row_start':792,'selected_V_row_stop_exclusive':828,'max_changed_parameter_entries':10404,'projection_scale':1.,'SVD_relative_rank_cutoff':1e-6,'residual_absolute_zero_cutoff':1e-6,'data_rank_min':1,'data_rank_max':24,'QK_and_outproj_gain_unchanged':True,'protected_subject_V_value_max_error':2.6226043701171875e-6,'same_layer_attention_max_error':0.,'physical_and_functional_hidden_values_bitexact':True,'parameter_fit_or_gradient_steps':0,'new_losses':0,'whole_language_gain':1.,'unexecuted_Qbias_and_meanV_designs_disclosed':True,'V3_after_screen_before_head_score':True,'postresult_control_diagnostics_not_method_selection':True,'FINAL384_accessed':False,'ALL5912_evaluated':False})
cost={'counts':{'development_n':336,'scene_clusters':5,'role_available_n':300,'UNKNOWN_n':36,'new_full_source_forwards':0,'formal_screen_chain_replays':19152,'formal_head_chain_replays':8064,'formal_value_chain_replays':8064,'independent_chain_replays':35280,'engineering_chain_replays':162,'source_only_firstlayer_diagnostic_calls':1008,'independent_readouts':12768,'table_task_values_including_repeated_controls':14784,'all_ordered_negatives':2056,'new_raw_GT_geometry_calls':0,'optimizer_or_loss_steps':0},'timing_seconds':{},'peak_bytes':{},'flags':{'unique_GPU':True,'source_restored':True,'source_training':False,'FINAL_or_fulltarget':False,'external_visual_model_inference':False}}
for tag,path in [('engineering_screen','research/round63/engineering_attempt001'),('screen','crossplatform/round63/screen336_attempt001'),('heads','crossplatform/round63/heads336_attempt001'),('engineering_value','research/round63/engineering_value_attempt001'),('edit','crossplatform/round63/edited336_attempt001')]:
 c=read(ROOT/path/'COSTS.json');cost['timing_seconds'][tag]=c['seconds'];cost['peak_bytes'][tag]=c['peak_GPU_bytes']
for mode in ['screen','heads','edit']:cost['timing_seconds'][mode+'_independent_replay']=read(H/(mode+'_audit_attempt001/AUDIT_RESULT.json'))['seconds']
cost['timing_seconds']['source_only_control_diagnostic']=control['seconds'];cost['peak_bytes']['source_only_control_diagnostic']=control['peak_GPU_bytes'];dump(H/'NORMALIZED_COSTS.json',cost)
events=[{'stage':'screen_engineering','attempt':1,'status':'completed','valid_outputs':2,'before_GT':True},{'stage':'screen','attempt':1,'status':'completed_prediction_and_scalar_audited','valid_outputs':7056,'before_GT':True},{'stage':'screen_strength_logging','attempt':1,'status':'invalid_diagnostic_only','failure_type':'cached_mutable_receipt_alias','producer_predictions_changed':False,'tolerance_changed':False,'prior_attempt_preserved':True},{'stage':'native_trace_reaggregation','attempt':1,'status':'completed_saved_only','repair_type':'quarantine_strength_validate_native_traces','producer_predictions_changed':False,'prior_attempt_preserved':True},{'stage':'conditional_Qbias_design','attempt':1,'status':'not_executed','before_GT':True},{'stage':'conditional_meanV_design','attempt':2,'status':'not_executed','before_GT':True},{'stage':'conditional_fullV_subspace_design','attempt':3,'status':'locked_after_screen_before_head_score','before_GT':False},{'stage':'heads','attempt':1,'status':'completed_audited','valid_outputs':3696,'before_GT':True},{'stage':'value_engineering','attempt':1,'status':'completed','valid_outputs':2,'before_GT':True},{'stage':'value_edit','attempt':1,'status':'completed_audited','valid_outputs':4032,'before_GT':True},{'stage':'source_control_diagnostic','attempt':1,'status':'postresult_descriptive','valid_outputs':1008,'before_GT':False}];dump(H/'NORMALIZED_HISTORY.json',{'events':events});print({'report':'research/round63/REPORT.md','decision':read(H/'DECISION.json')['state']})
