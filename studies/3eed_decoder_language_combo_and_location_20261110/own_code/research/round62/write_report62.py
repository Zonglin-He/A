"""Build R62 report from audited saved aggregates, no experiments or scoring."""
from common62 import *
H=ROOT/'research/round62'
main=read(H/'ANALYSIS_MAIN.json');sem=read(H/'ANALYSIS_SEMANTICS.json');roles=read(H/'role_diagnostics_attempt001/SUMMARY_PUBLIC_SAFE.json')
def table(mm):
 s='|配置|Drone mIoU %|Quad mIoU %|两平台平均 %|Acc@25 %|Acc@50 %|相对 ABC pp|改善/恶化/换 query|\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
 for m in mm:s+='|'+m['arm']+'|'+ '|'.join(f'{m[k]:.6f}' for k in ['Drone_mIoU_percent','Quad_mIoU_percent','balanced_mIoU_percent','balanced_Acc25_percent','balanced_Acc50_percent','gain_vs_ABC_pp'])+f"|{m['IoU_improved_vs_ABC']}/{m['IoU_worsened_vs_ABC']}/{m['query_changes_vs_ABC']}|\n"
 return s
text='''# R62：R59 组合、六层文字读取干预与语义对应检验

本轮已完成真实 DEV336 实验、预测封存、独立实现复算和保存任务分数审计。结论是：**R59＋末层文字增强没有测得额外收益；提前增强文字读取也没有突破末层；原有文字分支确实影响任务，但本轮增益尚不能归因为修复了主体／参照物绑定。** 不采用该组合，不开发动态门控，不新增损失、倍率扫描或多层叠加。保留固定末层增强为有限的开发诊断参照，R59 direct 仍是本轮更强的同信息无更新参照。

## 协议与实际执行

固定 Waymo-only epoch100、已多轮暴露的 DEV336（Drone168＋Quad168，2＋3个场景）、原有 A/B/C 三种采样、原始 A 的256个最终候选框、R57 候选对应及 UNKNOWN 质量、既有文字跨度及读出、R59 原公式与系数。本轮没有校准新的融合权重，没有源域训练、梯度更新、伪标签拟合或在线 EMA。没有读取或预测 FINAL384，没有运行 ALL5912。开发标签仅用于封存后任务评分和显式开发位置选择，未进入推理补丁、置换、候选对应或 R59 目标构造。

首次恢复 decoder0 的输入前缀；2条工程样本的6次完整前向加上正式336×3＝1008次前向，共1014次完整源模型前向。所有旧 A/B/C 的采样点索引、最终框、匹配分数、hidden 和 query/token 匹配表示均逐项精确复现。该工程检查不是新增独立样本。后续干预从前缀重放真实六层 decoder，包含每层中心／尺寸预测和下一层位置输入；不是只读取早期预测，也不是冻结中间位置后近似重放。最终任务评分始终选择原始 A256 框。

物理干预为指定 `decoder[k].cross_l.out_proj` 的 weight 与 bias 同乘1.5，每个输入从源权重开始、结束后恢复。不改变该次 attention 的 Q/K，也不重新分配该层 attention。一次只改一层。组合臂重新计算已编辑 A/B/C hidden、R59 中的表示方向、原始范数、目标及置信度，而非复用旧目标。

## 1. 组合实验：无已测互补收益

'''
lookup={m['arm']:m for m in main['metrics']}
text+=table([lookup[k] for k in ['source','frozen_ABC','r59_direct_all','native_l5','R59_l5']])
text+='''
R59＋末层增强相对 R59 的平均 mIoU、两个平台、Acc@25/50 均完全相同。最终 query 有2条改变，但对应 IoU 均未改变；不能说全部中间表示或全部候选索引相同。相对 ABC 的38条 query 变化中10条 IoU 改善、8条恶化均保留。零编辑的 ABC/R59 bridge 均保持全部 Top1，R59 分数逐元素精确复现；组合无增益不是忘记重算 R59 的结果。

## 2. 六层单位置干预：没有更早位置带来更大净收益

'''
text+=table([lookup['native_l'+str(i)] for i in range(6)])
text+='''
按预先固定的“相对 ABC 总体正增益最大、同分取最早层”规则选择 layer5开展机制诊断。这是**在已暴露 DEV 标签上选位置**，不是独立确认或部署时用标签选层。layer5只比layer2高0.004630 pp，不能称为稳定最优位置。

layer5相对ABC仅6条 query变化：4条IoU改善、1条恶化、1条不变；Drone贡献全部IoU净收益，Quad的1条query变化未改变IoU。Acc@25救回3条、未破坏原超过阈值者，但仍存在连续IoU恶化。layer0/1/2/3/4救回的Acc@25样本分别为2/2/3/1/1条，全部属于layer5已救回的3条，没有新增过阈值救回；早期层另有少量不同的连续IoU改善，同时也有负例。故目前是触及相近小批决策边界的局部修复，不是广泛恢复 grounding 能力。

### 候选身份与对应审计

前缀中的原始点锚、query槽位与生成顺序保持不变；跨描述从不按 query 编号硬对应。干预后内部框会变化，因此额外保留全部6048条样本×层×视角的几何风险标量，未按风险或GT删除样本，也未偷偷重建 R57 W。各层同槽内部框 IoU 中位汇总约0.896—0.971，最终选中 A 槽的同槽IoU中位约0.959—0.979。重复框组外的几何匹配风险与旧传输质量失去几何重叠的比例均在结果中列出。

这些检查确认计算槽位和扰动程度，**不把同槽位等同于有独立真值保证的物理实例身份**。因此固定原框及旧候选对应的结果应按本协议解释。

## 3. 所选位置的语义控制：原分支有用，增强收益的特定语义机制未建立

所有控制都使用相同候选、采样和读出。有效文字 token 做固定哈希生成的非零循环置换，同一输入的ABC共享置换；padding不动。V-only控制保留Q/K和attention权重，破坏权重位置与value的对应。逐query匹配分支L2范数，排除简单范数差；零向量回退数为0。同步置换K/V仅作不变性正控，不能冒充语义破坏。

'''
text+=table(sem['metrics'])
text+='''
控制定义：

- `selected_value1/15`：整支换成 V-only 错配方向，范数分别匹配原支／1.5倍原支。
- `selected_increment_value`：原支 b 与0.5倍错配方向 a 混合，再将总范数匹配1.5倍原支。
- `selected_increment_uniform`：同上，但新增方向取有效value均匀平均。
- `selected_increment_joint`：新增方向使用同步K/V置换所得输出，验证实现不变性。

整支V错配的gain1与1.5均低于ABC；放大错配分支不带来原有+0.202187pp增益（错配内部gain1.5−gain1为−0.003488pp）。这支持原有汇总方向对任务有作用，削弱“任何同范数整支替换都等价”的解释。

但**不能将整支损伤归为新增0.5倍的语义贡献**。包含原支的错配新增量与uniform新增量都达到22.489129%，只比正确增强低0.015518pp，且均仅改变一条最终query并恶化一条IoU；相对ABC仍保留+0.186669pp。该总分差异说明，本轮主要收益不要求新增量严格沿原attention选择性的方向。它并未证明语言无关或纯数值校准是唯一机制。

这不是近零控制：错配／uniform混合后的最终分支相对正确分支的变化中位汇总分别0.347114／0.335808，cosine约0.939026／0.942269，变化超过0.1的query比例均值分别99.927%／100%。中位汇总口径是1008个样本视角各自query中位值的中位值，不是所有query直接汇集的全局中位数。同步K/V正控保持全部候选选择，最大分数差2.40e−7；分支范数最大相对误差3.57e−7。

混合控制准确形式为 `c(b+0.5a)`，总范数匹配使c随方向变化；它保留b参与混合，**没有严格固定原支有效系数为1**。范数匹配也未匹配视觉残差夹角、通道分布或下游非线性。控制仍保留较强原方向，不能外推任意方向、所有置换或独立语义因果机制。约92.3%的总分增益保留比例不是语义／数值贡献的因果分解。

## 4. 主体、属性与参照角色的封存后诊断

复用已有文字解析，不新增LLM调用或事后改解析；按原始字符跨度与精确字面匹配取token。仅A视角检查ABC最终旧／新候选对的主体、属性、参照物和关系引用剩余片段的平均匹配logit变化。关系引用剩余片段不是独立谓词解析；跨视角query编号不强行等同。该分析条件于已改变的候选对，仅为描述，不是监督或身份真值。

全部336条保留：主体跨度336条可用，参照物301条，关系引用剩余310条，严格可定位的字面属性87条。未知与缺失全部保留，**UNKNOWN不表示原句没有该属性**。6条换query样本均缺少本规则可单独定位的属性跨度，不能据此声称已经修复衣着／颜色读取。

这6条中主体、参照物及关系引用的候选对margin变化均呈相近模式：5条增加、1条减少；IoU改善的4条中3条增加、1条减少，IoU恶化的1条同样增加。所选层的attention在全部ABC上逐元素完全不变。故没有证据把收益专门归到某一文字角色或“注意力纠正”，结果与现有文字残差混合后末端候选margin的小范围改变相容，未识别唯一机制。

## 测量、独立审计与资源

组合、六层和语义三个阶段均先封存预测，再由独立计算实现重放真实decoder并重建匹配读出，然后复用已保存的原始256候选IoU评分，最后由独立汇总程序核对总体、平台、场景和全部有向负比较。三个阶段共5376条独立复算读出，全部Top1精确一致，最大hidden差小于1e−4、最大分数差小于4.6e−7。任务及场景汇总误差小于7.2e−15个百分点。角色跨度与匹配分数另行独立复算1344条，最大margin数值差1.59e−6。

三个表共9408个任务值，均重用历史保存候选IoU；没有新的原始GT几何评分调用。重复出现的source/ABC/末层臂只是阶段对照，不是新增独立样本。全部1510条有向负比较、所有336条及UNKNOWN状态、5个场景均发布。五场景样本有聚类与历史曝光，未将336条视为336个独立域或场景。未据此提出显著性或新场景稳定性结论。

source前向正式采集约97.97秒；组合/位置/语义干预生产分别49.87/139.41/186.26秒；独立GPU复算分别35.25/71.95/100.07秒。2条工程检查另计。峰值torch allocated约0.68GB，非整机显存占用；从冻结前缀重放节省了算力，时间不是完整端到端部署时延。没有并发占用唯一GPU。

逐例状态中的 `no_parameter_update` 沿用“没有优化器／伪标签拟合”的旧字段，不能按字面当作没有直接赋值编辑：native_l0—5及组合臂实际临时修改了out_proj权重与偏置，selected_native15继承该实际编辑结果，语义破坏臂以forward hook控制分支输出。原封存字段及结果不改写，固定配置明确列出这一字段语义限制。它们都没有梯度拟合、没有跨输入学习状态；固定1.5干预属于静态修复诊断。

本地首次公开导出因视角不同导致标量列稀疏、CSV固定表头不兼容而失败；失败attempt保留。仅将完整对应风险表改为JSON表示并新建导出attempt；第二次导出在隐私扫描阶段主动取消，以补充旧状态字段的准确语义说明，未发布。第三次仅修订该说明。未改预测、科学输入、评分或阈值，也未重新运行科学实验。

## 采用与停止决定

1. 不采用R59＋末层增强组合：总体无额外收益。
2. 不将更早文字分支放大视为突破：无位置优于末层，无新增Acc@25救回范围。
3. 固定末层1.5保留为开发局部修复参照；不替换更强R59，不宣称输入状态驱动TTA或已修复文字绑定。
4. 结束本轮“整体文字分支放大”的位置探索，不追加倍率、六层联合增强、动态门控、通道删除或新损失。
5. Astra仅在研发阶段审阅实现、已存汇总及机制边界，笔记单独附上；运行时仅有3EED和固定程序，不调用Astra或其他外部视觉模型。本轮结果不是三模态语义对齐已成立的证据。

## 复现入口与保存结果

`own_code/research/round62/`包含注册、前缀采集、真实decoder重放、组合/位置实验、V对应控制、独立审计、保存IoU评分、角色诊断和导出程序；`protocol/`保存固定配置、实验卡与开发选择信息；`COMBO336/`、`LAYERS336/`、`SEM336/`保留每臂所有匿名任务分数、平台／场景汇总和全部负比较；`results/`保留层间救回重合、对应风险和分支扰动强度；`ASTRA_MECHANISM_NOTE.md`记录受约束的机制解释。

公开包不含原始媒体、文字描述、真实标识、坐标、query/点索引、特征、权重、密钥或第三方快照；因此公开代码和标量不能替代受许可的原始数据及私有特征来完整重新推理。最终公开状态由本地publication回执记录，报告本身不冒充远端字节验证回执。
'''
sc=read(H/'ALL_MAIN_SCENE_SCALARS_PUBLIC_SAFE.json');text+='\n## 五个场景的主对照（mIoU %）\n\n|平台/场景|n|ABC|R59|末层1.5|R59＋末层|\n|---|---:|---:|---:|---:|---:|\n'
for pl in ['drone','quad']:
 for scene in sorted({x['scene_cluster'] for x in sc if x['platform']==pl and x['stage']=='layers'}):
  rr={x['arm']:x for x in sc if x['platform']==pl and x['scene_cluster']==scene and x['stage']=='layers'};text+=f"|{pl}/S{scene}|{rr['frozen_ABC']['n']}|"+'|'.join(f"{rr[k]['mIoU_percent']:.6f}" for k in ['frozen_ABC','r59_direct_all','native_l5','R59_l5'])+'|\n'
(H/'REPORT.md').write_text(text)
decision={'state':'R62_COMPLETED_MEASURED_DEVELOPMENT_ONLY_NO_NEW_TTA_ADOPTED','n':336,'combo_extra_R59_gain_pp':0.,'best_native_layer_on_exposed_DEV':5,'best_native_layer_mean_percent':lookup['native_l5']['balanced_mIoU_percent'],'best_native_gain_vs_ABC_pp':lookup['native_l5']['gain_vs_ABC_pp'],'best_native_gain_vs_R59_pp':lookup['native_l5']['balanced_mIoU_percent']-lookup['r59_direct_all']['balanced_mIoU_percent'],'earlier_layer_new_rescue25_vs_last_n':0,'retain_fixed_gain_local_development_reference':True,'adopt_combo':False,'adopt_dynamic_TTA':False,'continue_overall_language_amplification_sweep':False,'whole_branch_value_correspondence_has_task_effect':True,'selective_increment_binding_repair_mechanism_established':False,'FINAL384_accessed':False,'ALL5912_evaluated':False,'source_retrained':False,'new_losses_or_gates_added':False,'all_negative_comparisons_retained':True,'no_further_experiment_queued':True}
dump(H/'DECISION.json',decision)
config={'source':'Waymo_only_epoch100','development_n':336,'gain':1.5,'site_positions_tested':list(range(6)),'final_original_candidate_n':256,'internal_box_positions_recomputed':True,'frozen_ABC_transport_and_UNKNOWN':True,'R59_formula_targets_recomputed_no_coefficient_search':True,'selected_semantic_layer':5,'semantic_site_selected_on_exposed_development_labels':True,'semantic_runtime_uses_labels':False,'query_slot_not_independent_identity_gold':True,'whole_V_permutation_and_output_norm_control':True,'mixed_increment_not_strictly_protected_source_coefficient':True,'final_direction_strength_measured_independently':True,'postseal_roles_A_only_descriptive':True,'legacy_no_parameter_update_means_no_optimizer_fit_not_no_weight_assignment':True,'native_and_combo_direct_weight_bias_assignment_then_restore':True,'semantic_lesions_forward_hooks':True,'optimizer_steps':0,'new_losses':0,'dynamic_gates':0,'FINAL384_accessed':False,'full5912_evaluated':False,'no_new_source_training':True}
dump(H/'PUBLIC_FIXED_CONFIGURATION.json',config)
cost={'counts':{'development_n':336,'scene_clusters':5,'full_source_forwards':1014,'full_source_formal_forwards':1008,'full_source_engineering_forwards':6,'task_table_values_including_repeated_stage_controls':9408,'raw_GT_geometry_scoring_calls':0,'independent_readouts':5376,'role_numeric_rows':1344,'all_ordered_negative_comparisons':1510,'optimizer_steps':0},'timing_seconds':{},'peak_bytes':{},'flags':{'unique_GPU':True,'source_restored':True,'no_FINAL_or_fulltarget':True,'no_external_runtime_model':True}}
for tag,path in [('engineering_capture','research/round62/engineering_capture_attempt001'),('engineering_combo','research/round62/engineering_combo_attempt001'),('formal_capture','crossplatform/round62/prefix0_336_attempt001'),('combo','crossplatform/round62/combo336_attempt001'),('layers','crossplatform/round62/layers336_attempt001'),('semantics','crossplatform/round62/semantics336_attempt001')]:
 c=read(ROOT/path/'COSTS.json');cost['timing_seconds'][tag]=c['seconds'];cost['peak_bytes'][tag]=c['peak_GPU_bytes']
for tag in ['combo','layers','semantics']:cost['timing_seconds'][tag+'_independent_replay']=read(H/(tag+'_audit_attempt001')/'AUDIT_RESULT.json')['seconds']
dump(H/'NORMALIZED_COSTS.json',cost)
dump(H/'NORMALIZED_HISTORY.json',{'events':[{'stage':'source_prefix_capture','attempt':1,'status':'completed','valid_outputs':336,'attempted_calls':1008,'completed_calls':1008,'before_GT':True},{'stage':'combo','attempt':1,'status':'completed_audited','valid_outputs':2688,'before_GT':True},{'stage':'locations','attempt':1,'status':'completed_audited','valid_outputs':4368,'before_GT':True},{'stage':'semantics','attempt':1,'status':'completed_audited','valid_outputs':2352,'before_GT':True},{'stage':'role_readout','attempt':1,'status':'postseal_descriptive_audited','valid_outputs':1344,'before_GT':False},{'stage':'local_public_export','attempt':1,'status':'failed','failure_type':'sparse_scalar_CSV_schema','valid_outputs':0,'before_GT':False,'producer_predictions_changed':False,'tolerance_changed':False,'prior_attempt_preserved':True},{'stage':'local_public_export','attempt':2,'status':'cancelled_before_publication','repair_type':'clarify_legacy_no_optimizer_status','before_GT':False,'producer_predictions_changed':False,'tolerance_changed':False,'prior_attempt_preserved':True},{'stage':'local_public_export','attempt':3,'status':'registered_saved_only_retry','repair_type':'document_actual_weight_edit_and_legacy_status','before_GT':False,'producer_predictions_changed':False,'tolerance_changed':False,'prior_attempt_preserved':True}]})
print({'report_written':True,'decision':decision['state']})
