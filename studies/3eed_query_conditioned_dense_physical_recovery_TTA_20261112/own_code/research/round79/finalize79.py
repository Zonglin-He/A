"""Measured decision and report from fixed-model, post-seal query-recovery results."""
from core79 import *
T=read(H79/'score_dev336_attempt001/TASK_SUMMARY.json')['ALL336'];B=T['balanced_mIoU_percent'];S=read(H79/'SOURCE128_SUMMARY.json');D=read(H79/'SOURCE_DIRECTION_SUMMARY.json');cost=read(B79/'prediction464_attempt001/COUNTS.json');audit=read(H79/'independent_native_audit_attempt002/AUDIT_RESULT.json');scalar=read(H79/'independent_scalar_audit_attempt001/AUDIT_RESULT.json');plan=read(H79/'experiment/DESIGN_LOCK.json');pin(plan);gate=all(B['query_radius_sgd1']>B[a]+1e-10 for a in ['uniform_sgd1','frozen_bi','frozen_direct3','shuffle_radius_sgd1','wrong_binding_radius_sgd1','R59']);dump(H79/'EXTENSION_GATE.json',{'state':'R79_FIXED_QUERY_WEIGHTING_EXTENSION_GATE','passed':gate,'primary':B['query_radius_sgd1'],'comparators':{a:B[a] for a in ['uniform_sgd1','frozen_bi','frozen_direct3','shuffle_radius_sgd1','wrong_binding_radius_sgd1','R59']},'not_selected_from_source_GT':True})
if gate:raise SystemExit('FIXED_QUERY_METHOD_EXTENSION_REQUIRED_BEFORE_CLOSEOUT')
assert sha(ROOT/'crossplatform/round6/waymo_source100/checkpoints/epoch100_model_attempt001.pth')=='0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd';assert sha(ROOT/plan['init_state'])==plan['init_state_sha256'];old=read(H79/'experiment/INITIAL_RESEARCH_STATE_PRIVATE.json');current=read(ROOT/'research/STATE.json')
for k in ['avis','affordance']:
 if k in old['directions']:assert old['directions'][k]==current['directions'][k]
decision={'state':'R79_COMPLETED_QUERY_SENSITIVITY_WEIGHTING_NOT_ADOPTED','adopt_query_weighting':False,'retain_uniform_SGD_development_candidate':B['uniform_sgd1']>B['frozen_bi'],'uniform_candidate_not_crossdomain_confirmed':True,'query_gain_vs_frozen_pp':B['query_radius_sgd1']-B['frozen_bi'],'query_gain_vs_uniform_pp':B['query_radius_sgd1']-B['uniform_sgd1'],'query_gain_vs_shuffle_pp':B['query_radius_sgd1']-B['shuffle_radius_sgd1'],'query_gain_vs_wrong_pp':B['query_radius_sgd1']-B['wrong_binding_radius_sgd1'],'uniform_gain_vs_frozen_pp':B['uniform_sgd1']-B['frozen_bi'],'same_parameter_radius_verified':True,'source_direction_diagnostics':D,'extension_gate_passed':False,'no_new_source_training':True,'no_external_model':True,'FINAL_access':False,'no_follow_on_queued':True};dump(H79/'DECISION.json',decision)
config=dict(plan);config.pop('inputs_sha256');config['source_ABC_and_R59_not_computed_or_emitted']=True;config['one_step_preservation_gradient_zero_at_initial_state']=True;dump(H79/'PUBLIC_FIXED_CONFIGURATION.json',config)
dump(H79/'NORMALIZED_COSTS.json',{'counts':{'source_inputs':128,'target_inputs':336,'source_arms':10,'target_arms':12,'target_task_values':4032,'source_task_values':1280,'target_ordered_negatives':scalar['ordered_negatives'],'source_ordered_negatives':scalar['source_ordered_negatives'],'new_grounding_suffix_calls':cost['grounding_suffix_calls'],'saliency_gate_backward_calls':cost['sensitivity_backward_calls'],'auxiliary_parameter_gradient_calls':cost['auxiliary_gradient_calls'],'actual_saved_updated_states':audit['actual_updated_states'],'source_training_updates':0,'new_full_point_backbone_captures':0,'new_raw_GT_geometry_scores':0},'timing_seconds':{'all_source_and_target_predictions':cost['seconds'],'successful_native_audit':audit['seconds']},'peak_bytes':{'prediction_GPU_allocated':cost['peak_GPU_bytes']},'flags':{'saliency_cost_counted':True,'same_information_but_different_compute_costs':True,'cached_prefix_timings_not_end_to_end':True,'all_negatives_and_failed_audit_retained':True}})
dump(H79/'NORMALIZED_HISTORY.json',{'events':[{'stage':'real_model_engineering','attempt':1,'status':'passed','valid_outputs':4,'before_GT':True},{'stage':'source_only_output_schema_draft','attempt':1,'status':'revised_before_prediction','repair_type':'omit_unavailable_source_ABC_R59','producer_predictions_changed':False,'prior_attempt_preserved':True},{'stage':'unlabeled_predictions','attempt':1,'status':'completed_sealed','valid_outputs':5312,'seconds':cost['seconds'],'before_GT':True},{'stage':'native_audit','attempt':1,'status':'failed','failure_type':'historical_source_normalization_dtype_contract','repair_type':'exact_actual_source_probabilities_and_historical_argmax','producer_predictions_changed':False,'tolerance_changed':False,'prior_attempt_preserved':True},{'stage':'native_audit','attempt':2,'status':'passed','valid_outputs':2320,'seconds':audit['seconds'],'before_GT':True},{'stage':'task_and_source_gradient_join','attempt':1,'status':'passed','valid_outputs':5312,'producer_predictions_changed':False}]})
dump(H79/'PUBLIC_PROVENANCE.json',{'state':'R79_FROZEN_R76_SOURCE_E6_QUERY_SALIENCY_RECOVERY','source_training':False,'original_checkpoint_and_R76_module_unchanged':True,'original_A256_unchanged':True,'source_labels_only_in_postseal_diagnostics':True,'R77_source_grounding_gradient_cache_reused_only_postseal':True,'target_labels_in_weights_pairs_step_or_eligibility':False,'correct_query_weights_and_shuffle_same_multiset':True,'geometry_wrong_control_partial_not_total_evidence_destruction':True,'source_ABC_R59_omitted_not_fabricated':True,'historical_reference_status_reconciled_with_its_actual_old_update_records':True,'source_baseline_dtype_probability_difference_max':1.9955991670261852e-7,'source_baseline_argmax_changes':0,'source_historically_exposed':True,'target_historically_exposed':True,'blind_test_claim':False,'FINAL_access':False,'prediction_seal_sha256':audit['prediction_seal_sha256'],'native_audit_sha256':sha(H79/'independent_native_audit_attempt002/AUDIT_RESULT.json')})
dump(H79/'REPORT_NUMERIC_AUDIT.json',{'state':'PASS_R79_REPORT_DERIVED_FROM_AUDITED_SAVED_VALUES','n':336,'task_values':4032,'source_task_values':1280,'target_negatives':scalar['ordered_negatives'],'source_negatives':scalar['source_ordered_negatives'],'query_vs_uniform_pp':decision['query_gain_vs_uniform_pp'],'query_vs_shuffle_pp':decision['query_gain_vs_shuffle_pp'],'uniform_vs_frozen_pp':decision['uniform_gain_vs_frozen_pp'],'same_radius_error':audit['max_matched_radius_error'],'source_checkpoint_unchanged':True,'paused_registries_unchanged':True})
names={'source':'原始3EED','frozen_bi':'R76冻结双向','frozen_direct3':'同信息冻结直接平均','uniform_sgd1':'均匀恢复SGD一步','query_raw_sgd1':'Query加权SGD原幅度','query_radius_sgd1':'Query加权，同均匀半径（主方法）','shuffle_radius_sgd1':'打乱权重位置，同半径','wrong_binding_radius_sgd1':'正确权重＋错配绑定，同半径','R76_Adam1':'历史R76 Adam一步','R76_Adam3':'历史R76 Adam三步','frozen_ABC':'原始ABC三采样','R59':'R59直接表示融合'}
target_table='\n'.join(f"| {names[a]} | {T['drone'][a]['mIoU_percent']:.6f} | {T['quad'][a]['mIoU_percent']:.6f} | {B[a]:.6f} | {(T['drone'][a]['Acc25_percent']+T['quad'][a]['Acc25_percent'])/2:.4f} | {(T['drone'][a]['Acc50_percent']+T['quad'][a]['Acc50_percent'])/2:.4f} |" for a in ARMS79)
src_table='\n'.join(f"| {names[a]} | {S[a]['mIoU_percent']:.6f} | {S[a]['Acc25_percent']:.4f} | {S[a]['Acc50_percent']:.4f} |" for a in S)
dir_table='\n'.join(f"| {names[a]} | {d['mean_descent_alignment_cosine']:.6f} | {d['grounding_descent_predicted_n']} | {d['grounding_increase_predicted_n']} | {d['median_parameter_L2']:.8f} |" for a,d in D.items())
w=read(H79/'WEIGHT_DIAGNOSTIC_ROWS.json');u=read(H79/'UPDATE_SUMMARY_ROWS.json');qstats={pl:next(r for r in u if r['platform']==pl and r['arm']=='query_radius_sgd1') for pl in ['waymo','drone','quad']};eff={t:float(np.mean([r['query_'+t+'_effective_fraction'] for r in w])) for t in ['P','I','D']};changes=[]
for pl in ['drone','quad']:
 r=T[pl]['query_radius_sgd1'];changes.append(f"{pl}相对冻结改好{r['improved_vs_frozen_bi']}条、改坏{r['worsened_vs_frozen_bi']}条、换query{r['query_changes_vs_frozen_bi']}条")
text=f'''# R79：候选竞争条件化的稠密物理恢复TTA

**本轮不采用Query敏感度加权：正确位置的加权没有超过均匀更新、打乱位置或错配绑定控制。** 保留均匀SGD一步的DEV开发正结果，但它不是新加权方法的收益，也未完成跨recording确认。依据预设门槛停止本轮，不追加权重／步长网格或新的源域训练。

## 固定模型与唯一机制变化

使用原R76双向epoch6（SHA84f3455...），不是R78继续训练的模型。原始3EED Waymo-only epoch100、RGB CNN、恢复头、匹配头和A256全部不变，仅更新原102498个Adapter／局部双向交互／门控参数。没有源域训练、新专家、预训练视觉网络或FINAL访问。原R76已有点/RGB/depth逐token恢复；本轮只改变局部误差的空间汇总，不把token本身等同、不把attention当实例标签、不复制query-text矩阵作为新纠错目标。

零步完整描述前向，在point_adapter后的1024点tokens及image_adapter后的1536图像tokens上设置分析用标量门。前者288维、后者64维，均未直接与query向量作距离对齐。只在门为1处计算最多三对当前候选log-odds分差的绝对梯度，平均得到两种非负敏感度。主第一名与其余三个高分不同区域假设竞争，不给任何候选正／负标签。候选去重使用原框和实际seed点支持（IoU≥.9，或IoU≥.5且支持Jaccard≥.5、共享seed≥4），不是语义类别筛选，也不宣称得到真实实例身份。

在每个遮挡、每类有效监督位置内单独归一化敏感度；保留25%均匀底座，先将敏感度质量限制在8/N内再混合（最终上界6.25/N）。全部敏感度为0时退回均匀。权重在零步模型上生成后detach并固定；不允许模型通过改变权重隐藏困难恢复。点损失使用点侧权重，RGB和深度各自使用图像侧权重。目标仍为冻结点特征前64维、真实RGB patch均值、实测稀疏logdepth，分别保留0.1/0.1/0.01系数。点特征是模型表示目标，不是真实语义标签；深度预测仍只从反向交互前图像特征读取，保持防复制约束。

原分布KL与query表征保持项按R76定义保留；**本轮一步更新从零步初始化开始，保持项在起点的梯度为0**，不能把它描述成已经约束了多步漂移。使用统一一步SGD alpha0.5/clip1、两种76001/76002互补遮挡、逐例恢复。选择SGD是固定既有R78算子以隔离加权，历史R76的Adam仅作额外参照，不把两种优化器差异归为加权贡献。所有核心加权／打乱／错配方向再缩放到本例均匀更新的同一L2参数半径，另保留未缩放Query原幅度对照。

## 真实样本与工程有效性

2条真实源VAL及Drone/Quad各1条检查通过：全1分析门与原生预测差0，均匀局部汇总与原辅助损失最大差3.73e-9；正确／打乱权重保持同一数值多重集合，产生非零真实更新。正式源128+目标336共464条，全部在使用标签前封存。再生成全部敏感度、权重和2320份真实更新参数并写回读出，误差：概率{audit['max_probability_error']}、参数{audit['max_regenerated_parameter_error']}、敏感度{audit['max_saliency_regeneration_error']}；同半径最大偏差{audit['max_matched_radius_error']:.3g}。

审计001发现裸source参照归一化精度约定不同：当前float32与历史float64概率最多差1.996e-7，464条argmax完全相同；审计002严格核对实际float32概率和历史相同选框，其余参考概率与更新态仍按原严格容差。未更改任何模型、预测、指标或容差；失败源码／锁／日志保留。源域没有ABC或R59观测缓存，因此根本不输出这两行，而不是冒充已测过。

## 源域方向诊断：仅在封存后用标签

源128是R77已暴露诊断样本，未用GT选择候选、权重、步长、是否更新。重用R77已封存的同一R76参数态grounding梯度，逐文件hash校验，乘以本轮实际写入的参数变化；这是一阶代理诊断，不替代实际mIoU，也不是在线GT筛选。

| 更新 | 平均descent alignment cosine | 预计任务损失下降数 | 预计上升数 | 参数L2中位数 |
|---|---:|---:|---:|---:|
{dir_table}

正确Query权重并没有得到比均匀或打乱更好的源域任务方向。即使局部敏感度高，也不代表恢复该位置会给参数提供正确的grounding修正。

| 源128方法 | mIoU % | Acc25 % | Acc50 % |
|---|---:|---:|---:|
{src_table}

## 目标DEV336全部结果

完整336条、Drone/Quad各168条、原有五个recording。所有历史曝光保留说明，不称新独立测试。新预测全部重算原生cross encoder和decoder后缀，复用冻结前缀和RGB缓存；任务评分只在封存及原生审计后join固定A256候选的既有IoU。旧Adam/ABC/R59是明确标注的历史参考，历史是否更新的状态取自原记录，不把本轮未重跑误写成旧方法零更新。

| 方法 | Drone mIoU % | Quad mIoU % | 两平台平均 % | 平均Acc25 % | 平均Acc50 % |
|---|---:|---:|---:|---:|---:|
{target_table}

主方法对冻结模型{B['query_radius_sgd1']-B['frozen_bi']:+.6f}pp，但对同半径均匀更新{B['query_radius_sgd1']-B['uniform_sgd1']:+.6f}pp，对打乱位置{B['query_radius_sgd1']-B['shuffle_radius_sgd1']:+.6f}pp，对错配绑定{B['query_radius_sgd1']-B['wrong_binding_radius_sgd1']:+.6f}pp。因此不能仅凭超过frozen就采用。{'; '.join(changes)}。所有正负例计入分母，没有按GT选择有恢复空间的样本才计分。

正确权重有效样本比例（1/sum(w²)除以有效token数）在P/RGB/depth平均为{eff['P']:.4f}/{eff['I']:.4f}/{eff['D']:.4f}，没有退化成均匀；回退权重集合总数={sum(r['uniform_fallback_weight_sets'] for r in w)}。正确加权proposal与均匀方向平均cosine：Drone{qstats['drone']['mean_cosine_to_uniform']:.6f}、Quad{qstats['quad']['mean_cosine_to_uniform']:.6f}。打乱控制更接近均匀方向，不是代码没使用权重；当前相关性代理改变了方向，却未改善最终任务。

错配仅破坏局部点—像素交互绑定，保留原RGB/depth等监督，不是完全打乱所有物理信息。它不必崩溃；但其表现相近或更好时，也没有证据说正确绑定赋予了额外收益。

## 裁决与保留的新观察

**停止当前Query敏感度加权配方，不调epsilon、cap、pair数或步长；未触发VAL1280／TRAIN1024扩展，FINAL未动。** 这仅否定本次“margin门敏感度作为恢复权重”原型，不证明所有局部恢复信号均无用或所有任务相关选择无效。

均匀SGD一步得到{B['uniform_sgd1']:.6f}%，相对冻结{B['uniform_sgd1']-B['frozen_bi']:+.6f}pp，是必须保留的开发正结果。它采用原辅助信号，与旧Adam不同，不是新加权方法贡献；源域却下降，且DEV已经多轮暴露，不能据此宣称稳定跨域收益。打乱权重控制的{B['shuffle_radius_sgd1']:.6f}%同样完整保留，不挑Query版本或最好控制冒充主方法。

当前可确认的是：敏感度与有效恢复方向不是一回事，单纯把当前模型最敏感的位置优先恢复没有通过总体比较。模型可能依赖背景／错误绑定，或门敏感度与可更新参数上的恢复梯度不一致；这些是竞争解释，不把它们当作已证明根因。若继续，应先固定当前更强均匀TTA参照、用固定范围验证其增量，再决定是否更换监督选择机制；本轮没有排队新实验。

机制参考来自 [TransFusion的query条件化证据读取](https://openaccess.thecvf.com/content/CVPR2022/html/Bai_TransFusion_Robust_LiDAR-Camera_Fusion_for_3D_Object_Detection_With_Transformers_CVPR_2022_paper.html) 与 [MeTTA的body-specific辅助修复](https://openaccess.thecvf.com/content/CVPR2024/html/Hu_Fast_Adaptation_for_Human_Pose_Estimation_via_Meta-Optimization_CVPR_2024_paper.html)；前者是监督3D检测，后者包含源域元训练。本轮只是任务条件化选择思想的受控迁移，不是它们的复现，不继承其成功保证，也不将机制拼接称为已成立新颖贡献。

## 成本与证据

生产源128+DEV336全部条件耗时{cost['seconds']:.3f}s，7888次grounding后缀、1392次gate敏感度反向、1856次辅助参数梯度计算，峰值torch分配{cost['peak_GPU_bytes']/1e9:.3f}GB。敏感度每例一次完整前向与最多三次反向，额外计算已经计入，不能称免费；此处是缓存前缀的整组实验时间，不是单臂端到端延迟。成功原生审计另{audit['seconds']:.3f}s；失败审计和输出schema草稿保留，不重跑预测。

目标4032任务值/{scalar['ordered_negatives']}有序负比较，源1280任务值/{scalar['source_ordered_negatives']}有序负比较，源标签方向诊断640行，所有场景／状态／负例公开。权重与敏感度向量、原始数据／文字／坐标／ID／特征／权重文件不公开；只导出代码、固定协议和匿名标量。原epoch100与R76e6文件hash不变，其他paused项目登记不变。**不采用当前加权TTA；均匀SGD仅保留开发候选。无后续排队实验。**
''';(H79/'REPORT.md').write_text(text);print(decision)
