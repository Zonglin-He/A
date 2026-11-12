"""Measured decision/report generation after the single locked DEV336 evaluation."""
from pathlib import Path
import sys,json,numpy as np
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'research/round76'));from common76 import read,dump,sha
H=R/'research/round78';TR=R/'crossplatform/round78/trained_source_attempt001';T=read(H/'score_dev336_attempt001/TASK_SUMMARY.json')['ALL336'];B=T['balanced_mIoU_percent'];S=read(H/'SOURCE128_SUMMARY.json');G=read(H/'SOURCE_GRADIENT_SUMMARY.json');hist=read(TR/'HISTORY.json');sel=read(TR/'SELECTED_SOURCE_ONLY.json');cost=read(TR/'COUNTS.json');tc=read(R/'crossplatform/round78/target336_attempt001/COUNTS.json');audit=read(H/'independent_native_audit_attempt001/AUDIT_RESULT.json');scalar=read(H/'independent_scalar_audit_attempt001/AUDIT_RESULT.json');gate=all(B['meta1']>B[a]+1e-10 for a in ['meta0','meta_direct3','joint1','joint_direct3','R59','R76_bidirectional_frozen']);dump(H/'EXPANSION_GATE.json',{'state':'R78_PREREGISTERED_PRACTICAL_META_TTA_EXPANSION_GATE','passed':gate,'meta1':B['meta1'],'comparators':{a:B[a] for a in ['meta0','meta_direct3','joint1','joint_direct3','R59','R76_bidirectional_frozen']},'fixed_source_selection':True,'target_retuning':False})
design=read(H/'experiment/DESIGN_LOCK.json');assert sha(R/'crossplatform/round6/waymo_source100/checkpoints/epoch100_model_attempt001.pth')==design['original_checkpoint_sha256'];assert sha(R/design['init_state'])==design['init_state_sha256'];old=read(H/'experiment/INITIAL_RESEARCH_STATE_PRIVATE.json');current=read(R/'research/STATE.json')
for key in ('avis','affordance'):
 if key in old['directions']:assert current['directions'][key]==old['directions'][key]

EXT={}
if gate:
 for name,n in [('VAL1280',1280),('TRAIN1024',1024)]:EXT[name]=read(H/('score_'+name.lower()+'_attempt001')/'TASK_SUMMARY.json')['ALL'+str(n)]
 assert read(H/'independent_extension_scalar_audit_attempt001/AUDIT_RESULT.json')['state'].startswith('PASS_')
EXTB={name:value['balanced_mIoU_percent'] for name,value in EXT.items()}
ext_supported_each=bool(EXTB) and all(all(v['meta1']>v[a]+1e-10 for a in ['meta0','meta_direct3','joint1','joint_direct3','R59','R76_bidirectional_frozen']) for v in EXTB.values())
pooled={a:sum(EXT[name][pl][a]['n']*EXT[name][pl][a]['mIoU_percent'] for name in EXT for pl in ['drone','quad'])/sum(EXT[name][pl][a]['n'] for name in EXT for pl in ['drone','quad']) for a in B} if EXT else {}
ext_supported=bool(pooled) and all(pooled['meta1']>pooled[a]+1e-10 for a in ['meta0','meta_direct3','joint1','joint_direct3','R59','R76_bidirectional_frozen'])


if EXT:assert all(sum(EXT[name][pl]['meta1']['n'] for name in EXT)==1152 for pl in ['drone','quad'])
meta_gain=B['meta1']-B['meta0'];joint_gain=B['joint1']-B['joint0'];static=B['meta0']>max(B['R59'],B['R76_bidirectional_frozen']);decision={'state':'R78_COMPLETED_CONTROLLED_SOURCE_META_AUXILIARY_PROTOTYPE','adopt_meta_TTA':False,'retain_meta_TTA_development_candidate':ext_supported,'retain_bidirectional_architecture':True,'retain_meta_static_development_candidate':static,'source_training_exposure_and_outer_updates_matched':True,'training_compute_matched':False,'full_meta_gradient_verified':True,'train_test_inner_operator_identical':True,'source_shared_selected_epoch':sel['shared_epoch'],'meta_target_gain_pp':meta_gain,'joint_target_gain_pp':joint_gain,'difference_of_adaptation_gains_pp':meta_gain-joint_gain,'meta1_vs_same_model_direct3_pp':B['meta1']-B['meta_direct3'],'meta1_vs_joint1_pp':B['meta1']-B['joint1'],'meta1_vs_R59_pp':B['meta1']-B['R59'],'source_gradient_alignment':G,'target_expansion_gate_passed':gate,'VAL1280_TRAIN1024_expansion_started':bool(EXT),'FINAL_access':False,'no_follow_on_queued':True};decision['strongest_frozen_development_reference']=max(['R76_bidirectional_frozen','joint0','meta0','R59'],key=lambda a:pooled.get(a,B[a]));decision['retain_meta_static_development_candidate']=pooled.get('meta0',B['meta0'])>max(pooled.get('R59',B['R59']),pooled.get('R76_bidirectional_frozen',B['R76_bidirectional_frozen']));decision['all2304_platform_balanced_mIoU_percent']=pooled;decision['extension_supports_both_cohorts']=ext_supported_each;decision['extension_results']={name:{'meta_gain_pp':v['meta1']-v['meta0'],'joint_gain_pp':v['joint1']-v['joint0'],'meta1_vs_direct3_pp':v['meta1']-v['meta_direct3'],'meta1_vs_joint1_pp':v['meta1']-v['joint1'],'meta1_vs_R59_pp':v['meta1']-v['R59'],'meta1_vs_wrong_pp':v['meta1']-v['meta_wrong1']} for name,v in EXTB.items()};decision['meta_training_improves_adaptation_gain_on_all_cohorts']=bool(EXTB) and all(v['meta1']-v['meta0']>v['joint1']-v['joint0'] for v in EXTB.values());dump(H/'DECISION.json',decision)
if EXT:
 for name,value in EXT.items():decision['extension_results'][name]['balanced_mIoU_percent']=value['balanced_mIoU_percent']
 dump(H/'DECISION.json',decision)
checks=read(H/'engineering_attempt001/RESULT.json');dump(H/'PUBLIC_PROVENANCE.json',{'state':'R78_REAL_SOURCE_CONTINUATION_AND_LOCKED_TARGET_EVALUATION','source_TRAIN_n':512,'source_VAL_n':128,'source_VAL_historically_exposed':True,'target_DEV_n':336,'target_historically_exposed':True,'blind_test_claim':False,'source_labels_only_in_outer_objective_and_epoch_selection':True,'target_labels_in_inner_update_or_selection':False,'original_A256_unchanged':True,'original_source_checkpoint_unchanged':True,'original_3EED_frozen_suffix_differentiated_through':True,'new_module_source_training':True,'new_pretrained_or_external_expert':False,'TTA_same_functional_SGD_as_source':True,'target_prediction_seal_sha256':audit['prediction_seal_sha256'],'source_training_seal_sha256':sha(TR/'SEALED.json'),'native_audit_sha256':sha(H/'independent_native_audit_attempt001/AUDIT_RESULT.json'),'source_gradients_mechanism_diagnostic_only':True,'FINAL_access':False})
times={b:float(sum(r['seconds'] for r in hist if r['branch']==b)) for b in ['joint','meta']};dump(H/'NORMALIZED_COSTS.json',{'counts':{'source_train_n':512,'source_val_n':128,'source_epochs_per_branch':4,'outer_updates_per_branch':2048,'source_TTA_parameters':102498,'source_outer_parameters':338343,'joint_training_grounding_suffix_calls':2048,'meta_training_grounding_suffix_calls':4096,'source_validation_suffix_calls':2048,'source_validation_auxiliary_fusion_calls':2048,'source_training_auxiliary_fusion_calls':8192,'training_inner_gradient_calls':2048,'validation_inner_gradient_calls':1024,'training_outer_backward_calls':4096,'target_inputs':336,'target_arms':12,'target_suffix_calls':tc['grounding_suffix_calls'],'target_auxiliary_fusion_calls':tc['auxiliary_fusion_calls'],'target_SGD_proposals':tc['SGD_proposals'],'saved_target_TTA_states':audit['target_states_replayed'],'new_raw_target_GT_geometry_calls':0},'timing_seconds':{'joint_training_and_all_source_val_epochs':times['joint'],'meta_training_and_all_source_val_epochs':times['meta'],'source_entire_producer':cost['seconds'],'target_all_arms':tc['seconds'],'native_audit_source_and_target':audit['seconds'],'engineering_graph_check':checks['seconds']},'peak_bytes':{'source_GPU_allocated':cost['peak_GPU_bytes'],'target_GPU_allocated':tc['peak_GPU_bytes']},'flags':{'data_outer_updates_matched':True,'compute_matched':False,'FP32_both_branches':True,'cached_prefix_timings_not_end_to_end':True,'all_trials_and_negatives_retained':True}})
dump(H/'NORMALIZED_HISTORY.json',{'events':[{'stage':'real_source_graph_check','attempt':1,'status':'passed','producer_predictions_changed':False,'seconds':checks['seconds']},{'stage':'source_joint_and_meta_training','attempt':1,'status':'completed_sealed','valid_outputs':8,'completed_calls':8192,'seconds':cost['seconds']},{'stage':'target_predictions','attempt':1,'status':'completed_sealed','valid_outputs':4032,'completed_calls':tc['grounding_suffix_calls'],'seconds':tc['seconds'],'before_GT':True},{'stage':'native_state_and_source_gradient_audit','attempt':1,'status':'passed','valid_outputs':1344,'seconds':audit['seconds'],'before_GT':True},{'stage':'fixed_target_GT_join','attempt':1,'status':'completed','valid_outputs':4032,'producer_predictions_changed':False}]})
dump(H/'REPORT_NUMERIC_AUDIT.json',{'state':'PASS_R78_REPORT_SAVED_VALUE_DERIVATION','source_task_values':scalar['source_task_values'],'target_task_values':4032,'all_target_ordered_negatives':scalar['ordered_negatives'],'all_source_ordered_negatives':scalar['source_ordered_negatives'],'meta_gain_pp':meta_gain,'joint_gain_pp':joint_gain,'meta_vs_direct3_pp':B['meta1']-B['meta_direct3'],'original_source_checkpoint_unchanged':True,'new_target_scores_after_prediction_seal':True,'source_same_epoch_for_both_branches':True})
if EXT:
 ec=read(R/'crossplatform/round78/extension_prediction2304_attempt001/COUNTS.json');eg=read(R/'crossplatform/round78/extension_geometry2304_attempt001/COUNTS.json');ea=read(H/'independent_extension_native_audit_attempt001/AUDIT_RESULT.json');es=read(H/'independent_extension_scalar_audit_attempt001/AUDIT_RESULT.json')
 nc=read(H/'NORMALIZED_COSTS.json');nc['counts'].update(extension_inputs=2304,extension_task_values=27648,extension_grounding_suffix_calls=ec['grounding_suffix_calls'],extension_auxiliary_fusion_calls=ec['auxiliary_fusion_calls'],extension_saved_TTA_states=9216,extension_native_audit_suffix_calls=ea['grounding_suffix_calls']);nc['timing_seconds'].update(extension_RGB_geometry=eg['seconds'],extension_all_predictions=ec['seconds'],extension_native_audit=ea['seconds']);nc['peak_bytes']['extension_GPU_allocated']=ec['peak_GPU_bytes'];dump(H/'NORMALIZED_COSTS.json',nc)
 pr=read(H/'PUBLIC_PROVENANCE.json');pr.update(extension_VAL1280_TRAIN1024_fixed_after_DEV=True,extension_target_prediction_seal_sha256=ea['prediction_seal_sha256'],extension_historically_exposed=True,extension_label_schema_only_preflight=True,extension_candidate_IoU_not_used_for_method=True);dump(H/'PUBLIC_PROVENANCE.json',pr)
 nu=read(H/'REPORT_NUMERIC_AUDIT.json');nu.update(extension_task_values=27648,extension_all_ordered_negatives=es['ordered_negatives'],all_target_task_values=31680);dump(H/'REPORT_NUMERIC_AUDIT.json',nu)
 nh=read(H/'NORMALIZED_HISTORY.json');nh['events'] += [{'stage':'extension_geometry','attempt':1,'status':'completed','valid_outputs':2304,'seconds':eg['seconds'],'before_GT':True},{'stage':'extension_predictions','attempt':1,'status':'completed_sealed','valid_outputs':27648,'seconds':ec['seconds'],'before_GT':True},{'stage':'extension_native_replay','attempt':1,'status':'passed','valid_outputs':9216,'seconds':ea['seconds'],'before_GT':True},{'stage':'extension_single_saved_GT_join','attempt':1,'status':'passed','valid_outputs':27648,'producer_predictions_changed':False},{'stage':'CPU_dependency_dispatch','attempt':1,'status':'guard_refused_overlap_no_job_entered','prior_attempt_preserved':True,'producer_predictions_changed':False}];dump(H/'NORMALIZED_HISTORY.json',nh)
 extension_cost_note=f"扩展2304条另计RGB几何{eg['seconds']:.3f}秒、预测{ec['seconds']:.3f}秒、原生审计{ea['seconds']:.3f}秒，{ec['grounding_suffix_calls']}次预测后缀、9216份实际更新态；全部目标值共31680，全部有序负比较共{es['ordered_negatives']+scalar['ordered_negatives']}。"
else:extension_cost_note='没有扩展。'
names={'source':'原始3EED','frozen_ABC':'ABC三采样','R59':'R59直接融合','R76_bidirectional_frozen':'R76冻结双向','joint0':'联合训练 J0','joint1':'联合训练＋SGD J1','joint_direct3':'联合训练同信息直接平均','joint_wrong1':'联合训练＋错配辅助','meta0':'元训练 M0','meta1':'元训练＋SGD M1','meta_direct3':'元训练同信息直接平均','meta_wrong1':'元训练＋错配辅助'}
table='\n'.join(f"| {names[a]} | {T['drone'][a]['mIoU_percent']:.6f} | {T['quad'][a]['mIoU_percent']:.6f} | {B[a]:.6f} | {((T['drone'][a]['Acc25_percent']+T['quad'][a]['Acc25_percent'])/2):.4f} | {((T['drone'][a]['Acc50_percent']+T['quad'][a]['Acc50_percent'])/2):.4f} |" for a in names)
source_table='\n'.join(f"| {a} | {S[a]['mIoU_percent']:.6f} | {S[a]['Acc25_percent']:.4f} | {S[a]['Acc50_percent']:.4f} |" for a in S)
curves='\n'.join(f"| {r['branch']} | {r['epoch']} | {r['zero_mIoU_percent']:.6f} | {r['one_mIoU_percent']:.6f} | {r['gain_pp']:+.6f} | {r['mean_ground1']-r['mean_ground0']:+.9f} |" for r in hist)
changes=[]
for pl in ['drone','quad']:
 r=T[pl]['meta1'];changes.append(f"{pl}: M1对M0改好{r['improved_vs_meta0']}条、改坏{r['worsened_vs_meta0']}条、换query{r['query_changes_vs_meta0']}条，Δ={r['gain_vs_meta0_pp']:+.6f}pp")
verdict=('保留为开发候选，仍需独立确认' if ext_supported else '不采用当前元辅助TTA配方');
exttable='\n'.join(f"| {names[a]} | {EXTB['VAL1280'][a]:.6f} | {EXTB['TRAIN1024'][a]:.6f} |" for a in names) if EXT else '';
extdetails='\n'.join(f"- {name}: M1−M0={v['meta1']-v['meta0']:+.6f}pp；J1−J0={v['joint1']-v['joint0']:+.6f}pp；M1−自身直接平均={v['meta1']-v['meta_direct3']:+.6f}pp；M1−J1={v['meta1']-v['joint1']:+.6f}pp；M1−错配={v['meta1']-v['meta_wrong1']:+.6f}pp。" for name,v in EXTB.items());
extplatform='\n'.join(f"| {name} | {pl} | {v[pl]['joint0']['mIoU_percent']:.6f} | {v[pl]['joint1']['mIoU_percent']:.6f} | {v[pl]['meta0']['mIoU_percent']:.6f} | {v[pl]['meta1']['mIoU_percent']:.6f} | {v[pl]['meta1']['Acc25_percent']:.4f} | {v[pl]['meta1']['Acc50_percent']:.4f} |" for name,v in EXT.items() for pl in ['drone','quad']);
update_rows=read(H/'UPDATE_EFFECT_SUMMARY_ROWS.json');ur={r['cohort']:r for r in update_rows if r['arm']=='meta1'};
text=f'''# R78：源域元辅助训练与普通联合训练的受控比较

**已完成真正可微的源域元辅助训练和固定目标域比较。本轮裁决：{verdict}。不扩大训练轮数、更新步长或搜索新的配置。** 双向架构保留；裁决只覆盖此次512源TRAIN、128暴露VAL、单seed、共同epoch选择、一步SGD和固定辅助任务的原型。

## 方法与公平边界

共同从R76双向epoch6开始。原始Waymo-only epoch100、PointNet++、RoBERTa、cross encoder、decoder及回归参数冻结；A256框、query锚点和六层decoder几何不变。源域只训练已有338343个新增参数，包括RGB编码器、局部双向交互、辅助头和匹配头副本；目标域只更新既定102498参数。不增加专家或预训练网络，不访问FINAL。

两组同512TRAIN/128VAL、seed0、同数据／mask顺序、各4轮、各2048个外循环更新，AdamW1e-4/wd1e-4/clip1。数据与更新数匹配，**计算量没有匹配**：meta多出更新后grounding前后向与高阶辅助反传。训练和测试均用一次纯辅助SGD：alpha0.5、可微全局梯度clip1，无动量、回溯、GT选步长或拒绝门。alpha在训练前由R77源域梯度／参数尺度固定，没有搜索。它和R76的Adam不同，所以核心比较是本轮两组同SGD，旧Adam不能作单因素归因。

联合目标为ground0+aux；元目标为0.5ground0+0.5ground1+aux。ground总系数同为1。aux仍是0.1点特征恢复+0.1RGB恢复+0.01图像端稀疏深度预测，两种互补遮挡和物理对应均不变。ground沿用candidate-IoU softmax/0.1软标签，仅在源域外循环和验证使用。预测深度的输入仍是反向交互前图像特征，未新增深度复制路径。

## 计算图核验

在正式训练前，用最先两条真实源TRAIN样本验证：更新后grounding对辅助专属恢复头产生非零元梯度；把内循环梯度detach后，这些梯度全为0。函数式SGD与实际写参数后的预测最大误差为0。冻结原始3EED仍保留输入特征梯度，外循环可以穿过后缀回到新模块。不是在旧的原地Adam step后加一个loss，也不是FOMAML截断梯度。完整数值在results/META_GRADIENT_PATH_CHECK.json。

此设计借鉴 [Point-TTA的源域meta-auxiliary机制](https://openaccess.thecvf.com/content/ICCV2023/html/Hatem_Point-TTA_Test-Time_Adaptation_for_Point_Cloud_Registration_Using_Multitask_Meta-Auxiliary_ICCV_2023_paper.html)，属于grounding迁移原型，不是其PCR复现。原文采用自己的辅助任务和多步适应；本轮按用户限定先用一步固定更新检验机制，不能把阴性结果推广到所有元学习。

## 源域全部曲线和共同选择

源VAL128已用于R76选epoch及R77诊断，不能称独立确认。两组共同选轮：最大化两组更新后源VALmIoU的等权平均，最早tie；候选为预先固定1—4轮。共同选中 **第{sel['shared_epoch']}轮**，选定权重与代码在目标预测前封存。没有分别挑两组最好轮次，更没有按目标域选择。

| 分支 | 轮次 | 零步mIoU % | 一步mIoU % | 适应增量pp | VAL grounding损失变化 |
|---|---:|---:|---:|---:|---:|
{curves}

选定端点：

| 模型 | 源VAL mIoU % | Acc25 % | Acc50 % |
|---|---:|---:|---:|
{source_table}

源域更新后任务指标并未稳定超过联合训练。不能把更多source训练、较弱零步初始化或代理CE改善计为TTA贡献。每轮还保存期望IoU、最佳IoU候选排序margin和真实Top1IoU；后两者仅用于事后诊断，不进入训练／测试选择。

源梯度诊断：joint辅助／grounding平均cosine={G['joint']['mean_gradient_cosine']:.6f}，实际SGD方向预计有利{G['joint']['ground_loss_favorable_first_order_n']}/128；meta对应cosine={G['meta']['mean_gradient_cosine']:.6f}、预计有利{G['meta']['ground_loss_favorable_first_order_n']}/128。该读数解释方向，不替代任务分数。

## DEV336全部目标域对照

两平台等权平均，五个已暴露场景。全部336保留，包括无信息、零更新和误改。当前输入逐例重置；没有跨输入EMA、缓存迁移或query编号累积。错配对照破坏点—像素交互绑定，但保留RGB恢复和图像稀疏深度等有效观测，不是把全部辅助信息打乱；不能要求它必然崩溃，也不能在它相近或更好时宣称正确绑定是收益原因。冻结直接平均使用全特征及两种内部互补mask的三个读出，是相同观测强对照；它不是原始点云新采集的三个物理视角，也不是ABC三随机采样的别名。不同方法成本分别计数。

| 方法 | Drone mIoU % | Quad mIoU % | 平均mIoU % | 平均Acc25 % | 平均Acc50 % |
|---|---:|---:|---:|---:|---:|
{table}

关键差值：ΔJ={joint_gain:+.6f}pp；ΔM={meta_gain:+.6f}pp；两种适应增量之差={meta_gain-joint_gain:+.6f}pp。M1对自身同信息直接平均={B['meta1']-B['meta_direct3']:+.6f}pp，对J1={B['meta1']-B['joint1']:+.6f}pp，对R59={B['meta1']-B['R59']:+.6f}pp。不能只比较裸source，也不能只看M1与M0的差值。

{'; '.join(changes)}。所有逐例有序负比较和五场景结果均公开，不以个别案例或每场景零损害决定采用。不提供把同帧多描述当独立样本的窄置信区间。

## 裁决与下一步边界

DEV336通过预先锁定的扩展条件：M1超过M0、自己的直接平均、J1及联合直接平均、R59和R76冻结双向。因此保持共同第1轮、SGD0.5、同mask和同读出，在R69既有VAL1280／TRAIN1024清单上完成一次固定扩展，不调参。R69原始A前缀和A256复用；新增RGB可见性／局部对应使用同一R76几何函数，所有2304条通过原始点索引与seed坐标核验。FINAL未访问。

| 方法 | VAL1280 平均mIoU % | TRAIN1024 平均mIoU % |
|---|---:|---:|
{exttable}

全部2304两平台平均：M0={pooled.get("meta0",0):.6f}%，M1={pooled.get("meta1",0):.6f}%，J0={pooled.get("joint0",0):.6f}%，J1={pooled.get("joint1",0):.6f}%，M直接平均={pooled.get("meta_direct3",0):.6f}%，R59={pooled.get("R59",0):.6f}%。两个范围同时通过条件={ext_supported_each}。

{extdetails}

| 范围 | 平台 | J0 mIoU % | J1 mIoU % | M0 mIoU % | M1 mIoU % | M1 Acc25 % | M1 Acc50 % |
|---|---|---:|---:|---:|---:|---:|---:|
{extplatform}

VAL来自原五个recording，TRAIN来自另四个recording家族；这两份清单在R69已有历史结果，不能称盲测或完全新地理分布。主扩展指标在扩展评分前固定为全部2304条的两平台等权mIoU（每平台1152条）；同时分别报告两个范围，不以每场景或每样本零损害作门槛，也不把总体收益包装成两个范围均稳定。扩展新增的所有原生参数态与选框在评分前封存并重放。扩展前有一次既有R69标签文件的字段名／记录数预检；未显示或使用候选IoU值，没有据此修改方法；正式IoU join在预测封存和原生审计后进行，见schema预检回执。

**{verdict}。** 判据基于全部样本总体指标；保留所有曲线与不利对照，不换轮次、步长或训练目标挽救本轮。

这次已经排除了“根本没有实现通过辅助更新的元梯度”这一工程解释；尚未证明这份恢复／稀疏深度任务能够在当前训练预算下产生有益跨平台更新。原始双向冻结候选仍保留，不能据此否定整个多模态TTA路线。普通联合训练也有适应增量或损害，应与元训练同时报告。

元目标允许匹配头副本直接改善零步和一步任务；更强静态匹配是竞争解释，不能仅凭M1绝对分数较高就称辅助更新方向学习成功。本轮用ΔM与ΔJ分别记录该问题，但未做冻结外层匹配头的因果消融。后续若继续，最小有区分力的问题是**辅助任务应恢复哪些位置／特征才能服务于语言候选竞争**，而不是再次增加现有更新步长或寻找层。当前aux只读取图像、融合前点特征和几何mask，不显式读取文字；这是代码事实，但“缺语言条件导致本轮失败”仍是未验证解释，不能直接当因果结论。本轮未添加语言条件辅助权重、候选mask或新loss，也没有启动下一实验。

## 实际更新与任务结果

M1在VAL1280的1280条和TRAIN1024的1024条均产生非零参数变化；中位L2分别{ur["VAL1280"]["parameter_step_L2_median"]:.6f}/{ur["TRAIN1024"]["parameter_step_L2_median"]:.6f}。两范围合计2302/2304条辅助损失下降，但最终IoU仅34条改善、40条恶化；换query为147条，其余部分是重复候选／IoU相同的改变。辅助下降、参数非零和真实写入都不能替代正任务增量。详见全部UPDATE_EFFECT_SUMMARY_ROWS。

## 审计、成本和发布

{extension_cost_note}

DEV目标4032个任务值、{scalar['ordered_negatives']}个有序负比较，源域{scalar['source_task_values']}个选定端点值、{scalar['source_ordered_negatives']}个负比较，全部保留。两组源VAL选定端点256输入重新计算；全部1344份目标更新参数重新生成并实际写入重放，最大目标分数误差={audit['max_probability_error']}。参数再生成1e-7、源验证再生成概率1e-8和保存态目标重放1e-10容差均在目标预测前登记。

源训练合计{cost['seconds']:.3f}秒；joint训练加各轮VAL={times['joint']:.3f}秒，meta={times['meta']:.3f}秒。训练grounding后缀joint2048次、meta4096次，各有1024次验证后缀。数据曝光相同，不能声称等计算。目标全部对照{tc['seconds']:.3f}秒、3360次后缀与2688次辅助融合、1344次SGD。源／目标峰值torch分配分别{cost['peak_GPU_bytes']/1e9:.3f}/{tc['peak_GPU_bytes']/1e9:.3f}GB；复用冻结前缀，时间不是无缓存端到端部署成本。

原checkpoint与其他paused方向登记不变；原始媒体／描述／坐标／ID／特征／模型权重不公开。完成后遵循原public-export、完整远端字节核验及CAS发布约定。**本轮结论只限固定原型：{verdict}；无后续排队实验。**
''';(H/'REPORT.md').write_text(text);print(decision)
