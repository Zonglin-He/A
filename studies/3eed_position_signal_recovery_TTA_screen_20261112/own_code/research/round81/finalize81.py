"""Answer location/signal efficacy from audited DEV selection and locked extension."""
from core81 import *
sel=read(H81/'DEV_SELECTION.json');audit=read(H81/'independent_scalar_audit_attempt001/AUDIT_RESULT.json');assert audit['state'].startswith('PASS_');parts=[('DEV336',336)]+([('VAL1280',1280),('TRAIN1024',1024)] if sel['eligible'] else []);sums={p:read(H81/('score_'+p.lower()+'_attempt001/TASK_SUMMARY.json'))['ALL'+str(n)] for p,n in parts};B={p:s['balanced_mIoU_percent'] for p,s in sums.items()};chosen=sel['selected'];shuffle=chosen+'__shuffle';same='all__'+sel['signal'] if sel['signal']!='D' else chosen;gains={}
for p,b in B.items():
 gains[p]={ref:b[chosen]-b[ref] for ref in ['frozen_bi','frozen_direct3','R80_uniform_raw','all__ALL',same] if ref in b}
 if shuffle in b:gains[p]['shuffled_target_control']=b[chosen]-b[shuffle]
repeats=sel['eligible'] and all(gains[p]['frozen_bi']>0 and gains[p]['frozen_direct3']>0 for p in ['VAL1280','TRAIN1024']);signal_support=sel['eligible'] and all(gains[p]['shuffled_target_control']>0 for p in ['VAL1280','TRAIN1024']);scope_support=sel['eligible'] and same!=chosen and all(gains[p][same]>0 for p in ['VAL1280','TRAIN1024']);decision={'state':'R81_COMPLETED_BOUNDED_POSITION_SIGNAL_SCREEN','selected_position':sel['scope'],'selected_signal':sel['signal'],'selected_parameters':sel['parameters'],'selection_used_DEV_labels_only':True,'per_input_TTA_uses_task_GT':False,'repeatable_self_adaptation_increment_both_fixed_extension_cohorts':repeats,'correct_signal_target_binding_advantage_both_cohorts':signal_support,'selected_scope_advantage_over_same_signal_all_parameters_both_cohorts':scope_support,'retain_as_fixed_extension_development_candidate':repeats,'robust_independent_confirmation':False,'gains_pp':gains,'source_training':False,'external_models':False,'FINAL_access':False,'no_follow_on_queued':True};dump(H81/'DECISION.json',decision)
names={'source':'原始3EED','frozen_bi':'R76冻结双向','frozen_direct3':'同信息冻结直接平均','R80_uniform_raw':'R80均匀恢复原固定一步','frozen_ABC':'ABC三采样','R59':'R59直接融合'}
for s,t in CONFIGS:names[s+'__'+t]=s+' / '+t
names[shuffle]=sel['scope']+' / '+sel['signal']+'（打乱目标位置）';tables=[]
for p,s in sums.items():
 arms=list(B[p]);body='\n'.join(f"| {names.get(a,a)} | {s['drone'][a]['mIoU_percent']:.6f} | {s['quad'][a]['mIoU_percent']:.6f} | {B[p][a]:.6f} | {(s['drone'][a]['Acc25_percent']+s['quad'][a]['Acc25_percent'])/2:.4f} | {(s['drone'][a]['Acc50_percent']+s['quad'][a]['Acc50_percent'])/2:.4f} |" for a in arms);tables.append('### '+p+'\n\n| 配置 | Drone mIoU % | Quad mIoU % | 等权平均 % | Acc25 % | Acc50 % |\n|---|---:|---:|---:|---:|---:|\n'+body)
stats=read(H81/'UPDATE_SUMMARY_ROWS.json');effects=[]
for p,n in parts:
 tasks=read(H81/('score_'+p.lower()+'_attempt001/ALL_TASK_ROWS_PRIVATE.json'));s=sums[p]
 for pl in ['drone','quad']:
  rr=[r for r in tasks if r['platform']==pl];v=np.array([r['IoU'][chosen]-r['IoU']['frozen_bi'] for r in rr]);effects.append({'cohort':p,'platform':pl,'n':len(rr),'improved':int((v>1e-10).sum()),'worsened':int((v<-1e-10).sum()),'gain_vs_frozen_pp':float(v.mean()*100),'query_changes':sum(r['queries'][chosen]!=r['queries']['frozen_bi'] for r in rr),'largest_single_positive_contribution_balanced_pp':float(v.max()*50/len(rr)),'largest_single_negative_contribution_balanced_pp':float(v.min()*50/len(rr))})
dump(H81/'SELECTED_TASK_EFFECT_ROWS.json',effects)
devcost=read(B81/'dev336_attempt001/COUNTS.json');costs={'counts':{'DEV_inputs':336,'DEV_unique_configs':11,'DEV_states':devcost['states'],'DEV_proposals':devcost['attempted_proposals'],'DEV_accepted_updates':devcost['accepted_updates'],'source_training_updates':0,'task_values':audit['task_values'],'ordered_negatives':audit['ordered_negatives']},'timing_seconds':{'DEV_producer':devcost['seconds'],'DEV_native_audit':read(H81/'independent_dev_native_audit_attempt001/AUDIT_RESULT.json')['seconds']},'peak_bytes':{'DEV_GPU':devcost['peak_GPU_bytes']},'flags':{'source_training':False,'FINAL_access':False,'cached_frozen_prefixes':True,'end_to_end_per_method_latency':False}}
if sel['eligible']:
 c=read(B81/'extension2304_attempt001/COUNTS.json');costs['counts'].update(extension_inputs=2304,extension_states=c['states'],extension_proposals=c['attempted_proposals'],extension_accepted_updates=c['accepted_updates']);costs['timing_seconds'].update(extension_producer=c['seconds'],extension_native_audit=read(H81/'independent_extension_native_audit_attempt001/AUDIT_RESULT.json')['seconds']);costs['peak_bytes']['extension_GPU']=c['peak_GPU_bytes']
dump(H81/'NORMALIZED_COSTS.json',costs);dump(H81/'NORMALIZED_HISTORY.json',{'events':[{'stage':'engineering','attempt':1,'status':'passed','valid_outputs':2,'before_GT':True},{'stage':'DEV_predictions','attempt':1,'status':'passed','valid_outputs':336,'before_GT':True},{'stage':'DEV_native_audit','attempt':1,'status':'passed','valid_outputs':3696,'before_GT':True},{'stage':'global_DEV_configuration_selection','attempt':1,'status':'passed','valid_outputs':1,'producer_predictions_changed':False}]+([{'stage':'locked_extension_predictions_and_audit','attempt':1,'status':'passed','valid_outputs':2304,'before_GT':True}] if sel['eligible'] else [])})
plan=read(H81/'experiment/DESIGN_LOCK.json');plan.pop('inputs_sha256');plan.pop('source_plan');plan.pop('init_state');dump(H81/'PUBLIC_FIXED_CONFIGURATION.json',plan);dump(H81/'PUBLIC_PROVENANCE.json',{'state':'R81_GRAPH_CONNECTED_POSITION_SIGNAL_AND_LABEL_FREE_LINESEARCH','source_and_prepared_model_unchanged':True,'original_A256_unchanged':True,'DEV_global_selection_label_assisted':True,'no_GT_in_input_update_or_acceptance':True,'all_cohorts_historically_exposed':True,'blind_test_claim':False,'third_mask_is_same_frame_and_target_positions_can_overlap':True,'depth_target_positions_not_independent_holdout_labels':True,'shuffled_control_preserves_effective_per_type_target_multiset':True,'new_source_training':False,'external_models':False,'FINAL_access':False});dump(H81/'REPORT_NUMERIC_AUDIT.json',{'state':'PASS_R81_REPORT_USES_AUDITED_SAVED_VALUES','gains_pp':gains,'DEV_selected_parameters':sel['parameters'],'task_values':audit['task_values'],'ordered_negatives':audit['ordered_negatives']})
old=read(H81/'experiment/INITIAL_RESEARCH_STATE_PRIVATE.json');now=read(ROOT/'research/STATE.json')
for k in ['avis','affordance']:assert old['directions'][k]==now['directions'][k]
assert sha(ROOT/read(H81/'experiment/DESIGN_LOCK.json')['init_state'])==read(H81/'experiment/DESIGN_LOCK.json')['init_state_sha256'];assert sha(ROOT/'crossplatform/round6/waymo_source100/checkpoints/epoch100_model_attempt001.pth')=='0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd'
report=f'''# R81：更新位置与恢复信号联合筛查

目标是找到实际能提高grounding的位置与信号，而非仅证明参数能更新。固定原R76双向e6和原始epoch100/A256，现有恢复头与匹配头/RGB CNN/骨干不变，不源训、不外加专家、不访问FINAL。

## 计算图与有边界的候选

五个位置为image_adapter(2128)、point_adapter(9520)、point_to_image+gate_I(45313)、image_to_point+gate_P(45537)、全部原TTA参数(102498)。单信号P为冻结点特征前64维的遮挡恢复(系数.1)，RGB为实测patch均值(.1)，D为图像侧实测稀疏logdepth(.01)。P不是物体GT或真实语义标签，而是原生冻结表示目标。

D只对允许参数里的image_adapter有梯度，all/D与之完全等价；RGB对image_to_point/gate_P没有梯度。工程真实Drone/Quad检查证实这些结构性零路径。原物理辅助头也不调用末端匹配MLP，不能用这份损失直接更新该MLP。排除无梯度/等价项后，P五位置、RGB四位置、D一个位置、联合信号全位置，共11个组合。不是所有可能模块/损失/强度的穷尽搜索，也未扫描decoder层/head。

在source状态分别计算三类初始辅助梯度，复用于各位置组合；每个组合固定一个按位置截取的SGD方向，clip1，回溯试探不重算梯度。最大alpha.5，依次减半到.0078125，首个满足Armijo(c=1e-4,tol1e-10)的提案接受，否则source回退。接受目标为选中恢复项＋原完整输入分布KL/query保持。保持项起点梯度0，但参与试探后的接受，因此梯度来自恢复，文字决策通过接受约束影响步长；不能将其当作独立的新语义标签。不是按GT调步长，也没有多步训练。实际改参数、重放原网络、恢复source，全部负例及回退均计入分母。

## DEV选型与锁定扩展

DEV所有11组合先预测封存，再独立检查原生辅助分量梯度、重建参数和全部接受轨迹/实际读出，才join已有A256候选IoU。DEV全局等权mIoU选一个配置，同分优先少参数；不按平台/样本选。开发标签参与这一次全局选型，不能称该配置由无标签自动发现。

选中 {chosen}，{sel['parameters']}参数，DEV {sel['DEV_balanced_mIoU_percent']:.6f}%，对自身冻结{sel['DEV_gain_vs_frozen_pp']:+.6f}pp、对直接平均{sel['DEV_gain_vs_direct3_pp']:+.6f}pp，对原R80固定大步{sel['DEV_gain_vs_raw_R80_pp']:+.6f}pp。R80原方法即使DEV更高也完整保留，不把它隐藏。选型完成后锁定VAL1280/TRAIN1024，加入同信号全位置、三信号全位置回溯及选中信号打乱目标位置的控制。后者在每个有效监督集合内部置换，保留目标多重集合和数量；不打乱文本、不制造目标身份标签。控制最后仍用正确输入/标定读出。

## 全体任务结果

'''+ '\n\n'.join(tables)+f'''

## 判断分开给出

选中组合在两个扩展范围均优于自己的冻结及同信息直接平均：{repeats}。正确目标绑定在两个范围均优于打乱目标：{signal_support}。选中位置相对同信号全参数均有优势：{scope_support}。这些是不同主张，不能由其中一个代替另外两个。每个平台、recording、纠正/损伤及最大单例贡献见完整标量表；不要求每场景上涨，不把同场景帧当独立场景置信证据。

第三遮挡76003只用于更新后的恢复读数，不参与梯度、步长或配置选择。它来自同一物理帧，目标位置可与训练遮挡重叠；深度仍在原valid集合评价，所以不是新视角，也不应称独立未见标签测试。辅助改善只解释行为，不决定grounding有效性。

R80原均匀固定步、ABC、R59是已完成同checkpoint输入的历史参照，明确复用，不重新跑以冒充新增结果；选中组合及机制控制实际重算。全部DEV/VAL/TRAIN已有历史曝光，扩展为固定配置开发评价，未使用FINAL，不宣称盲测确认。物理恢复损失无需目标身份标签，不等于其梯度天然有利于语言定位；位置/信号在本系数/接受规则下有效，也不等于脱离更新幅度与保持约束后具有内在最优性。

## 完成与资源边界

仅本次11组合和至多一个锁定候选的扩展，无更多位置/alpha网格、源域训练或新专家排队。全部负比较{audit['ordered_negatives']}条及{audit['task_values']}个任务标量保留（含后补控制的新2304值；副表重复引用旧控制不另加证据量）。实际写入参数的结果均独立重放，非选中参数保持source。DEV完整重放全部搜索试探；扩展重算全部最终状态、最终目标与读出，拒绝试探检查保存的Armijo界/首个接受关系，未逐一重跑其前向损失，两种覆盖分别公开。模型前缀缓存复用，耗时为整组条件运行，不是端到端单方法时延。私有媒体/描述/坐标/ID/特征/梯度/参数文件不公开；公开代码、协议和匿名标量。原checkpoint及R76e6 hash不变，其他paused项目登记不变。
''';(H81/'REPORT.md').write_text(report);print(decision)
