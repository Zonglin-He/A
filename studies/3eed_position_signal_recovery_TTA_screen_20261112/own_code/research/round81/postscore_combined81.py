"""Separate failed primary selection from the prespecified full-fusion secondary cue."""
from core81 import *
d=read(H81/'DECISION.json');assert read(H81/'independent_signal_control_scalar_audit_attempt001/AUDIT_RESULT.json')['state'].startswith('PASS_');control=read(H81/'SIGNAL_CONTROL_SUMMARY.json');results={}
for part,n in [('VAL1280',1280),('TRAIN1024',1024)]:
 s=read(H81/('score_'+part.lower()+'_attempt001/TASK_SUMMARY.json'))['ALL'+str(n)];b=s['balanced_mIoU_percent'];c=control[part]['balanced_mIoU_percent'];assert c['all__ALL']==b['all__ALL'];results[part]={'mIoU_percent':b['all__ALL'],'gain_vs_frozen_pp':b['all__ALL']-b['frozen_bi'],'gain_vs_direct3_pp':b['all__ALL']-b['frozen_direct3'],'gain_vs_raw_R80_pp':b['all__ALL']-b['R80_uniform_raw'],'gain_vs_P_only_all_scope_pp':b['all__ALL']-b['all__P'],'gain_vs_shuffled_target_pp':b['all__ALL']-c['all__ALL__shuffle'],'shuffle_gain_vs_direct3_pp':c['all__ALL__shuffle']-b['frozen_direct3'],'improved_vs_frozen':sum(s[p]['all__ALL']['improved_vs_frozen_bi'] for p in ['drone','quad']),'worsened_vs_frozen':sum(s[p]['all__ALL']['worsened_vs_frozen_bi'] for p in ['drone','quad']),'query_changes_vs_frozen':sum(s[p]['all__ALL']['query_changes_vs_frozen_bi'] for p in ['drone','quad'])}
task=all(r['gain_vs_frozen_pp']>0 and r['gain_vs_direct3_pp']>0 for r in results.values());binding=all(r['gain_vs_shuffled_target_pp']>0 for r in results.values());candidate={'state':'R81_WEAK_FULL_FUSION_COMBINED_RECOVERY_CANDIDATE','parameters':102498,'update_modules':['point_adapter','image_adapter','point_to_image','image_to_point','gate_I','gate_P'],'frozen_modules':['original_3EED_backbones_cross_encoder_decoder','RGB_CNN','matching_and_reconstruction_heads','original_A256_boxes'],'signal':'.1_masked_frozen_point_feature_recovery_plus_.1_RGB_patch_recovery_plus_.01_sparse_image_only_log_depth','update':'one_SGD_direction_clip1_Armijo_alpha.5_halved_to.0078125;selected_auxiliary_plus_source_output_query_keep;episodic_source_reset','was_prespecified_secondary_comparator':True,'not_the_DEV_selected_primary':True,'signal_shuffle_control_added_postscore':True,'small_positive_vs_frozen_and_same_information_direct_both_cohorts':task,'correct_target_advantage_both_cohorts':binding,'independent_confirmation':False,'practical_large_gain_established':False,'results':results,'no_new_source_training_or_external_expert':True,'FINAL_access':False};dump(H81/'EXPLORATORY_FULL_FUSION_CANDIDATE.json',candidate)
d.update(exploratory_full_combined_backtracking=candidate,retain_full_combined_as_weak_development_candidate=task and binding,do_not_relabel_failed_primary_as_success=True,adopt_as_robust_method=False);dump(H81/'DECISION.json',d)
cost=read(H81/'NORMALIZED_COSTS.json');c=read(B81/'signal_control2304_attempt001/COUNTS.json');au=read(H81/'independent_signal_control_audit_attempt001/AUDIT_RESULT.json');cost['counts'].update(postscore_signal_control_existing_inputs=2304,postscore_signal_control_new_states=2304,postscore_signal_control_proposals=c['attempted_proposals']);cost['timing_seconds'].update(signal_control_producer=c['seconds'],signal_control_audit=au['seconds']);cost['peak_bytes']['signal_control_GPU']=c['peak_GPU_bytes'];dump(H81/'NORMALIZED_COSTS.json',cost)
h=read(H81/'NORMALIZED_HISTORY.json');h['events'].append({'stage':'postscore_full_signal_shuffle_control','attempt':1,'status':'passed','valid_outputs':2304,'producer_predictions_changed':False});dump(H81/'NORMALIZED_HISTORY.json',h)
provenance=read(H81/'PUBLIC_PROVENANCE.json');provenance.update(one_full_signal_control_added_after_primary_scoring=True,original_method_and_selected_primary_unchanged=True,full_combined_method_prespecified_before_extension=True,secondary_positive_is_not_independent_confirmation=True);dump(H81/'PUBLIC_PROVENANCE.json',provenance);num=read(H81/'REPORT_NUMERIC_AUDIT.json');num['exploratory_full_combined_results']=results;dump(H81/'REPORT_NUMERIC_AUDIT.json',num)
lead=f'''\n**结果：开发选中的point_to_image/P位置没有在扩展数据上超过直接平均。预设全融合/三信号/回溯对照有小幅同向增量；补充的打乱目标控制低于正确恢复。该配方可保留为弱开发候选，但不是已找到幅度充分、独立确认的提升。**\n\n''';p=H81/'REPORT.md';text=p.read_text();head,rest=text.split('\n',1);p.write_text(head+'\n'+lead+rest)
table='\n'.join(f"| {p} | {r['mIoU_percent']:.6f} | {r['gain_vs_frozen_pp']:+.6f} | {r['gain_vs_direct3_pp']:+.6f} | {r['gain_vs_shuffled_target_pp']:+.6f} | {r['gain_vs_raw_R80_pp']:+.6f} |" for p,r in results.items())
add=f'''

## 一个弱但可部署的次要信号：全融合联合恢复＋无标签回溯

这不是将DEV选型失败后悄悄换成赢家。全模块/联合恢复是扩展前预设的对照，位置、损失、步长规则均未改变；读出扩展结果后，只新增了该对照的打乱目标分支，以检验信号对应。锁、代码、全部失败primary结果与新控制均保留。新控制不使用新的输入范围、不改超参，但属于读分后的机制诊断，不能当作独立确认。

| 范围 | 正确联合恢复 mIoU % | 对冻结 pp | 对同信息平均 pp | 对打乱恢复目标 pp | 对R80原大步 pp |
|---|---:|---:|---:|---:|---:|
{table}

最有支持的位置仍是整个已准备的轻量融合接口：两个Adapter、两条局部交互的Q/K/V/O及gate_I/P，共102498参数。恢复头/匹配头、RGB CNN、原3EED骨干和decoder、原候选框保持不变。信号为冻结点表征恢复＋实测RGB patch均值恢复＋实测稀疏深度恢复，无目标身份伪标签、无外部专家。联合恢复比P-only全模块在两个扩展范围略高，但并未证明三个分量每个都不可替代。

正确目标更新在两个范围均略高于直接平均，打乱目标更新在两个范围均低于直接平均。这比“任意更新都能涨一点”更有信息，但更新方向、误差尺度和回溯接受相互作用，不能仅凭这一控制断言语义实例绑定已经修好。对直接平均的增量只有0.026615pp和0.009612pp，不能包装成大幅或稳定统计显著收益。

本轮回答分层：确实存在可影响选框、具有无标签梯度路径的融合位置；观测恢复在原R76上提供一个很弱的任务信号；未找到比全模块联合恢复更有力且跨范围重复的单一路径/单一信号。尤其不能将GT能力诊断的收益等同于无标签监督有效，也不能把辅助误差下降当定位收益。原CNN/decoder其他可更新方案没有在本轮穷尽；禁止据此推断所有TTA位置无用。

此次补充只新增2304份预测与任务值，副表复用6912个旧参照值，独立输入数仍为2640，不将重复引用当额外证据。所有新控制参数态、最终目标和读出已重算，目标置换保留每个有效集合的多重集合。无进一步模型、搜索或源训练排队。
'''
with p.open('a') as f:f.write(add)
print(candidate)
