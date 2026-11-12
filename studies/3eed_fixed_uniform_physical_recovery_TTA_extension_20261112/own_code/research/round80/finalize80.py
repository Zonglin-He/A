"""Measured fixed-method decisions; conditional ablation gate is explicit."""
from core80 import *
pin(read(H80/'experiment/SCORE_CODE_LOCK.json'))
audit=read(H80/'independent_scalar_audit_attempt001/AUDIT_RESULT.json');assert audit['state'].startswith('PASS_');sums={part:read(H80/('score_'+part.lower()+'_attempt001/TASK_SUMMARY.json'))['ALL'+str(n)] for part,n in [('DEV336',336),('VAL1280',1280),('TRAIN1024',1024)]};B={p:s['balanced_mIoU_percent'] for p,s in sums.items()};gain={p:{'vs_frozen_pp':b['uniform_sgd1']-b['frozen_bi'],'vs_direct3_pp':b['uniform_sgd1']-b['frozen_direct3'],'vs_uniform_wrong_pp':b['uniform_sgd1']-b['uniform_wrong_sgd1']} for p,b in B.items()};gate=all(gain[p][ref]>0 for p in ['VAL1280','TRAIN1024'] for ref in ['vs_frozen_pp','vs_direct3_pp']);dump(H80/'COMPLEMENTARY_ABLATION_GATE.json',{'state':'R80_PREREGISTERED_REPEATABLE_SELF_ADAPTATION_GATE','passed':gate,'cohort_gains':gain,'not_required_to_beat_R59_or_all_internal_systems':True})
if gate:raise SystemExit('R80_PREREGISTERED_COMPLEMENTARY_ABLATION_REQUIRED;DO_NOT_CLOSE_OR_TUNE')
old=read(H80/'experiment/INITIAL_RESEARCH_STATE_PRIVATE.json');current=read(ROOT/'research/STATE.json')
for k in ['avis','affordance']:assert current['directions'][k]==old['directions'][k]
plan=read(H80/'experiment/DESIGN_LOCK.json');assert sha(ROOT/plan['init_state'])==plan['init_state_sha256'];sourceplan=read(ROOT/plan['source_plan']);assert sha(ROOT/'crossplatform/round6/waymo_source100/checkpoints/epoch100_model_attempt001.pth')=='0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd'
decision={'state':'R80_COMPLETED_FIXED_UNIFORM_RECOVERY_EXTENSION','adopt_uniform_recovery_TTA':False,'retain_bidirectional_structure':True,'R79_DEV_positive_retained':True,'repeatable_positive_vs_own_frozen_and_direct_both_cohorts':gate,'cohort_gains':gain,'pooled2304_balanced_mIoU_percent':read(H80/'POOLED2304_SUMMARY.json'),'conditional_complementary_ablation_executed':False,'reason_no_conditional_ablation':'fixed_primary_did_not_repeat_positive_self_adaptation_vs_zero_and_direct3_in_both_extension_cohorts;not_an_R59_rank_gate','no_source_training':True,'no_external_models':True,'original_A256_unchanged':True,'FINAL_access':False,'no_follow_on_queued':True};dump(H80/'DECISION.json',decision)
events=[{'stage':'engineering','attempt':1,'status':'passed','valid_outputs':4,'before_GT':True},{'stage':'predictions','attempt':1,'status':'passed','valid_outputs':2640,'before_GT':True},{'stage':'independent_native_audit','attempt':1,'status':'passed','valid_outputs':5280,'before_GT':True},{'stage':'saved_IoU_join','attempt':1,'status':'passed','valid_outputs':18480,'producer_predictions_changed':False}];dump(H80/'NORMALIZED_HISTORY.json',{'events':events})
cost=read(B80/'prediction2640_attempt001/COUNTS.json');native=read(H80/'independent_native_audit_attempt001/AUDIT_RESULT.json');dump(H80/'NORMALIZED_COSTS.json',{'counts':{'target_inputs':2640,'extension_inputs':2304,'new_updated_states':4944,'reused_R79_uniform_states':336,'states_replayed':5280,'target_values':18480,'ordered_negatives':audit['ordered_negatives'],'grounding_suffix_calls':cost['grounding_suffix_calls'],'auxiliary_parameter_gradient_calls':cost['auxiliary_parameter_gradient_calls'],'source_new_forwards':0,'sensitivity_calls':0},'timing_seconds':{'producer':cost['seconds'],'native_audit':native['seconds']},'peak_bytes':{'producer_GPU':cost['peak_GPU_bytes']},'flags':{'source_training':False,'FINAL_access':False,'preexisting_prefix_cache':True,'not_end_to_end_per_method_timing':True}})
public=dict(plan);public.pop('inputs_sha256');public.pop('source_plan');public.pop('init_state');dump(H80/'PUBLIC_FIXED_CONFIGURATION.json',public)
dump(H80/'PUBLIC_PROVENANCE.json',{'state':'R80_ORIGINAL_R76_E6_NO_SOURCE_TRAINING_FIXED_UNIFORM_TTA','DEV_historical_exposure':True,'VAL1280_historical_exposure':True,'TRAIN1024_historical_exposure':True,'blind_test_claim':False,'R78_model_not_used':True,'R78_prepared_geometry_only_reused':True,'new_extension_predictions':True,'new_DEV_uniform_wrong_predictions':True,'R79_DEV_core_predictions_reused':True,'source128_reference_reused_only':True,'predictions_sealed_before_saved_task_label_join':True,'original_epoch100_and_R76_checkpoint_unchanged':True,'other_paused_registries_unchanged':True,'FINAL_access':False,'prediction_seal_sha256':native['prediction_seal_sha256']})
names={'source':'原始3EED','frozen_bi':'R76冻结双向','frozen_direct3':'同信息冻结直接平均','uniform_sgd1':'均匀恢复SGD一步','uniform_wrong_sgd1':'均匀恢复＋错配绑定SGD一步','frozen_ABC':'ABC三采样','R59':'R59直接表示融合'}
tables=[];scenetables=[];effects=[];contrib=read(H80/'ALL_CONTRIBUTION_ROWS.json');endpoints=read(H80/'ALL_ENDPOINT_SCALAR_ROWS.json');roster=allrows80();partby={r['id']:'DEV336' for r in target_rows()};partby.update(read(ROOT/'research/round69/input/COHORT_BY_ID_PRIVATE.json'));endpointrows=[]
for p in sums:
 s=sums[p];rows='\n'.join(f"| {names[a]} | {s['drone'][a]['mIoU_percent']:.6f} | {s['quad'][a]['mIoU_percent']:.6f} | {B[p][a]:.6f} | {(s['drone'][a]['Acc25_percent']+s['quad'][a]['Acc25_percent'])/2:.4f} | {(s['drone'][a]['Acc50_percent']+s['quad'][a]['Acc50_percent'])/2:.4f} |" for a in ARMS80);tables.append('### '+p+'\n\n| 方法 | Drone mIoU % | Quad mIoU % | 平台等权平均 % | Acc25 % | Acc50 % |\n|---|---:|---:|---:|---:|---:|\n'+rows)
 for pl in ['drone','quad']:
  z=s[pl]['uniform_sgd1'];effects.append({'cohort':p,'platform':pl,'n':z['n'],'gain_vs_frozen_pp':z['gain_vs_frozen_bi_pp'],'improved':z['improved_vs_frozen_bi'],'worsened':z['worsened_vs_frozen_bi'],'query_changes':z['query_changes_vs_frozen_bi'],'gain_vs_direct3_pp':z['gain_vs_frozen_direct3_pp'],'gain_vs_wrong_pp':z['gain_vs_uniform_wrong_sgd1_pp']})
 sr=[x for x in read(H80/'ALL_SCENE_ROWS.json') if x['cohort']==p];scs=[]
 for pl,sc in sorted({(x['platform'],x['scene_alias']) for x in sr}):
  dd={x['arm']:x for x in sr if x['platform']==pl and x['scene_alias']==sc};scs.append(f"| {pl}/{sc} | {dd['frozen_bi']['n']} | {dd['frozen_bi']['mIoU_percent']:.6f} | {dd['uniform_sgd1']['mIoU_percent']:.6f} | {dd['uniform_sgd1']['mIoU_percent']-dd['frozen_bi']['mIoU_percent']:+.6f} | {dd['uniform_sgd1']['mIoU_percent']-dd['uniform_wrong_sgd1']['mIoU_percent']:+.6f} |")
 scenetables.append('### '+p+' recording分组\n\n| 平台/组内别名 | n | 冻结 % | 均匀TTA % | 增量 pp | 正确−错配 pp |\n|---|---:|---:|---:|---:|---:|\n'+'\n'.join(scs))
 rr=[x for x in contrib if x['cohort']==p and x['reference']=='frozen_bi'];best=max(rr,key=lambda x:x['contribution_to_balanced_mIoU_pp']);worst=min(rr,key=lambda x:x['contribution_to_balanced_mIoU_pp']);effects.append({'cohort':p,'platform':'balanced_tail','largest_single_positive_contribution_pp':best['contribution_to_balanced_mIoU_pp'],'largest_single_negative_contribution_pp':worst['contribution_to_balanced_mIoU_pp'],'remaining_gain_after_largest_positive_pp':gain[p]['vs_frozen_pp']-best['contribution_to_balanced_mIoU_pp']})
for p in sums:
 ids={i+1 for i,r in enumerate(roster) if partby[r['id']]==p}
 for a in ['uniform_sgd1','uniform_wrong_sgd1']:
  rr=[r for r in endpoints if int(r['row'][1:]) in ids and r['arm']==a];endpointrows.append({'cohort':p,'arm':a,'n':len(rr),'nonzero_updates':sum(r['parameter_step_L2']>0 for r in rr),'mean_step_L2':float(np.mean([r['parameter_step_L2'] for r in rr])),'median_step_L2':float(np.median([r['parameter_step_L2'] for r in rr])),'auxiliary_before_available_n':sum('auxiliary_before' in r for r in rr),'auxiliary_decrease_n':sum(r['auxiliary_after']<r.get('auxiliary_before',1e99) for r in rr if 'auxiliary_before' in r),'query_changes':sum(r['query_changed'] for r in rr)})
dump(H80/'TASK_EFFECT_AND_TAIL_ROWS.json',effects);dump(H80/'UPDATE_SUMMARY_ROWS.json',endpointrows);source=read(ROOT/'research/round79/SOURCE128_SUMMARY.json');dump(H80/'SOURCE128_REFERENCE_SUMMARY.json',{a:source[a] for a in ['source','frozen_bi','frozen_direct3','uniform_sgd1']});dump(H80/'REPORT_NUMERIC_AUDIT.json',{'state':'PASS_R80_REPORT_USES_AUDITED_ALL_COHORT_VALUES','cohort_gains':gain,'source_training':False,'all_negative_comparisons':audit['ordered_negatives'],'no_method_change_after_scores':True})
effecttext='\n'.join(f"| {r['cohort']} | {r['platform']} | {r['improved']} | {r['worsened']} | {r['query_changes']} | {r['gain_vs_frozen_pp']:+.6f} |" for r in effects if r['platform']!='balanced_tail');tailtext='\n'.join(f"- {r['cohort']}：最大单例正贡献{r['largest_single_positive_contribution_pp']:+.6f}pp，最大负贡献{r['largest_single_negative_contribution_pp']:+.6f}pp，去掉最大正贡献后的其余合计{r['remaining_gain_after_largest_positive_pp']:+.6f}pp（原分母）。" for r in effects if r['platform']=='balanced_tail')
report='''# R80：固定R76双向模型的均匀物理恢复TTA扩展评价

本轮只检验R76 epoch6＋R79均匀恢复一步SGD，不继续源域训练、不引入Query权重或新模块。DEV正结果完整保留，但扩展范围没有同时建立相对自身冻结模型与同信息直接平均的可重复正增量，因此不采用为已验证TTA方法；不以未超过R59作为否定依据。

## 固定协议与复用边界

原Waymo-only epoch100、R76 e6（84f3455...）、RGB CNN、辅助头、匹配头、A256框及骨干不变。原102498个Adapter／局部双向交互／门控参数，一步SGD alpha0.5、clip1，两种76001/76002互补遮挡，逐例source重置。点特征前64维恢复、RGB patch均值、图像侧稀疏log-depth保留原0.1/0.1/0.01系数；点特征是冻结表示目标，并非真实语义标签。深度从反向交互前图像特征读取以防复制。零步分布/query保持项保留，但一步起点梯度为0，不宣称已经阻止漂移。恢复目标本身不依赖正确目标身份，也不由完整文字生成；本轮实际辅助更新不是新的语言条件化更新，文字仍参与更新后的grounding决策。

核心四端点：同模型零更新、完整＋两遮挡概率直接平均、均匀恢复一步、均匀权重错配绑定一步。错配只打乱逻辑点—像素绑定，真实RGB/depth恢复目标保持；使用完全相同步长／裁剪／预算，不额外匹配参数半径，范数差异单独报告。因此不能把两者差异只归因于方向。ABC/R59仅作系统参照，不是推进硬门槛。

DEV336已完成的source/zero/direct/uniform/ABC/R59直接复用R79封存结果，只补新uniform-wrong端点。VAL1280、TRAIN1024使用原R69固定清单和采样A原框，复用R78已验证的RGB几何缓存；实际加载原R76模型重新生成核心预测，未使用R78继续训练模型的输出冒充本轮。两组数据及DEV已有历史曝光，均为固定配置扩展开发评价，不是独立盲测。原始RGB/点云不需重读；没有新source训练／专家／FINAL访问。

## 真实执行与审计

真实DEV两条与扩展Drone/Quad各一条工程检查通过，R79均匀一步的参数与概率逐项完全一致。2640条全保留；新4944份更新态，另外336份DEV均匀态来自已封存R79。预测先封存，再独立重建全部5280份参数态，重放真实cross encoder/decoder/最终语言读出与原框，才join固定候选的已有IoU。所有错误、UNKNOWN、回退、负比较保留，不按GT挑可恢复样本。

## 各范围总体结果

'''+ '\n\n'.join(tables)+'''

## 自身适应的纠正与损伤

| 范围 | 平台 | IoU改善 | IoU恶化 | 换query | 相对自身冻结 pp |
|---|---|---:|---:|---:|---:|
'''+effecttext+'\n\n'+tailtext+'''

## recording分组

不把同场景帧/描述当独立场景证据。下面是各范围内部的recording别名；别名不保证跨范围对应同一原场景，不将其合并为更多独立场景，也不从单Drone场景构造总体CI。允许个别场景下降，但研究主张必须限于实际覆盖。

'''+ '\n\n'.join(scenetables)+f'''

## 源域参考与限制

源128只复用R79已暴露诊断结果，不再做梯度容量实验：R76冻结{source['frozen_bi']['mIoU_percent']:.6f}%，均匀一步{source['uniform_sgd1']['mIoU_percent']:.6f}%，直接平均{source['frozen_direct3']['mIoU_percent']:.6f}%。源域下降不能单独否定目标域增量，也不能将DEV上升称为已证明的目标域特化。

本轮生产{cost['seconds']:.3f}s，真实grounding后缀{cost['grounding_suffix_calls']}次，辅助参数梯度{cost['auxiliary_parameter_gradient_calls']}次；成功原生复核另{native['seconds']:.3f}s。使用冻结前缀/RGB几何缓存，是整组实验成本，不是端到端单臂时延。模型与数据来源未变，不扩大到所有3EED训练设置。

## 裁决

预先锁定的互补/单模态恢复消融条件是：均匀主方法在VAL1280和TRAIN1024均优于自己的零更新与直接平均。本轮没有满足，所以不触发该后续消融；不是因R59排名停止，也没有改用表现最好的控制充当主方法。仅凭辅助误差下降、非零更新或DEV正结果，不将“物理恢复稳定改善grounding”写成已成立结论。

保留双向结构和R79 DEV正结果；不采用当前固定均匀恢复TTA作为稳定方法，不继续本轮调参、Query加权或源训练。若继续研究，须区分源域准备后的静态跨域表现、辅助监督对任务的作用和有限步优化；本轮没有启动新算法。原checkpoint与R76e6 hash不变，其他paused项目登记不变。代码／配置／总体及逐例匿名标量／全部负例公开，数据、文字、坐标、ID、特征和权重不公开。
''';(H80/'REPORT.md').write_text(report);print(decision)
