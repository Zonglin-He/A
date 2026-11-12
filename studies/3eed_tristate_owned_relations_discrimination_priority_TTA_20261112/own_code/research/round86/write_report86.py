"""Describe the registered experiment and all results; no posthoc promotion."""
from core86 import *
main=read(H86/'score_dev336_attempt001/TASK_SUMMARY.json')['ALL336'];supp=read(H86/'score_supp30_attempt001/TASK_SUMMARY.json')['ALL30'];cov=read(H86/'EVIDENCE_COVERAGE_ROWS.json');up=read(H86/'ALL_UPDATE_ROWS.json');ep=read(H86/'EVIDENCE_PREPARATION_ROWS.json');targets=read(H86/'POSTSEAL_TARGET_PROXY_COVERAGE.json');opt=read(H86/'independent_optimization_audit_attempt001/AUDIT_RESULT.json');native=read(H86/'independent_native_audit_attempt001/AUDIT_RESULT.json');counts=read(B86/'prediction366_attempt001/COUNTS.json');primary='F_joint1_guard';v=main['balanced_mIoU_percent'];refs=['frozen_bi','F_direct_guard','F_candidate1_guard','F_mismatch1_guard','F_sum1_guard','F_joint1_raw','R59'];gains={r:v[primary]-v[r] for r in refs};passed=all(gains[r]>0 for r in ['frozen_bi','F_direct_guard','F_mismatch1_guard'])
decision={'state':'R86_COMPLETED_TRISTATE_OWNED_RELATION_AND_PRIORITY_DEVELOPMENT_PROTOTYPE','primary_arm':primary,'primary_steps':1,'secondary_steps':3,'primary_gains_pp':gains,'passes_fixed_primary_increment_checks':passed,'robust_TTA_adoption':False,'primary_not_replaced_by_secondary_after_scoring':True,'FINAL_access':False,'source_retraining':False,'new_visual_experts':False,'no_follow_on_queued':True};dump(H86/'DECISION.json',decision)
def table(s):
 lines=['|配置|Drone mIoU %|Quad mIoU %|两平台平均 %|平均Acc@25 %|平均Acc@50 %|','|---|---:|---:|---:|---:|---:|']
 for a in ARMS86:lines.append('|'+a+'|'+f"{s['drone'][a]['mIoU_percent']:.6f}|{s['quad'][a]['mIoU_percent']:.6f}|{s['balanced_mIoU_percent'][a]:.6f}|{np.mean([s[p][a]['Acc25_percent'] for p in ['drone','quad']]):.6f}|{np.mean([s[p][a]['Acc50_percent'] for p in ['drone','quad']]):.6f}|")
 return '\n'.join(lines)
text='''# R86：三态证据、依存归属与候选判别优先的完整适应原型

本轮保留R85/R76双向RGB–LiDAR架构，实际跑完366条、两种基座和全部42臂。主评价固定DEV336；其外同30条R84案例只作历史补充诊断，单独分母。没有源域再训练、新视觉专家、新框、回归更新或FINAL访问。DEV及补充均已历史暴露，不宣称独立测试或论文贡献已经成立。

## 两处修改与配套控制

**证据语义。** 颜色/关系分别记录支持、反证、未知。颜色仍是13类软像素描述，要求单颜色请求在三种已有支持范围下保持方向；brown/beige/gray也参与比较，不能据此声称通用外观识别。明确相机左右采用可见支持与投影框组成的区间，区间重叠保持未知；没有把可见表面均值当完整中心。源模型类别/参照实体分布仍只是先验，稀疏支持和邻物污染仍可能造成错误。

关系沿保存的UD依存头、主语和相对从句归属建立，保留参照物之间的边；不再把两个实体之间最后一个关系词挂到所有实体对。参照–参照执行一次同步消息传递，目标–参照仍保留多个假设。合成相对从句和多对象存在句在实际既有Stanza解析器上均得到预期两条边，没有凭空生成目标–第三实体边；这验证了指定逻辑反例，不是整套真实语言的解析金标。

候选按原有点支持/几何重复组共享证据。一项因素只有同时含支持和反证才启用；只从被反证候选移走观测支持的有限概率，按原先验和支持强度分配到有支持候选。未参与该比较的UNKNOWN目标概率逐项严格不变，不把其质量重新归一化给赢家。这不保证被支持的组就是文字指向实例；同色、在左侧都不足以确认身份。

同时保留两种读出：raw直接使用分布；guard只允许原第一名的既有重复组，以及存在局部正证据反对原组时得到支持的替换组。guard是最终选择资格限制，非全局正确性认证，置为−1的分数不是归一化概率。raw与guard所有结果分别保留，不能把guard的变化算作参数更新。适应目标的UNKNOWN质量保持不变，不意味着共享参数更新后所有UNKNOWN输出能逐项不动；原先验KL和最终资格读出分别发挥约束作用。

**优化主次。** 原完整输入KL加两遮挡KL均值仍为候选项C，原局部点/RGB/深度恢复为R。candidate仅沿gC更新并检查实际C端点；joint保留gC，将与其冲突的gR正交投影后叠加，且有限步必须同时满足C和C+R的固定Armijo验收。sum控制仍沿gC+gR，只验收总和。学习率0.1、范数裁剪1、回溯1至1/256；没有GT验收、阈值网格或事后择步。三步仅作预先声明的拟合强度对照，一步主臂不因结果换成三步。

这是借鉴[PCGrad原论文](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)与[作者实现](https://github.com/tianheyu927/PCGrad)的有主次投影设计。原PCGrad对任务对称/随机投影，本轮明确只保护候选梯度，不能命名为官方复现。一阶方向保护还不足以保证有限步下降，所以仍审计实际端点。

**基座与监督覆盖。** S为原始3EED残差基座，F为完整冻结R76基座；各自零步严格恢复各自输出，分别比较增量，不能把重新获得R76静态收益算成TTA。保留102,498个Adapter、交互Q/K/V/O和门控可更新参数；原3EED、CNN、匹配副本和恢复头均冻结。每输入/实验臂恢复同一准备状态。所有原A256框、坐标和锚点身份固定。

遮挡仍互补，允许锚点特征成为目标；每种遮挡先分配50%全局均匀、不依赖候选排名的配额，再分配局部配额。被遮挡点从反向交互键移除，深度头只看交互前图像，工程改写被遮挡输入的泄漏检查为0。没有单独扫描采样比例，不能仅由覆盖变化归因这个配额的独立贡献。两遮挡分别减去自身零步输出，故是同一目标的不同梯度路径，不是三份独立目标身份判断。

## DEV336：全部注册配置

'''+table(main)+'\n\n'
text+=f"固定主臂{primary}相对冻结F {gains['frozen_bi']:+.6f} pp，相对同证据直接guard {gains['F_direct_guard']:+.6f} pp，相对候选单独更新 {gains['F_candidate1_guard']:+.6f} pp，相对证据错配 {gains['F_mismatch1_guard']:+.6f} pp，相对只验收总和 {gains['F_sum1_guard']:+.6f} pp。优于自身冻结、同信息直接和错配的共同检查为 **{passed}**。\n\n"
text+='|基座|可用因素观测/336|原第一名有反证|直接raw换框|直接guard换框|raw选中无资格组|UNKNOWN目标最大误差|\n|---|---:|---:|---:|---:|---:|---:|\n'
for b in ['S','F']:
 rr=[r for r in cov if r['part']=='DEV336' and r['base']==b];text+=f"|{b}|{sum(r['active'] for r in rr)}|{sum(r['base_top_has_counter'] for r in rr)}|{sum(r['raw_changes_base'] for r in rr)}|{sum(r['guard_changes_base'] for r in rr)}|{sum(r['raw_winner_ineligible'] for r in rr)}|{max(r['unknown_probability_error'] for r in rr):.1g}|\n"
text+='\n各基座分别统计改善/损伤，不能用低覆盖解释后删除无更新输入：\n\n|臂|相对基座|IoU改善|IoU恶化|净mIoU pp|\n|---|---|---:|---:|---:|\n'
for b,ref in [('S','source'),('F','frozen_bi')]:
 for suffix in ['direct_raw','direct_guard','candidate1_guard','joint1_raw','joint1_guard','joint3_guard','mismatch1_guard','sum1_guard']:
  a=b+'_'+suffix;better=sum(main[pl][a]['improved_vs_'+ref] for pl in ['drone','quad']);worse=sum(main[pl][a]['worsened_vs_'+ref] for pl in ['drone','quad']);text+=f'|{a}|{ref}|{better}|{worse}|{v[a]-v[ref]:+.6f}|\n'
taskrows=read(H86/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');deltas=np.array([r['IoU'][primary]-r['IoU']['F_direct_guard'] for r in taskrows]);text+=f"\n主臂相对同证据F直接guard：{int((deltas>1e-10).sum())}条IoU改善、{int((deltas< -1e-10).sum())}条恶化；最大单条正/负贡献分别为{float(deltas.max()*100/336):+.6f} / {float(deltas.min()*100/336):+.6f} pp（主集两平台各168条）。不因某一个例删除或改方法。\n"
tt=[r for r in targets if r['part']=='DEV336'];gg=[r for r in up if r['part']=='DEV336' and r['base']=='F' and r['variant']=='joint' and r['step']==1 and r['classification_before'] is not None];cos=[r['gradient_cosine'] for r in gg if r['gradient_cosine'] is not None]
text+=f'''
## 执行正确性与有效性是两件事

独立优化审计：接受更新共{opt['accepted_steps']}步，其中候选优先joint/mismatch共{opt['accepted_priority_steps']}步，实际候选项上升{opt['accepted_priority_class_increases']}步；只验收总和的sum控制则有{opt['sum_control_candidate_increases']}步候选项上升。后者是控制臂的真实现象，全部保留，不混作主臂。主F首步有证据输入的候选/恢复梯度范数中位数分别为{float(np.median([r['classification_gradient_norm'] for r in gg])):.8g} / {float(np.median([r['recovery_gradient_norm'] for r in gg])):.8g}，可定义夹角余弦为负的有{sum(c<0 for c in cos)}/{len(cos)}条。未启用因素的损失记为未计算，而不是虚构为0。

DEV336最高IoU候选的封存后几何代理中，点恢复零目标覆盖：R85保护锚点{sum(r['R85_protected_proxy_targets']==0 for r in tt)}条，R85不保护{sum(r['R85_unprotected_proxy_targets']==0 for r in tt)}条，R86新增全局配额后{sum(r['new_point_targets_in_oracle_proxy']==0 for r in tt)}条。只用于解释，不用于采样或伪标签。该代理不是实例mask，覆盖变化也不是任务提升。

原关系枚举全366合计{sum(r['old_relation_count'] for r in ep)}条，新依存归属合计{sum(r['new_relation_count'] for r in ep)}条，其中实际可执行参照–参照边{sum(r['reference_to_reference_edges'] for r in ep)}条。有因素的观测{sum(r['factor_count']>0 for r in ep)}条。不能把删掉的每条旧边都宣布为错边：本轮只处理部分语法、最多四参照物、明确相机左右，其他应保留UNKNOWN，覆盖和可靠性均有限。

## 三条指定案例及30条补充

原图、完整句、全景/局部、各状态框、可见点和候选分数/排名/正反证已生成到本地PRIVATE图页。下面三条仍是诊断，不能改变总体分母或按它们补规则。新的query只有数值IoU，不自动宣称身份已认证。

|R84匿名案例|source|R76冻结|R85一步|新F直接guard|新F候选一步|新F联合一步|新S联合一步|
|---|---:|---:|---:|---:|---:|---:|---:|
'''
cr=read(H86/'R84_CASE_FOLLOWUP_ROWS.json')
for ci in [21,4,20]:
 r=cr[ci];text+='|'+r['case']+'|'+'|'.join(f'{r[a+"_IoU"]:.6f}' for a in ['source','frozen_bi','R85_full1','F_direct_guard','F_candidate1_guard','F_joint1_guard','S_joint1_guard'])+'|\n'
if (H86/'CASE_VISUAL_REVIEW.md').exists():text+='\n'+(H86/'CASE_VISUAL_REVIEW.md').read_text()+'\n'
text+='\n30条按历史案例选出，不作为扩大样本确认；完整结果与所有负比较单列：\n\n'+table(supp)+'\n\n'
text+=f'''## 审计、成本与结论边界

首次CPU准备对多参照存在句过度拒绝，在GPU与任务评分前修复一次；两个版本和原代码均保留，非按任务标签选规则。真实Drone/Quad工程输入均非零更新；随后完整366封存预测与模型状态，独立重放{native['saved_endpoint_replays']}个真实端点，最大概率/分数误差{native['max_probability_or_score_error']:.3g}。候选框、依存建边、冻结缓存归属、三态目标、UNKNOWN质量及资格读出均回核。之后只关联既有已封存A256 IoU表，新GT几何评分0次。全部42臂、平台/5场景、无更新/UNKNOWN/回退和有向负例保留。场景不增加，未做单场景总体CI。

生产运行{counts['seconds']:.2f}秒、峰值CUDA已分配{counts['peak_GPU_bytes']}字节；这是缓存点前缀、冻结实体预测和既有传感器支持上的增量成本，不是从原始数据起算的端到端速度。复用缓存不包含外挂视觉模型知识。原source与R76准备状态不改，其他暂停项目不改。

候选项端点优先下降已被实际执行验证，但下降的只是本轮候选目标，不能推出它符合真实身份。若主臂未超过自己的冻结、同信息直接读出和错配，则不采用本配方，不以三步/另一基座/guard单独效果事后替换主结果。即使出现局部收益，DEV336多轮曝光也不足以宣布稳定TTA或novelty。本轮不再追加步长/阈值搜索，不排队后续实验，FINAL保持未访问。
'''
text+='\n## 固定主臂裁决\n\n'+('主臂通过本轮开发净增量检查，但未经过独立确认，不作稳健方法采纳。' if passed else '主臂没有同时超过自身冻结、同信息直接读出和错配，不采用当前完整配方。优化执行缺口与监督身份可靠性必须分开：前者能用端点审计验证已修，后者只由总体任务对照判断，不能用辅助损失下降替代。')+'\n'
text+='''
本轮更具体的裁决：F直接guard相对冻结只有2条IoU恶化、没有改善；主臂相对直接读出的正增量，主要是在撤回G造成的损伤。主臂与候选单独更新、总和验收在任务读出上相同，没有显示判别优先投影带来额外选框收益。去掉资格限制的raw确有5条改善、2条恶化，相对冻结+0.118152 pp，但其证据错配raw达到23.026997%，高于正确绑定的22.729748%，不能据此宣称依赖真实三模态绑定。资格限制也不是已成立保护：同信息下raw比guard高0.119501 pp，原图复核显示它会阻止同车辆附近较好框的选择。

因此，已被修复的是优化主次与UNKNOWN概率语义的实现缺口；仍未被证实的是局部支持/参照先验能否识别正确实例，以及如何把同一物体的候选组组织成可靠比较。不能把本轮负结果归于零梯度或没有执行更新，也不能据此否定全部3EED-TTA；本配方不采用，不继续用加步数/改主臂挽救。
''';(H86/'REPORT.md').write_text(text);dump(H86/'PUBLIC_FIXED_CONFIGURATION.json',read(H86/'experiment/DESIGN_LOCK.json'));prep=read(B86/'evidence366_attempt002/COUNTS.json');eng=read(B86/'engineering_attempt001/COUNTS.json')
dump(H86/'NORMALIZED_COSTS.json',{'counts':{'main':336,'supplements':30,'arms':len(ARMS86),'accepted_updates':counts['accepted_updates'],'grounding_suffix_calls_producer':counts['grounding'],'objective_evaluations':counts['objective'],'trial_objectives':counts['trial_objectives'],'independent_endpoint_replays':native['saved_endpoint_replays'],'new_GT_scores':0},'timing_seconds':{'evidence_preparation':prep['seconds'],'engineering':eng['seconds'],'producer':counts['seconds'],'native_audit':native['seconds'],'actual_loss_replay':read(H86/'independent_loss_replay_audit_attempt001/AUDIT_RESULT.json')['seconds']},'peak_bytes':{'producer_GPU_allocated':counts['peak_GPU_bytes']},'flags':{'cached_prefix_not_end_to_end':True,'FINAL_access':False,'new_visual_experts':False}})
dump(H86/'NORMALIZED_HISTORY.json',{'events':[{'stage':'dependency_preparation','attempt':1,'status':'completed_overreject','failure_type':'multiobject_existential_ownership_overreject','before_GT':True,'prior_attempt_preserved':True},{'stage':'dependency_preparation','attempt':2,'status':'completed','repair_type':'allow_multiple_explicit_entity_owners','before_GT':True,'prior_attempt_preserved':True},{'stage':'complete_prototype','attempt':1,'status':'completed','producer_predictions_changed':False}]})
dump(H86/'PUBLIC_PROVENANCE.json',{'prediction_seal_sha256':sha(B86/'prediction366_attempt001/SEALED.json'),'design_lock_sha256':sha(H86/'experiment/DESIGN_LOCK.json'),'evidence_seal_sha256':sha(B86/'evidence366_attempt002/SEALED.json'),'native_audit_sha256':sha(H86/'independent_native_audit_attempt001/AUDIT_RESULT.json'),'scalar_audit_sha256':sha(H86/'independent_scalar_audit_attempt001/AUDIT_RESULT.json'),'raw_media_or_description_or_model_states_exported':False});print(decision)
