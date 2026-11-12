"""Fixed-primary report plus paired component effects, never promotes a winner."""
from core87 import *
su=read(H87/'score_dev336_attempt001/TASK_SUMMARY.json')['ALL336'];values=su['balanced_mIoU_percent'];counts=read(B87/'prediction336_attempt001/COUNTS.json');audit=read(H87/'independent_native_audit_attempt001/AUDIT_RESULT.json');cov=read(H87/'EVIDENCE_COVERAGE_ROWS.json');updates=read(H87/'ALL_UPDATE_ROWS.json');tasks=read(H87/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');primary='full_tta1';refs=['frozen_bi','full_direct','wrong_tta1','R59','R86_TTA'];gains={r:values[primary]-values[r] for r in refs};passed=all(gains[r]>0 for r in ['frozen_bi','full_direct','wrong_tta1']);decision={'state':'R87_COMPLETE_SOFT_ALIAS_JOINT_BINDING_COMPONENT_DEVELOPMENT_EVALUATED','primary_arm':primary,'primary_steps':1,'secondary_steps':3,'primary_gains_pp':gains,'passes_primary_increment_checks':passed,'robust_TTA_adoption':False,'no_posthoc_primary_promotion':True,'FINAL_access':False,'new_source_training':False,'new_external_vision_models':False,'no_follow_on_queued':True};dump(H87/'DECISION.json',decision)
text='''# R87：软候选关联、主体—参照物联合绑定与组件作用

本轮保留R76双向RGB–LiDAR结构、Waymo-only源模型和全部原始A256框，在历史暴露DEV336上完成32个注册臂。没有访问FINAL，没有源域再训练、外挂视觉专家或新增框。一步主臂与三步强度对照在评分前固定；不按三条旧案例设计特判，不将选出的good/bad案例用于后续调规则。

## 论文依据与真实迁移边界

'''+(H87/'LITERATURE.md').read_text()+'''

## 方法与对照

1. 软候选关联A同时使用框IoU、原始seed支持Dice、投影像素支持Dice及冻结点特征均值余弦；所有权重有明确观测来源，非贪心合并、非传递连通分量，不把它命名为实例mask。对角为1，关联强度保留而非硬切组。通过邻域语义平均与密度校正减少重复query对实体先验的重复计票；同样会惩罚密集的正确候选，必须由no_alias/hard_alias对照检验。
2. 完整冻结R76与自身主体短语分布的几何平均作为主体语义先验，参照物使用同一个冻结3EED的短语分布；完整词组进入已有语言编码路径，不使用目标GT跨度或伪造新RGB语义。短语在原完整句的固定query锚点和六层位置下重算，不生成新框，也不沿用新的proposal编号。固定空间锚点不是已认证的实例标识：query仍可能读取邻物；本轮没有CORE式重新检测短语框及学习其关联的训练阶段。语义分布仍未校准，主体短语也不足以独立指代。
3. 原颜色支持/反证经软邻域聚合进入实体一元势；已修复归属的相机左右关系形成成对势。不同实体角色间加入软同实例排他势1−A，参照物增加固定先验质量0.5的NULL状态，NULL交互势为1。NULL表示未观测到参照物，不作为新的候选赢家。固定四轮cavity sum-product同时更新角色；树上与精确小图核对，环图只作有限近似，不声称全局唯一正确实例。不同名词词组可能共指同物体，排他假设本身也是待验证条件。
4. 主方法不沿用R86的硬资格限制；最终仍从原A256中选择。与R86的UNKNOWN概率逐项保留含义不同：这里显式保留未知参照物，整体角色分布仍是由模型先验构成的可错假设，不把每个候选称为经过观测认证。
5. 每输入恢复同一R76准备状态，更新既有102498个Adapter/双向交互投影与门控，其他参数冻结。继续使用R86完整/两遮挡候选KL及原恢复项、同一遮挡与局部权重，SGD0.1/裁剪1、固定1至1/256回溯，实际候选与总损失均需验收。没有另行搜索学习率、步长、NULL先验或融合强度。

所有G组件都保留同信息直接读出和同目标TTA一步：hard_alias=换回原硬组；no_alias=独立query；independent=参照先验固定、不接收其他角色的反馈；no_subject=去主体短语语义先验；no_relation=保留软角色排他但去空间关系；no_RGB=只去G的RGB颜色属性，学生仍有原RGB分支；no_null=不允许参照物缺失；no_exclusivity=去角色同实例排他。wrong统一打乱反馈与query的对应，包含角色语义、几何、RGB和关联矩阵，模型输入保持不变；不是仅换一组随机负标签。

冻结模型另保留关闭LiDAR→RGB反向路径及将RGB替换为常数灰图（同时移除颜色和结构）的干预：它们是相同权重下的推理撤回诊断，有分布变化，不能等同于重新训练的单向架构对照。candidate_only1使用相同完整G、只执行候选项更新，用于检查恢复项是否提供任务增量。

## DEV336全部注册结果

|方法|Drone mIoU %|Quad mIoU %|两平台平均 %|平均Acc@25 %|平均Acc@50 %|
|---|---:|---:|---:|---:|---:|
'''
for arm in ARMS87:text+='|'+arm+'|'+f"{su['drone'][arm]['mIoU_percent']:.6f}|{su['quad'][arm]['mIoU_percent']:.6f}|{values[arm]:.6f}|{np.mean([su[p][arm]['Acc25_percent'] for p in ['drone','quad']]):.6f}|{np.mean([su[p][arm]['Acc50_percent'] for p in ['drone','quad']]):.6f}|\n"
text+='\n固定主臂增量：'+', '.join(k+f' {v:+.6f} pp' for k,v in gains.items())+f'。同时超过自身冻结、同信息直接读出与错配：**{passed}**。不按本表事后更换主臂。\n'
text+='\n## 组件是否有作用\n\n以下均为完整方法减对应消融的配对差值；正值表示组件对该读出有净帮助，负值表示保留它反而更差。允许个别负例，不要求每个场景上涨。组件有交互，单项消融不能相加成贡献百分比。\n\n|被移除/替换组件|直接读出差值 pp|TTA一步差值 pp|TTA全方法更好/更坏条数|\n|---|---:|---:|---|\n'
comp=read(H87/'COMPONENT_EFFECT_ROWS.json')
for mode in MODES87[1:]:
 a=next(r for r in comp if r['component_removed']==mode and r['readout']=='direct');b=next(r for r in comp if r['component_removed']==mode and r['readout']=='tta1');text+=f"|{mode}|{a['component_net_pp']:+.6f}|{b['component_net_pp']:+.6f}|{b['full_better_n']}/{b['full_worse_n']}|\n"
text+=f"\n相同完整G下，联合恢复相对候选单独更新：{values[primary]-values['candidate_only1']:+.6f} pp。冻结双向相对撤回反向路径：{values['frozen_bi']-values['frozen_no_reverse']:+.6f} pp；相对常数灰RGB：{values['frozen_bi']-values['frozen_gray_RGB']:+.6f} pp。后两项只支持已准备模型对路径/输入的依赖，不能排除撤回输入分布变化。\n"
text+='\n## good/bad案例与总体错误分布\n\n案例从预测封存并评分后的完整主集，按相对冻结和相对直接读出的最大正/负变化选取，再补绝对高IoU与候选可恢复低IoU各两条；稳定顺序去重，最多12条。GT仅用于这次事后描述，不用于G、适应、验收或方法选择。绝对正确与方法纠错分开记：原模型已正确且本轮没变，不算新方法救回。原句/原图/坐标/候选索引只在本地PRIVATE图册中，公开保留匿名标量和结论。\n\n|参照|IoU改善|IoU恶化|无IoU变化|Top1候选变化|\n|---|---:|---:|---:|---:|\n'
for ref in ['source','frozen_bi','full_direct','wrong_tta1']:
 d=np.array([r['IoU'][primary]-r['IoU'][ref] for r in tasks]);text+=f"|{ref}|{int((d>1e-10).sum())}|{int((d< -1e-10).sum())}|{int((abs(d)<=1e-10).sum())}|{sum(r['queries'][primary]!=r['queries'][ref] for r in tasks)}|\n"
text+='\n封存后用原A256最高IoU作几何上限诊断：候选池覆盖不足与可恢复排序失败分开，不能将前者都归因于G。\n\n|平台/阈值|候选上限覆盖|主方法成功|可恢复排序失败|候选覆盖不足|冻结成功被破坏|冻结失败被救回|\n|---|---:|---:|---:|---:|---:|---:|\n'
for er in read(H87/'ERROR_DECOMPOSITION_ROWS.json'):
 if er['arm']=='full_tta1':text+='|'+er['platform']+'/'+str(er['threshold'])+'|'+'|'.join(str(er[k]) for k in ['oracle_candidate_covered','task_above_threshold','recoverable_ranking_failures','candidate_coverage_failures','frozen_success_now_failure','frozen_failure_now_success'])+'|\n'
text+='\n|匿名案例|平台|筛选原因|冻结双向|直接G|TTA一步|错配TTA|\n|---|---|---|---:|---:|---:|---:|\n'
for r in read(H87/'GOOD_BAD_CASE_ROWS.json'):text+=f"|{r['case']}|{r['platform']}|{r['selection_reasons']}|{r['frozen_bi_IoU']:.6f}|{r['full_direct_IoU']:.6f}|{r['full_tta1_IoU']:.6f}|{r['wrong_tta1_IoU']:.6f}|\n"
if (H87/'CASE_VISUAL_REVIEW.md').exists():text+='\n'+(H87/'CASE_VISUAL_REVIEW.md').read_text()+'\n'
up=[r for r in updates if r['mode']=='full' and r['step']==1];fc=[r for r in cov if r['mode']=='full'];text+=f'''
## 覆盖、审计与成本

主集全G非零修正目标{sum(r['active'] for r in fc)}/336，存在可执行关系{sum(r['edges']>0 for r in fc)}条；可定义NULL后验的平均值{float(np.mean([r['reference_null_mean'] for r in fc if r['reference_null_mean'] is not None])):.6f}。这些不是伪标签正确率。主臂实际接受首步{sum(r['accepted'] for r in up)}/336，全部方案共{audit['accepted_updates']}次接受，候选项上升{audit['accepted_candidate_increases']}次。没有通过跳过负例提高分数。

预测及模型状态封存后，独立重建软关联/各组件目标、重放{audit['saved_endpoint_replays']}个模型端点并重算{audit['actual_loss_scalar_checks']}个损失标量，最大读出误差{audit['max_probability_or_score_error']:.3g}、目标误差{audit['max_target_error']:.3g}、损失误差{audit['max_actual_loss_error']:.3g}。之后仅关联既有已封存A256 IoU；新GT几何评分0。全部32臂、336条、5个历史场景、阈值及有向负比较保留。场景没有增加，不构造逐帧独立CI或声称新的独立测试。

第一次预检只连接显式关系，星形图目标边缘分布会等价于固定参照边缘化，不能充分检验联合绑定；在任何任务评分前保留原准备/两输入工程结果，补入软角色排他及对应消融，重新锁定最终32臂。没有依据任务GT修改算法。生产GPU增量运行{counts['seconds']:.2f}秒，峰值已分配{counts['peak_GPU_bytes']}字节；复用冻结前缀/自身实体预测/传感器支持，不是端到端运行时间。

## 固定主臂裁决

'''+('主臂通过本轮开发增量检查，但历史DEV曝光与有限场景仍不足以稳健采纳。' if passed else '主臂未同时超过冻结双向、同信息直接读出和匹配错配控制，不采用本配方。')+''' 组件是否有净作用按完整配对表解释；辅助损失下降、软图更一致或模型确实更新均不能代替任务收益。没有继续追加阈值/步长网格，不触及FINAL，不将某条个例修复升级为总体机制证据。
'''

text+='\n## 本轮原注册方法的具体判断\n\n主体短语先验相对去掉它的TTA有+0.439195 pp，恢复项在同G下有+0.109897 pp，G的RGB属性项+0.064297 pp；这些是在当前组合中的净作用，不是独立可加贡献。主体先验也造成CASE003/004的明显误改，不能把正均值变成可信伪标签认证。软关联相对不做关联为−0.088909 pp，相对原硬关联为−0.199120 pp；新软分组没有成立。联合与固定参照的一步输出在全部336条的最终query完全相同；空间关系只有+0.002897 pp且改善/恶化各1条，没有建立可靠的关系增益。NULL和软排他分别为−0.132253/−0.112421 pp，不采用它们作为已有效组件。\n\n全体恢复项/分类项的端点下降已核对。再从封存概率分离完整输入与遮挡分类：310条有目标修正中，仅1条完整输入KL上升；但最终169/336条没有选择G的最高分。CASE003在完整KL和遮挡KL都下降时仍从正确人物改成另一人物。因此当前缺口不只是候选项没有验收，还包括分布拟合与最终离散选择之间的联系。\n'
if (H87/'STRUCTURAL_REPAIR_DECISION.json').exists():
 sr=read(H87/'score_structural336_attempt001/TASK_SUMMARY.json')['ALL336'];sv=sr['balanced_mIoU_percent'];sd=read(H87/'STRUCTURAL_REPAIR_DECISION.json');text+='\n## 一次单列的角色清单结构修复验证\n\n该部分在原32臂评分和全局角色审计后另锁定，DEV已曝光，不能替换原主臂或称为独立确认。保留双向网络、原A256、所有更新参数与验收、遮挡及系数，仅修参照清单的依存论元/字段头/复合短语及关系优先选取，再由同一冻结3EED重算新增短语分数。新增自身短语前向137次，不引入外部专家。一次修复后停止，不再追着个例加规则。\n\n同时补足一个更强的无更新对照：固定50/50平均冻结模型与G分布。系数没有按GT选择或扫描；它用完全相同信息，检验TTA是否仅靠较保守地保留source就能超过较差的直接G。原G同样的50/50诊断单列，非原注册主结果。\n\n|修复验证臂|Drone mIoU %|Quad mIoU %|平均 mIoU %|\n|---|---:|---:|---:|\n'
 for arm in sr['balanced_mIoU_percent']:text+='|'+arm+'|'+f"{sr['drone'][arm]['mIoU_percent']:.6f}|{sr['quad'][arm]['mIoU_percent']:.6f}|{sv[arm]:.6f}|\n"
 text+='\n修复后full_tta1相对冻结/完整直接G/保留source一半直接读出/错配/修复前TTA的增量：'+', '.join(k+f' {v:+.6f} pp' for k,v in sd['secondary_gains_pp'].items())+'。修复前G的固定50/50平均为'+f"{sv['original_full_half']:.6f}%"+'。原方法和修复方法的所有负例分别保留。\n'
 text+='\n|修复后的组件|直接G净作用 pp|50/50读出净作用 pp|TTA净作用 pp|\n|---|---:|---:|---:|\n'
 cr=read(H87/'STRUCTURAL_COMPONENT_ROWS.json')
 for mode in ['hard_alias','no_alias','independent','no_subject']:text+='|'+mode+'|'+'|'.join(f"{next(r['component_net_pp'] for r in cr if r['component_removed']==mode and r['readout']==k):+.6f}" for k in ['direct','half','tta1'])+'|\n'
 text+='\n|原固定案例|修复前TTA|修复后G|修复后50/50|修复后TTA|修复后错配TTA|\n|---|---:|---:|---:|---:|---:|\n'
 for r in read(H87/'STRUCTURAL_CASE_FOLLOWUP_ROWS.json'):text+='|'+r['case']+'|'+'|'.join(f"{r[k+'_IoU']:.6f}" for k in ['original_full_tta1','full_direct','full_half','full_tta1','wrong_tta1'])+'|\n'
 if (H87/'STRUCTURAL_REPAIR_REVIEW.md').exists():text+='\n'+(H87/'STRUCTURAL_REPAIR_REVIEW.md').read_text()+'\n'

(H87/'REPORT.md').write_text(text);dump(H87/'PUBLIC_FIXED_CONFIGURATION.json',read(H87/'experiment/DESIGN_LOCK.json'));prep=read(B87/'evidence336_attempt002/COUNTS.json');dump(H87/'NORMALIZED_COSTS.json',{'counts':{'main':336,'arms':len(ARMS87),'accepted_updates':counts['accepted_updates'],'grounding_suffix_calls':counts['grounding'],'objective_evaluations':counts['objective'],'independent_endpoint_replays':audit['saved_endpoint_replays'],'new_GT_scores':0},'timing_seconds':{'preparation':prep['seconds'],'producer':counts['seconds'],'independent_native_and_loss_audit':audit['seconds']},'peak_bytes':{'producer_GPU_allocated':counts['peak_GPU_bytes']},'flags':{'cached_prefix_not_end_to_end':True,'FINAL_access':False,'new_external_vision_models':False}});dump(H87/'NORMALIZED_HISTORY.json',{'events':[{'stage':'role_graph_preflight','attempt':1,'status':'completed','failure_type':'star_marginal_equivalent_to_independent','before_GT':True,'prior_attempt_preserved':True},{'stage':'role_graph_preflight','attempt':2,'status':'completed','repair_type':'soft_distinct_role_coupling_with_NULL_control','before_GT':True,'prior_attempt_preserved':True},{'stage':'complete_prototype','attempt':1,'status':'completed','producer_predictions_changed':False}]});dump(H87/'PUBLIC_PROVENANCE.json',{'prediction_seal_sha256':sha(B87/'prediction336_attempt001/SEALED.json'),'design_lock_sha256':sha(H87/'experiment/DESIGN_LOCK.json'),'native_audit_sha256':sha(H87/'independent_native_audit_attempt001/AUDIT_RESULT.json'),'scalar_audit_sha256':sha(H87/'independent_scalar_audit_attempt001/AUDIT_RESULT.json'),'private_original_media_or_descriptions_exported':False});print(decision)

# Separate repair costs/history and strengthen the adoption statement, without changing the original primary.
if (H87/'STRUCTURAL_REPAIR_DECISION.json').exists():
 extra=read(B87/'structural_prediction336_attempt001/COUNTS.json');aud=read(H87/'independent_structural_native_audit_attempt001/AUDIT_RESULT.json');prep2=read(B87/'structural_evidence336_attempt002/COUNTS.json');cost=read(H87/'NORMALIZED_COSTS.json');cost['counts'].update(structural_inputs_same_DEV=336,structural_arms=32,structural_accepted_updates=extra['accepted_updates'],structural_grounding_suffix_calls=extra['grounding'],structural_endpoint_replays=aud['saved_endpoint_replays'],structural_final_phrase_calls=prep2['new_own_model_phrase_suffix_calls'],structural_prior_phrase_calls_retained=read(B87/'structural_evidence336_attempt001/COUNTS.json')['new_own_model_phrase_suffix_calls']);cost['timing_seconds'].update(structural_producer=extra['seconds'],structural_audit=aud['seconds'],structural_final_preparation=prep2['seconds']);cost['peak_bytes']['structural_GPU_allocated']=extra['peak_GPU_bytes'];dump(H87/'NORMALIZED_COSTS.json',cost);history=read(H87/'NORMALIZED_HISTORY.json');history['events']+=[{'stage':'structural_inventory','attempt':1,'status':'after_original_DEV_exposure','failure_type':'all_noun_slots_instead_of_arguments','before_GT':False,'prior_attempt_preserved':True},{'stage':'structural_fixture','attempt':1,'status':'failed','failure_type':'head_expectation_not_maximal_NP','before_GT':False,'prior_attempt_preserved':True},{'stage':'structural_fixture','attempt':2,'status':'failed','failure_type':'adjacent_nominal_apposition_not_merged','before_GT':False,'prior_attempt_preserved':True},{'stage':'structural_fixture','attempt':3,'status':'completed','repair_type':'generic_apposition_NP_merge','before_GT':False,'prior_attempt_preserved':True},{'stage':'structural_complete','attempt':1,'status':'completed','producer_predictions_changed':False}];dump(H87/'NORMALIZED_HISTORY.json',history);d=read(H87/'DECISION.json');d.update(full_soft_joint_recipe_adopted=False,structural_correction_separate_development_diagnostic=read(H87/'STRUCTURAL_REPAIR_DECISION.json'));dump(H87/'DECISION.json',d)
