"""Measured conclusions, secondary repair and actually viewed11case pages."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round94';rd=lambda p:json.loads(p.read_text());d=rd(H/'DECISION.json');s=rd(H/'score_dev336_attempt001/TASK_SUMMARY.json')['ALL336']['balanced_mIoU_percent'];ss=rd(H/'score_solver336_attempt001/TASK_SUMMARY.json')['ALL336'];effects=rd(H/'COMPONENT_EFFECT_ROWS.json');T=rd(H/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');sol=rd(H/'score_solver336_attempt001/ALL_TASK_ROWS_PRIVATE.json');sb={x['id_PRIVATE']:x for x in sol};notes={
'CASE001':('different_visible_objects','新q和精确混合均保留近处车辆；初版Adam和修复GN仍转向远处区域。证据未要求该误改，说明共享末层更新未忠实执行这份正确保护。'),
'CASE002':('same_candidate_wrong_person','新q直接读出改到左侧主体，0到0.5376；混合仍选另一行人，Adam与GN也没有改正。源分布保护会阻碍这份有益变化，不能再只归因于G没信号。'),
'CASE003':('same_candidate','同一SUV框保持，旧物理更新中的重复框改善本轮未复现。普通梯度次对照可换到较高IoU重复框，不等同实例身份监督。'),
'CASE004':('same_candidate','直接证据和一步更新保留同一左侧行人框；三步Adam才改到较差重复框。更充分拟合当前目标没有带来更好的定位。'),
'CASE005':('different_visible_regions_target_partially_clipped','从远处区域回到左侧边缘车辆附近，q/Adam/GN均0到0.3185，shuffleAdam未实现。目标边缘裁切，精确身份细节仍部分未知；全体总分中正常计入。'),
'CASE006':('same_visible_object_duplicate_boxes','同一SUV附近由0.2468换到0.2748，q/Adam/GN均执行并跨Acc25；不是找回另一物体。'),
'CASE007':('same_visible_object_duplicate_boxes','同一SUV附近改到较差重复框0.2788到0.2503，q与更新共同继承此负例，仍计总体。'),
'CASE008':('same_candidate','q与混合转到同一白色车辆的较差重复框0.5116到0.3722；Adam/GN保留原框。更新相对直接读出变好只是避免伪目标损伤，没有超过零步。'),
'CASE009':('same_candidate','画面右边缘同一人物的重复框，q/混合略差；Adam/GN一步保持原框，三步Adam继承较差框。不能把保持原框称新实例纠错。'),
'CASE010':('same_candidate','同一白色车辆已有更好重复框，q/混合从0.2841到0.3596，Adam/GN都未执行这份提升。'),
'CASE011':('same_candidate','q从SUV附近转到远处上沿区域并降到0；混合、Adam和GN保留原低IoU框。该例证明新证据仍可误导，不因总体净收益而删除。')}
reviews=[]
for c in rd(H/'experiment/GOOD_BAD_CASES_PRIVATE.json'):
 identity,note=notes[c['case']];reviews.append({'case':c['case'],'platform':c['task']['platform'],'full_scene_caption_crop_BEV_reviewed':True,'initial_primary_identity':identity,'observation':note,'secondary_solver_saved_prediction_checked':True,'secondary_raw_image_overlay_not_separately_viewed':True,'postseal_descriptive_not_deployment_rule':True})
(H/'CASE_REVIEW_ANONYMOUS_ROWS.json').write_text(json.dumps(reviews,indent=2,ensure_ascii=False)+'\n');direct=next(x for x in effects if x['arm']=='binding_direct' and x['reference']=='R91_frozen' and x['platform']=='balanced');fit=rd(H/'SOLVER_UPDATE_ROWS.json');execs=rd(H/'SOLVER_EXECUTION_ROWS.json');d.update(direct_better_n=direct['better_n'],direct_worse_n=direct['worse_n'],direct_largest_positive_contribution_pp=direct['largest_positive_contribution_pp'],direct_net_pp=direct['net_pp'],original_primary_not_promoted=True,secondary_solver_scope_limited_by_fixed_trust_region=True,new_TTA_recipe_adopted=False);(H/'DECISION.json').write_text(json.dumps(d,indent=2)+'\n')
L=['','## 修复后的证据与执行必须分开裁决','',f'密集对应q直接读出 **{s["binding_direct"]:.6f}%**，相对零步{direct["net_pp"]:+.6f}pp，{direct["better_n"]}改善/{direct["worse_n"]}恶化；Acc25救回{direct["rescues_Acc25"]}/破坏{direct["breaks_Acc25"]}。比打乱delta直接{s["binding_direct"]-s["shuffle_direct"]:+.6f}pp，比反向delta直接{s["binding_direct"]-s["reverse_direct"]:+.6f}pp。正常对应修正与候选匹配有关，值得保留开发信号；不称已可靠解决全部身份绑定。','',f'限制：Drone直接mIoU略降、Quad升；Acc50略降；最大单例正贡献{direct["largest_positive_contribution_pp"]:+.6f}pp，大于净增益。当前R93_raw参考{s["R93_raw"]:.6f}%也高于新直接读出。因此不把此结果包装成新的总体最优/稳定跨域TTA。','', '最关键固定对照：新q保住旧近处车辆并改正旧行人，说明R93的两个难例不再都卡在反馈构造；初版匹配投影更新却仍破坏前者、漏掉后者。原始KL最优混合在行人例本就选错，因此源输出保护会阻碍部分正确证据；但即使不按混合拟合，后续数值修复也未解决全部执行。','', '## 唯一一次公开披露的后开发求解器修复','', '以下在初版DEV成绩已暴露后锁定，属于同一开发集上的修复诊断，不是独立重复。原主方法binding_bt1与负结论保留，不以secondary替换。详见protocol/SOLVER_REPAIR_PROTOCOL.md。新证据、图像、候选、权重初值与层位不变；以中心化候选logit平方误差为目标，用阻尼Gauss–Newton一次求解同一末端层的局部变化。lambda=.01trace(JcJcT)/256，最大参数L2为source层范数5%，九档回溯，没有GT选步/网格。普通梯度控制具有同目标、同GN提案L2；混合目标GN隔离源保护，打乱目标GN检验证据对应。','', '| 方法 | Drone mIoU % | Quad mIoU % | 等权mIoU % | Acc25 % | Acc50 % |','|---|---:|---:|---:|---:|---:|']
for x in rd(H/'SOLVER_METHOD_ROWS.json'):
 if x['platform']=='balanced':
  a=x['arm'];L.append(f'| {a} | {ss["drone"][a]["mIoU_percent"]:.6f} | {ss["quad"][a]["mIoU_percent"]:.6f} | {x["mIoU_percent"]:.6f} | {x["Acc25_percent"]:.6f} | {x["Acc50_percent"]:.6f} |')
L+=['','| 方法 | 对照 | 平台 | Δpp | 改善/恶化/不变 | 换query |','|---|---|---|---:|---:|---:|']
for x in rd(H/'SOLVER_EFFECT_ROWS.json'):L.append(f'| {x["arm"]} | {x["reference"]} | {x["platform"]} | {x["net_pp"]:+.6f} | {x["better_n"]}/{x["worse_n"]}/{x["unchanged_n"]} | {x["query_changes"]} |')
L+=['','| 更新 | 60条q要求换第一名中实际执行数 | 全336与q一致数 |','|---|---:|---:|']
for a in ['binding_bt1','gn_bt','gradient_bt','gn_mixture_bt','gn_shuffle_bt']:
 ee=[x for x in execs if x['arm']==a];L.append(f'| {a} | {sum(x["teacher_changes_source_top1"] and x["matches_teacher_top1"] for x in ee)} | {sum(x["matches_teacher_top1"] for x in ee)} |')
gn=[x for x in fit if x['method']=='gn'];aa=rd(H/'independent_solver_audit_attempt001/AUDIT_RESULT.json');L+=['',f'GN提案336/336触及固定5%幅度上限；平均中心化分数误差只完成{np.mean([1-x["after"]/max(x["before"],1e-30) for x in gn])*100:.6f}%修正。元数据unconstrained_L2字段实际存放限幅后的提案L2，按其与source_relative_radius_limit相等识别触限，不把它当真实未限幅范数。当前信赖域/阻尼限制仍是欠拟合的竞争解释；这份单步配置不能证明末层表达能力不可能。没有追加放大幅度或减阻尼网格。','',f'独立构造显式256×18496Jacobian验证因子Gram方法，并逐trial完整原生重放{aa["trial_replays"]}次；参数最大差{aa["max_parameter_state_error"]:.12g}、概率{aa["max_probability_error"]:.12g}、loss{aa["max_loss_error"]:.12g}、Top1不一致{aa["top1_changes"]}。没有新增证据调用或GT优化。','', '| 固定/总体案例 | 原图观察与二次求解端点 |','|---|---|']
for x in reviews:L.append(f'| {x["case"]} | {x["observation"]} |')
L+=['','最终保留：R91双向架构，密集对应差分作为有有限开发价值的新证据组件；全部负例和旧参照不变。未修好：参数更新尚未超过自身零步、q直接或同信息最优混合，GN也未完成关键两例的同时修复，不采纳这两份TTA配方。角色/实例支持并未获得独立身份认证，127条可恢复Acc25失败仅4条在新目标中获得明确合格候选优势，覆盖仍低；固定A256的几何上限也未改变。阶段A/B共用同336观测，不重复计算独立样本数。终止本轮有限修复，不启动新训练/专家/层位搜索。','']
p=H/'REPORT.md';assert '## 唯一一次公开披露' not in p.read_text();p.write_text(p.read_text()+'\n'.join(L));print({'direct_gain':direct['net_pp'],'direct_better':direct['better_n'],'direct_worse':direct['worse_n'],'solver_main':ss['balanced_mIoU_percent']['gn_bt'],'case_pages_reviewed':len(reviews)})
