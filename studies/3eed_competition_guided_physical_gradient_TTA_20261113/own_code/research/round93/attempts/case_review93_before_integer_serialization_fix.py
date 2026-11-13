"""Post-seal manual review recorded after viewing all eight real-image pages."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round93';P=R/'crossplatform/round93/prediction336_attempt001';rd=lambda p:json.loads(p.read_text());cases=rd(H/'experiment/GOOD_BAD_CASES_PRIVATE.json');tasks=rd(H/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');notes={
'CASE001':('different_visible_objects','原近处车辆仍被主TTA和直接偏好改到远处上沿区域。task-only和反向控制保住原框。该例接受反馈真实反对原第一名，故错误不只是物理项单独造成。'),
'CASE002':('same_candidate_wrong_visible_person','主BT和直接使用仍选中另一行人；raw仍可改到左侧主体。唯一接受偏好不包含原第一名，未覆盖最终关键竞争者。不能由两候选偏好推出它超过未监督第一名。'),
'CASE003':('same_visible_object_duplicate_boxes','同一SUV附近的重复框由0.2332改到0.2724，沿用原物理后备；该例没有接受pair，不归因于新增竞争监督。'),
'CASE004':('same_visible_object_duplicate_boxes','仍是左侧同一行人附近较差重复框，0.4461降到0.3280。只保护原第一名的控制能避免，反向偏好BT仍未避免。'),
'CASE005':('different_visible_regions_target_identity_partial','主TTA和直接读出从远处背景回到右侧车辆附近，但仅0.1443，目标边缘裁切，精确物体身份保留部分未知。胜出的是接受pair之外的第三候选，反向BT也得到相同框，不能认证为正确偏好带来的身份纠错。'),
'CASE006':('same_candidate','主BT保持白色车辆原框，直接使用偏好则转向前景深色车辆并降到零。此项TTA相对直接的改善是避免直接重排损伤，没有超过零步。'),
'CASE007':('same_candidate','主BT保留原SUV框；直接使用改到同一SUV更合适的重复框，0.2468到0.2748，跨过Acc25。参数更新没有执行这份有益变化。'),
'CASE008':('same_visible_object_duplicate_boxes','同一前景车辆区域两个很差重复框从0.0060到0.0014；高IoU候选一直存在但排名很低。无接受pair，属于原物理后备，不能叫新增语义身份错误。')}
review=[]
for c in cases:
 ident,note=notes[c['case']];review.append({'case':c['case'],'platform':c['task']['platform'],'full_scene_caption_crop_BEV_reviewed':True,'primary_zero_to_TTA_identity':ident,'observation':note,'postseal_descriptive_not_deployment_rule':True})
(H/'CASE_REVIEW_ANONYMOUS_ROWS.json').write_text(json.dumps(review,indent=2,ensure_ascii=False)+'\n');worst=0.;changes=scalechanges=0;differentproposal=0
for r in rd(H/'experiment/DEV336_PRIVATE.json'):
 z=np.load(P/'samples_PRIVATE'/(r['id']+'.npz'));s=dict(zip(z['arm_names'],z['scores']));worst=max(worst,float(abs(s['competition_projected_bt']-s['competition_sum_bt']).max()));changes+=int(s['competition_projected_bt'].argmax()!=s['competition_sum_bt'].argmax());rec=rd(P/'records_PRIVATE'/(r['id']+'.json'));scalechanges+=rec['competition_projected_bt']['scale']!=rec['competition_sum_bt']['scale'];g=np.load(P/'gradients_PRIVATE'/(r['id']+'.npz'));differentproposal+=np.max(abs(g['competition_projected_proposal']-g['competition_sum_proposal']))>1e-10
raw=np.array([x['IoU']['competition_projected_raw']-x['IoU']['R91_frozen'] for x in tasks]);d=rd(H/'DECISION.json');d.update(projection_vs_sum_Top1_changes=changes,projection_vs_sum_max_probability_difference=worst,projection_vs_sum_scale_changes=scalechanges,projection_vs_sum_nonidentical_proposals=differentproposal,primary_raw_mIoU_percent=23.203782000561535,raw_vs_ownzero_better_n=int((raw>1e-10).sum()),raw_vs_ownzero_worse_n=int((raw<-1e-10).sum()),raw_largest_positive_contribution_pp=float(raw.max()*100/336),reviewed_cases=len(review));summary=rd(H/'score_dev336_attempt001/TASK_SUMMARY.json')['ALL336'];d['primary_raw_mIoU_percent']=summary['balanced_mIoU_percent']['competition_projected_raw'];(H/'DECISION.json').write_text(json.dumps(d,indent=2)+'\n')
lines=['','## 真实原图复核与未采用的raw正信号','',f'主方法与sum在{differentproposal}/336条提案实际不同，{scalechanges}/336条接受scale改变、概率最大差{worst:.10g}，最终Top1差{changes}/336。梯度手术确实执行，但未提供额外选框收益。','',f'预注册raw为{d["primary_raw_mIoU_percent"]:.6f}%，相对零步{d["primary_raw_mIoU_percent"]-summary["balanced_mIoU_percent"]["R91_frozen"]:+.6f}pp，{d["raw_vs_ownzero_better_n"]}改善/{d["raw_vs_ownzero_worse_n"]}恶化。Drone {summary["drone"]["competition_projected_raw"]["mIoU_percent"]:.6f}%、Quad {summary["quad"]["competition_projected_raw"]["mIoU_percent"]:.6f}%；前者比零步更低、后者提高。最大单例正贡献{d["raw_largest_positive_contribution_pp"]:+.6f}pp。保留此开发正结果，但current_top1_raw更高，sum_raw与主raw相同，不能归因于正确竞争偏好或投影。没有事后将raw替换成锁定BT主方法。','', '| 案例 | 原图身份变化/不确定性 | 实际观察 |','|---|---|---|']
for x in review:lines.append(f'| {x["case"]} | {x["primary_zero_to_TTA_identity"]} | {x["observation"]} |')
lines+=['','关键两例没有同时解决：原近处车辆的误改仍在；原可由raw纠正的行人，BT仍压回。大部分接受pair确认既有排序，少数真实反对第一名的反馈仍可指错。因此，不把下一步重新简化成“增大梯度”或“保护第一名”；首先需要在模型已有候选竞争中提供能覆盖实际错误第一名、且不依赖其同源自信的可测量纠错依据。当前结论不要求换架构、重训source或另扫层位。','']
p=H/'REPORT.md';assert '## 真实原图复核' not in p.read_text();p.write_text(p.read_text()+'\n'.join(lines));print({'reviewed':len(review),'different_proposals':differentproposal,'BTscale_changes':scalechanges,'BT_Top1_changes_vs_sum':changes,'raw_better':int((raw>1e-10).sum()),'raw_worse':int((raw<-1e-10).sum())})
