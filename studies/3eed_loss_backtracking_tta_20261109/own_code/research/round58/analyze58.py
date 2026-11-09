"""Postseal aggregate/task and optimization diagnostics, no retuning or identity claims."""
from common58 import *
import collections
H=ROOT/'research/round58';P=ROOT/'crossplatform/round58/adapted336_attempt001';audit=read(H/'saved_scalar_audit_attempt001/AUDIT_RESULT.json');assert audit['state'].startswith('PASS_');tasks=read(H/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');s=read(H/'score_dev336_attempt001/TASK_SUMMARY.json')['ALL336'];scenes=[r for r in read(H/'score_dev336_attempt001/ALL_SCENE_SUMMARIES_PRIVATE.json') if r['scope']=='ALL336'];rows=[];process=[];first=[];pair=[]
for arm in ARMS:
 rows.append({'arm':arm,'Drone_mIoU_percent':s['drone'][arm]['mIoU_percent'],'Quad_mIoU_percent':s['quad'][arm]['mIoU_percent'],'balanced_mIoU_percent':s['balanced_mIoU_percent'][arm],'balanced_Acc25_percent':float(np.mean([s[p][arm]['Acc25_percent'] for p in ['drone','quad']])),'balanced_Acc50_percent':float(np.mean([s[p][arm]['Acc50_percent'] for p in ['drone','quad']])),'gain_vs_source_pp':s['balanced_mIoU_percent'][arm]-s['balanced_mIoU_percent']['source'],'gain_vs_frozen_ABC_pp':s['balanced_mIoU_percent'][arm]-s['balanced_mIoU_percent']['frozen_ABC'],'gain_vs_r56_one_pp':s['balanced_mIoU_percent'][arm]-s['balanced_mIoU_percent']['r56_one'],'improve_vs_source':sum(s[p][arm]['improved_vs_source'] for p in ['drone','quad']),'worsen_vs_source':sum(s[p][arm]['worsened_vs_source'] for p in ['drone','quad']),'query_changes_vs_source':sum(s[p][arm]['query_changes_vs_source'] for p in ['drone','quad']),'scene_mean_improve_vs_source':sum(r['mIoU_percent'][arm]>r['mIoU_percent']['source']+1e-10 for r in scenes),'scene_mean_worsen_vs_source':sum(r['mIoU_percent'][arm]<r['mIoU_percent']['source']-1e-10 for r in scenes)})
for arm in ARMS:
 if not arm.startswith('backtrack_'):continue
 ref='original_'+arm[len('backtrack_'):]
 for base in [ref,'frozen_ABC','r56_one']:
  delta=np.array([t['IoU'][arm]-t['IoU'][base] for t in tasks]);pair.append({'arm':arm,'reference':base,'balanced_delta_pp':float(delta.mean()*100),'improved':int((delta>1e-10).sum()),'worsened':int((delta< -1e-10).sum()),'query_changes':sum(t['queries'][arm]!=t['queries'][base] for t in tasks)})
for idx,t in enumerate(tasks):
 r=read(P/'cases_PRIVATE'/(t['id_PRIVATE']+'.json'))
 for key,data in r['episodic'].items():
  rule,teacher=key.split('_');trace=data['trace'];ff=trace[0];first.append({'row':f'DEV336_{idx+1:05d}','platform':t['platform'],'rule':rule,'teacher':teacher,'KL_before':ff['KL_before'],'KL_after_same_target':ff['KL_after_same_detached_target'],'accepted':ff['accepted'],'scale':ff['scale'],'trial_evaluations':ff['trial_count'],'nonzero_parameter_change':ff['student_step_L2']>0,'full_step_loss_increased':ff['full_step_loss_increased']})
  for step,x in enumerate(trace,1):process.append({'row':f'DEV336_{idx+1:05d}','platform':t['platform'],'protocol':'episodic','rule':rule,'teacher':teacher,'attempt':step,'KL_before':x['KL_before'],'KL_after_same_target':x['KL_after_same_detached_target'],'accepted':x['accepted'],'scale':x['scale'],'trial_evaluations':x['trial_count'],'student_step_L2':x['student_step_L2'],'teacher_step_L2':x['teacher_step_L2'],'full_step_loss_increased':x['full_step_loss_increased']})
 for key,x in r['online'].items():
  rule,teacher=key.split('_');process.append({'row':f'DEV336_{idx+1:05d}','platform':t['platform'],'protocol':'online','rule':rule,'teacher':teacher,'attempt':x['stream_index']+1,'KL_before':x['KL_before'],'KL_after_same_target':x['KL_after_same_detached_target'],'accepted':x['accepted'],'scale':x['scale'],'trial_evaluations':x['trial_count'],'student_step_L2':x['student_step_L2'],'teacher_step_L2':x['teacher_step_L2'],'full_step_loss_increased':x['full_step_loss_increased']})
grouped=[]
for protocol in ['episodic','online']:
 for rule in ['original','backtrack']:
  for teacher in ['fixed','ema']:
   rr=[r for r in process if r['protocol']==protocol and r['rule']==rule and r['teacher']==teacher];grouped.append({'protocol':protocol,'rule':rule,'teacher':teacher,'n_attempts':len(rr),'accepted':sum(r['accepted'] for r in rr),'rejected':sum(not r['accepted'] for r in rr),'accepted_nonzero_parameter_changes':sum(r['accepted'] and r['student_step_L2']>0 for r in rr),'loss_increased':sum(r['KL_after_same_target']>r['KL_before']+1e-12 for r in rr),'full_proposal_loss_increased':sum(r['full_step_loss_increased'] for r in rr),'trial_evaluations':sum(r['trial_evaluations'] for r in rr),'median_accepted_scale':float(np.median([r['scale'] for r in rr if r['accepted']])) if any(r['accepted'] for r in rr) else None,'median_KL_before':float(np.median([r['KL_before'] for r in rr])),'median_KL_after':float(np.median([r['KL_after_same_target'] for r in rr]))})
firstsummary=[]
for rule in ['original','backtrack']:
 for kind in ['fixed','ema']:
  rr=[r for r in first if r['rule']==rule and r['teacher']==kind];firstsummary.append({'rule':rule,'teacher':kind,'n':len(rr),'accepted':sum(r['accepted'] for r in rr),'rejected':sum(not r['accepted'] for r in rr),'nonzero_updates':sum(r['nonzero_parameter_change'] for r in rr),'loss_increased':sum(r['KL_after_same_target']>r['KL_before']+1e-12 for r in rr),'median_KL_before':float(np.median([r['KL_before'] for r in rr])),'median_KL_after':float(np.median([r['KL_after_same_target'] for r in rr])),'scale_counts':dict(collections.Counter(str(r['scale']) for r in rr))})
scenesafe=[]
for pl in ['drone','quad']:
 for no,r in enumerate(sorted([x for x in scenes if x['platform']==pl],key=lambda x:x['scene_PRIVATE']),1):
  for arm in ARMS:scenesafe.append({'platform':pl,'scene_cluster':no,'n':r['n'],'arm':arm,'mIoU_percent':r['mIoU_percent'][arm]})
online_history=[]
for pl in ['drone','quad']:
 for rule in ['original','backtrack']:
  for teacher in ['fixed','ema']:
   rr=[r for r in process if r['platform']==pl and r['protocol']=='online' and r['rule']==rule and r['teacher']==teacher];accepted=sum(r['accepted'] for r in rr);online_history.append({'platform':pl,'rule':rule,'teacher':teacher,'attempts':len(rr),'accepted_updates':accepted,'initial_teacher_term_coefficient':.99**accepted if teacher=='ema' else 1.})
result={'online_state_history':online_history,'n':336,'arms':rows,'backtracking_pairwise_task_contrasts':pair,'optimization_groups':grouped,'first_step_summary':firstsummary,'postseal_descriptive_only':True,'shared_hyperparameters_no_score_selected_configuration':True,'source_retraining':False,'protected_holdout_access':False};dump(H/'ANALYSIS_SUMMARY.json',result);dump(H/'ALL_PROCESS_SCALARS_PUBLIC_SAFE.json',process);dump(H/'CHECKPOINT_PROCESS_SCALARS_PUBLIC_SAFE.json',[r for r in process if r['protocol']=='online' or r['attempt'] in [1,3,16]]);dump(H/'FIRST_STEP_SCALARS_PUBLIC_SAFE.json',first);dump(H/'ALL_SCENE_METRICS_PUBLIC_SAFE.json',scenesafe);print(json.dumps({'arms':rows,'first_step':firstsummary,'groups':grouped}),flush=True)
