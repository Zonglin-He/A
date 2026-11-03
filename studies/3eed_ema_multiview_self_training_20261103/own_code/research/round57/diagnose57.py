"""Descriptive saved-outcome/optimization analysis; cannot change any prediction."""
from common57 import *
H=ROOT/'research/round57';tasks=read(H/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');pred=ROOT/'crossplatform/round57/adapted336_attempt001';fits=read(H/'ALL_FIT_DIAGNOSTICS_PUBLIC_SAFE.json');a=read(H/'ANALYSIS_SUMMARY.json');pairs=[]
for arm,ref in [('epi_ema_teacher_1','frozen_ABC'),('epi_ema_teacher_3','frozen_ABC'),('epi_ema_teacher_16','frozen_ABC'),('online_ema_AB_teacher_post','frozen_ABC'),('online_ema_AB_student_post','online_fixed_AB_student_post')]:
 d=np.array([t['IoU'][arm]-t['IoU'][ref] for t in tasks]);changed=[i for i,t in enumerate(tasks) if t['queries'][arm]!=t['queries'][ref]];rr={'arm':arm,'reference':ref,'improve':int((d>1e-10).sum()),'worsen':int((d< -1e-10).sum()),'query_changes':len(changed),'balanced_delta_pp':float(d.mean()*100),'improved_scene_clusters':len({(t['platform'],t['scene_PRIVATE']) for t,v in zip(tasks,d) if v>1e-10}),'worsened_scene_clusters':len({(t['platform'],t['scene_PRIVATE']) for t,v in zip(tasks,d) if v< -1e-10})};pairs.append(rr)
kl={}
for kind in ['fixed','ema']:
 kl[kind]={}
 for k in [1,3,16]:
  rr=[r for r in fits if r['protocol']=='episodic' and r['teacher']==kind and r['step']==k];kl[kind][str(k)]={'median_KL_before':float(np.median([r['KL_before'] for r in rr])),'median_KL_after_same_target':float(np.median([r['KL_after_same_detached_target'] for r in rr])),'loss_decreased_cases':sum(r['KL_after_same_detached_target']<r['KL_before'] for r in rr)}
for kind in ['fixed','ema']:
 records=[read(pred/'cases_PRIVATE'/(t['id_PRIVATE']+'.json'))['episodic'][kind]['trace'] for t in tasks]
 kl[kind]['end_to_end']={'median_initial_KL':float(np.median([r[0]['KL_before'] for r in records])),'median_final_KL':float(np.median([r[-1]['KL_after_same_detached_target'] for r in records])),'final_loss_below_initial_cases':sum(r[-1]['KL_after_same_detached_target']<r[0]['KL_before'] for r in records),'targets_move_in_EMA_comparison':kind=='ema'}
res={'pairwise_task_diagnostics':pairs,'episodic_optimization':kl,'interpretation':'Postseal descriptive analysis, not deployment filtering, identity truth, hyperparameter selection or new method.'};dump(H/'POSTSEAL_MECHANISM_DIAGNOSTICS.json',res);print(res)
