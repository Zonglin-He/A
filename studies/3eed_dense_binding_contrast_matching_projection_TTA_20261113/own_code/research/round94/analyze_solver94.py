"""Disclosed secondary execution diagnostic, task totals and all solver negatives."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round94';P=R/'crossplatform/round94/solver_repair336_attempt001';rd=lambda p:json.loads(p.read_text())
def dump(n,x):(H/(n+'.json')).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
T=rd(H/'score_solver336_attempt001/ALL_TASK_ROWS_PRIVATE.json');old=rd(H/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');oldby={x['id_PRIVATE']:x for x in old};su=rd(H/'score_solver336_attempt001/TASK_SUMMARY.json')['ALL336'];arms=rd(H/'score_solver336_attempt001/SCORING_RECEIPT.json')['arm_names'];methods=[];effects=[];updates=[];trials=[];execution=[]
for a in arms:
 for p in ['balanced','drone','quad']:methods.append({'platform':p,'arm':a,'n':336 if p=='balanced' else 168,'mIoU_percent':su['balanced_mIoU_percent'][a] if p=='balanced' else su[p][a]['mIoU_percent'],'Acc25_percent':float(np.mean([su[q][a]['Acc25_percent'] for q in ['drone','quad']])) if p=='balanced' else su[p][a]['Acc25_percent'],'Acc50_percent':float(np.mean([su[q][a]['Acc50_percent'] for q in ['drone','quad']])) if p=='balanced' else su[p][a]['Acc50_percent']})
for a,b in [('gn_bt',b) for b in ['R91_frozen','binding_direct','binding_optimal_mixture','binding_bt1','gradient_bt','gn_mixture_bt','gn_shuffle_bt','gn_raw']]:
 for p in ['balanced','drone','quad']:
  tt=T if p=='balanced' else [x for x in T if x['platform']==p];v=np.array([x['IoU'][a]-x['IoU'][b] for x in tt]);effects.append({'arm':a,'reference':b,'platform':p,'n':len(tt),'net_pp':float(v.mean()*100),'better_n':int((v>1e-10).sum()),'worse_n':int((v<-1e-10).sum()),'unchanged_n':int((abs(v)<=1e-10).sum()),'query_changes':sum(x['queries'][a]!=x['queries'][b] for x in tt)})
for t in T:
 rec=rd(P/'records_PRIVATE'/(t['id_PRIVATE']+'.json'))
 for name,r in rec.items():
  updates.append({'row':t['row'],'platform':t['platform'],'method':name,**{k:v for k,v in r.items() if k!='trials'},'trial_n':len(r['trials'])})
  for i,x in enumerate(r['trials']):trials.append({'row':t['row'],'platform':t['platform'],'method':name,'attempt':i+1,**{k:v for k,v in x.items() if k!='state_key'}})
 for a in ['binding_bt1','gn_bt','gradient_bt','gn_mixture_bt','gn_shuffle_bt']:
  o=oldby[t['id_PRIVATE']];execution.append({'row':t['row'],'platform':t['platform'],'arm':a,'teacher_changes_source_top1':o['queries']['binding_direct']!=o['queries']['R91_frozen'],'matches_teacher_top1':t['queries'][a]==o['queries']['binding_direct'],'current_IoU':t['IoU'][a],'postseal_not_update_gate':True})
for n,x in [('SOLVER_METHOD_ROWS',methods),('SOLVER_EFFECT_ROWS',effects),('SOLVER_UPDATE_ROWS',updates),('SOLVER_TRIAL_ROWS',trials),('SOLVER_EXECUTION_ROWS',execution)]:dump(n,x)
d=rd(H/'DECISION.json');s=su['balanced_mIoU_percent'];d.update(single_secondary_solver_repair_after_initial_DEV=True,secondary_solver={'arm':'gn_bt','mIoU_percent':s['gn_bt'],'gains_pp':{b:s['gn_bt']-s[b] for b in ['R91_frozen','binding_direct','binding_optimal_mixture','binding_bt1','gradient_bt','gn_shuffle_bt']},'robustly_adopted':False,'not_promoted_to_original_primary':True},dense_direct_retained_as_development_evidence=True,dense_direct_mIoU_percent=s['binding_direct'],dense_direct_gain_vs_ownzero_pp=s['binding_direct']-s['R91_frozen'],new_TTA_recipe_adopted=False);dump('DECISION',d);print(d['secondary_solver'])
