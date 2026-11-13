"""Whole-cohort task/feedback/projection contribution tables, all negatives and fixed case rules."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round93';P=R/'crossplatform/round93/prediction336_attempt001';read=lambda p:json.loads(p.read_text())
def dump(n,x):(H/(n+'.json')).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
tasks=read(H/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');su=read(H/'score_dev336_attempt001/TASK_SUMMARY.json')['ALL336'];arms=read(P/'SEALED.json')['arm_names'];rows=read(H/'experiment/DEV336_PRIVATE.json');ev=read(P/'EVIDENCE_COVERAGE_ROWS.json');dirs=read(P/'DIRECTION_UPDATE_ROWS.json');method=[];effects=[];updates=[];trials=[];coverage=[]
for a in arms:
 method.append({'platform':'balanced','arm':a,'mIoU_percent':su['balanced_mIoU_percent'][a],'Acc25_percent':float(np.mean([su[p][a]['Acc25_percent'] for p in ['drone','quad']])),'Acc50_percent':float(np.mean([su[p][a]['Acc50_percent'] for p in ['drone','quad']]))})
 for p in ['drone','quad']:method.append({'platform':p,'arm':a,**{k:su[p][a][k] for k in ['n','mIoU_percent','Acc25_percent','Acc50_percent','unknown_n','fallback_n','no_parameter_update_n']}})
pairs=[('competition_projected_bt',r,'primary') for r in ['R91_frozen','R91_mean3','R91_feature3','competition_direct','R92_bt','R92_raw_static','competition_sum_bt','competition_only_bt','current_top1_bt','reverse_preference_bt','wrong_binding_bt','competition_projected_raw']]+[('competition_direct','R91_frozen','direct_feedback'),('competition_direct','reverse_preference_direct','direct_preference_direction'),('competition_sum_bt','competition_only_bt','physical_contribution')]
for a,b,kind in pairs:
 for p in ['balanced','drone','quad']:
  tt=tasks if p=='balanced' else [t for t in tasks if t['platform']==p];dv=np.array([t['IoU'][a]-t['IoU'][b] for t in tt]);effects.append({'component':kind,'arm':a,'reference':b,'platform':p,'n':len(tt),'net_pp':float(dv.mean()*100),'better_n':int((dv>1e-10).sum()),'worse_n':int((dv<-1e-10).sum()),'unchanged_n':int((abs(dv)<=1e-10).sum()),'query_changes':sum(t['queries'][a]!=t['queries'][b] for t in tt),'largest_positive_contribution_pp':float(max(0,dv.max())*100/len(dv)),'largest_negative_contribution_pp':float(min(0,dv.min())*100/len(dv))})
for r,t in zip(rows,tasks):
 for a,x in read(P/'records_PRIVATE'/(r['id']+'.json')).items():
  updates.append({'row':t['row'],'platform':r['platform'],'arm':a,**{k:v for k,v in x.items() if k not in ['trials','state_key']},'trial_n':len(x['trials'])})
  if a.endswith('_bt'):
   for k,x in enumerate(x['trials']):trials.append({'row':t['row'],'platform':r['platform'],'arm':a,'attempt':k+1,**{n:v for n,v in x.items() if n!='state_key'}})
for p in ['pooled','drone','quad']:
 ee=ev if p=='pooled' else [x for x in ev if x['platform']==p];ids={x['row'] for x in ee};tt=[x for x in tasks if x['row'] in ids];coverage.append({'platform':p,'n':len(ee),'spatial_eligible_observations':sum(x['spatial_eligible_pairs_n']>0 for x in ee),'feedback_observations':sum(x['accepted_pairs_n']>0 for x in ee),'accepted_pair_count':sum(x['accepted_pairs_n'] for x in ee),'observations_with_pair_opposing_zero_top1':sum(x['accepted_pairs_opposing_zero_top1_n']>0 for x in ee),'direct_changes_top1':sum(x['direct_changes_top1'] for x in ee),'roles_missing_n':sum(x['roles_missing'] for x in ee),'postseal_not_gate':True})
for cohort,fn in [('feedback',lambda x:x['accepted_pairs_n']>0),('no_feedback',lambda x:x['accepted_pairs_n']==0),('opposes_zero',lambda x:x['accepted_pairs_opposing_zero_top1_n']>0)]:
 ids={x['row'] for x in ev if fn(x)};tt=[x for x in tasks if x['row'] in ids]
 if not tt:continue
 for a,b in [('competition_projected_bt','R91_frozen'),('competition_projected_bt','competition_direct'),('competition_direct','R91_frozen'),('competition_projected_bt','R92_bt')]:
  dv=np.array([x['IoU'][a]-x['IoU'][b] for x in tt]);coverage.append({'platform':'pooled','cohort':cohort,'arm':a,'reference':b,'n':len(tt),'net_pp':float(dv.mean()*100),'better_n':int((dv>1e-10).sum()),'worse_n':int((dv<-1e-10).sum()),'query_changes':sum(x['queries'][a]!=x['queries'][b] for x in tt),'postseal_not_gate':True})
scene=[];sc=read(H/'score_dev336_attempt001/ALL_SCENE_SUMMARIES_PRIVATE.json')
for p in ['drone','quad']:
 names=sorted({x['scene_PRIVATE'] for x in sc if x['platform']==p})
 for x in [x for x in sc if x['platform']==p]:
  for a in arms:scene.append({'platform':p,'scene_alias':p[0].upper()+'_S'+str(names.index(x['scene_PRIVATE'])+1),'n':x['n'],'arm':a,'mIoU_percent':x['mIoU_percent'][a]})
chosen=[];why={}
def take(i,reason):
 if i not in chosen:chosen.append(i)
 why.setdefault(i,[]).append(reason)
for c in read(R/'research/round92/experiment/GOOD_BAD_CASES_PRIVATE.json'):
 if c['case'] in ['CASE002','CASE003','CASE004','CASE005']:take(next(i for i,r in enumerate(rows) if r['id']==c['row']['id']),'fixed_R92_'+c['case'])
for a,b in [('competition_projected_bt','R91_frozen'),('competition_projected_bt','competition_direct'),('competition_projected_bt','R92_bt'),('competition_direct','R91_frozen')]:
 dv=np.array([t['IoU'][a]-t['IoU'][b] for t in tasks])
 for i in [i for i in np.argsort(-dv,kind='stable') if dv[i]>1e-10][:2]:take(int(i),'gain_'+a+'_vs_'+b)
 for i in [i for i in np.argsort(dv,kind='stable') if dv[i]<-1e-10][:2]:take(int(i),'harm_'+a+'_vs_'+b)
cases=[{'case':'CASE'+str(j+1).zfill(3),'row':rows[i],'task':tasks[i],'evidence':ev[i],'reasons':why[i]} for j,i in enumerate(chosen)];(H/'experiment/GOOD_BAD_CASES_PRIVATE.json').write_text(json.dumps(cases,indent=2,ensure_ascii=False)+'\n');casepub=[{'case':x['case'],'platform':x['task']['platform'],'reasons':';'.join(x['reasons']),'accepted_pairs_n':x['evidence']['accepted_pairs_n'],**{a+'_IoU':x['task']['IoU'][a] for a in arms}} for x in cases]
for n,x in [('ALL_METHOD_ROWS',method),('COMPONENT_EFFECT_ROWS',effects),('ALL_SCENE_ROWS',scene),('EVIDENCE_COVERAGE_ROWS',ev),('EVIDENCE_SUMMARY_ROWS',coverage),('DIRECTION_UPDATE_ROWS',dirs),('ALL_UPDATE_ROWS',updates),('ALL_ACCEPTANCE_TRIAL_ROWS',trials),('GOOD_BAD_CASE_ROWS',casepub),('HELDOUT_ROWS',read(H/'independent_native_audit_attempt001/HELDOUT_ROWS.json'))]:dump(n,x)
s=su['balanced_mIoU_percent'];a='competition_projected_bt';dec={'state':'R93_COMPLETED_FIXED_R91_COMPETITION_GUIDED_PHYSICAL_DIRECTION_DEVELOPMENT_EVALUATED','mainline':'R91_R76_full_disc','primary':a,'primary_mIoU_percent':s[a],'gains_pp':{r:s[a]-s[r] for r in ['R91_frozen','R91_mean3','R91_feature3','competition_direct','R92_bt','competition_sum_bt','competition_only_bt','current_top1_bt','reverse_preference_bt','wrong_binding_bt']},'feedback_observations':sum(x['accepted_pairs_n']>0 for x in ev),'conflicting_feedback_observations':sum(x['conflict'] and x['task_active'] for x in dirs if x['method']=='competition_projected'),'observations_with_preference_opposing_zero_top1':sum(x['accepted_pairs_opposing_zero_top1_n']>0 for x in ev),'own_zero_update_gain_positive':s[a]>s['R91_frozen'],'independent_confirmation':False,'robustly_adopted':False,'architecture_retained_as_user_selected_mainline':True,'source_training':False,'FINAL_access':False,'no_follow_on_queued':True};dump('DECISION',dec);print(dec)
