"""Sealed whole-DEV task, dense evidence utility, controls and fixed case rules."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round94';P=R/'crossplatform/round94/prediction336_attempt001';rd=lambda p:json.loads(p.read_text())
def dump(n,x):(H/(n+'.json')).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
tasks=rd(H/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');su=rd(H/'score_dev336_attempt001/TASK_SUMMARY.json')['ALL336'];arms=rd(P/'SEALED.json')['arm_names'];rows=rd(H/'experiment/DEV336_PRIVATE.json');method=[];effects=[];updates=[];trials=[]
for a in arms:
 method.append({'platform':'balanced','arm':a,'mIoU_percent':su['balanced_mIoU_percent'][a],'Acc25_percent':float(np.mean([su[p][a]['Acc25_percent'] for p in ['drone','quad']])),'Acc50_percent':float(np.mean([su[p][a]['Acc50_percent'] for p in ['drone','quad']]))})
 for p in ['drone','quad']:method.append({'platform':p,'arm':a,**{k:su[p][a][k] for k in ['n','mIoU_percent','Acc25_percent','Acc50_percent','unknown_n','fallback_n','no_parameter_update_n']}})
pairs=[('binding_bt1',b,'primary') for b in ['R91_frozen','R91_mean3','R59','R93_bt','binding_direct','binding_optimal_mixture','shuffle_bt1','reverse_bt1','viewmean_bt1','binding_raw1','binding_bt3']]+[('binding_direct',b,'evidence') for b in ['R91_frozen','shuffle_direct','reverse_direct','viewmean_direct']]+[('binding_optimal_mixture','R91_frozen','exact_zero_update_optimum')]
for a,b,kind in pairs:
 for p in ['balanced','drone','quad']:
  tt=tasks if p=='balanced' else [x for x in tasks if x['platform']==p];dv=np.array([x['IoU'][a]-x['IoU'][b] for x in tt]);effects.append({'component':kind,'arm':a,'reference':b,'platform':p,'n':len(tt),'net_pp':float(dv.mean()*100),'better_n':int((dv>1e-10).sum()),'worse_n':int((dv<-1e-10).sum()),'unchanged_n':int((abs(dv)<=1e-10).sum()),'query_changes':sum(x['queries'][a]!=x['queries'][b] for x in tt),'rescues_Acc25':sum(x['IoU'][a]>.25 and x['IoU'][b]<=.25 for x in tt),'breaks_Acc25':sum(x['IoU'][a]<=.25 and x['IoU'][b]>.25 for x in tt),'largest_positive_contribution_pp':float(max(0,dv.max())*100/len(tt)),'largest_negative_contribution_pp':float(min(0,dv.min())*100/len(tt))})
for r,t in zip(rows,tasks):
 for name,recs in rd(P/'records_PRIVATE'/(r['id']+'.json')).items():
  for rec in recs:
   updates.append({'row':t['row'],'platform':r['platform'],'method':name,**{k:v for k,v in rec.items() if k not in ['trials','endpoint_key']},'trial_n':len(rec['trials'])})
   for i,tri in enumerate(rec['trials']):trials.append({'row':t['row'],'platform':r['platform'],'method':name,'step':rec['step'],'attempt':i+1,**{k:v for k,v in tri.items() if k!='state_key'}})
ys={x['id_PRIVATE']:np.array(x['candidate_IoUs_PRIVATE']) for x in rd(R/'research/round55/score_dev336_attempt001/ALL_CANDIDATE_IOUS_PRIVATE.json')};utility=[]
for r,t in zip(rows,tasks):
 ev=np.load(P/'evidence_PRIVATE'/(r['id']+'.npz'));y=ys[r['id']];good=y>.25;q=t['queries']['R91_frozen'];p0=np.log(np.load(P/'samples_PRIVATE'/(r['id']+'.npz'))['scores'][arms.index('R91_frozen')]);target=ev['binding_target_PRIVATE'];recover=y[q]<=.25 and good.any();delta=ev['delta_PRIVATE'];d0=ev['correct_base_PRIVATE'].mean(0)-ev['wrong_base_PRIVATE'][0].mean(0);d1=ev['correct_base_PRIVATE'].mean(0)-ev['wrong_base_PRIVATE'][1].mean(0);cor=float(np.corrcoef(d0,d1)[0,1]) if min(d0.std(),d1.std())>1e-12 else 0.;utility.append({'row':t['row'],'platform':r['platform'],'source_IoU':float(y[q]),'oracle_IoU':float(y.max()),'postseal_recoverable_Acc25':bool(recover),'dense_target_ranks_any_acceptable_above_wrong_source':bool(recover and np.any(target[good]>target[q])),'direct_rescues_Acc25':bool(t['IoU']['binding_direct']>.25 and t['IoU']['R91_frozen']<=.25),'direct_breaks_Acc25':bool(t['IoU']['binding_direct']<=.25 and t['IoU']['R91_frozen']>.25),'direct_changes_top1':t['queries']['binding_direct']!=q,'two_derangement_correction_correlation':cor,'postseal_not_identity_truth_or_update_gate':True})
scene=[];sc=rd(H/'score_dev336_attempt001/ALL_SCENE_SUMMARIES_PRIVATE.json')
for p in ['drone','quad']:
 names=sorted({x['scene_PRIVATE'] for x in sc if x['platform']==p})
 for x in [x for x in sc if x['platform']==p]:
  for a in arms:scene.append({'platform':p,'scene_alias':p[0].upper()+'_S'+str(names.index(x['scene_PRIVATE'])+1),'n':x['n'],'arm':a,'mIoU_percent':x['mIoU_percent'][a]})
chosen=[];why={}
def take(i,reason):
 if i not in chosen:chosen.append(i)
 why.setdefault(i,[]).append(reason)
for c in rd(R/'research/round93/experiment/GOOD_BAD_CASES_PRIVATE.json')[:4]:take(next(i for i,r in enumerate(rows) if r['id']==c['row']['id']),'fixed_R93_'+c['case'])
for a,b in [('binding_bt1','R91_frozen'),('binding_bt1','binding_optimal_mixture'),('binding_direct','R91_frozen')]:
 dv=np.array([t['IoU'][a]-t['IoU'][b] for t in tasks])
 for i in [int(i) for i in np.argsort(-dv,kind='stable') if dv[i]>1e-10][:2]:take(i,'gain_'+a+'_vs_'+b)
 for i in [int(i) for i in np.argsort(dv,kind='stable') if dv[i]<-1e-10][:2]:take(i,'harm_'+a+'_vs_'+b)
cases=[{'case':'CASE'+str(j+1).zfill(3),'row':rows[i],'task':tasks[i],'reasons':why[i]} for j,i in enumerate(chosen)];(H/'experiment/GOOD_BAD_CASES_PRIVATE.json').write_text(json.dumps(cases,indent=2,ensure_ascii=False)+'\n');pub=[{'case':c['case'],'platform':c['task']['platform'],'reasons':';'.join(c['reasons']),**{a+'_IoU':c['task']['IoU'][a] for a in arms}} for c in cases]
for n,x in [('ALL_METHOD_ROWS',method),('COMPONENT_EFFECT_ROWS',effects),('ALL_SCENE_ROWS',scene),('EVIDENCE_COVERAGE_ROWS',rd(P/'EVIDENCE_COVERAGE_ROWS.json')),('EVIDENCE_UTILITY_ROWS',utility),('ALL_UPDATE_ROWS',updates),('ALL_ACCEPTANCE_TRIAL_ROWS',trials),('GOOD_BAD_CASE_ROWS',pub)]:dump(n,x)
s=su['balanced_mIoU_percent'];refs=['R91_frozen','R91_mean3','R59','R93_bt','binding_direct','binding_optimal_mixture','shuffle_bt1','reverse_bt1','viewmean_bt1'];d={'state':'R94_COMPLETED_DENSE_BINDING_CONTRAST_FINAL_PROJECTION_DEVELOPMENT_EVALUATED','mainline':'R91_R76_full_disc','primary':'binding_bt1','primary_mIoU_percent':s['binding_bt1'],'gains_pp':{b:s['binding_bt1']-s[b] for b in refs},'corrective_target_coverage_of_recoverable_Acc25':sum(x['dense_target_ranks_any_acceptable_above_wrong_source'] for x in utility),'recoverable_Acc25':sum(x['postseal_recoverable_Acc25'] for x in utility),'target_direct_rescues_Acc25':sum(x['direct_rescues_Acc25'] for x in utility),'target_direct_breaks_Acc25':sum(x['direct_breaks_Acc25'] for x in utility),'passes_fixed_development_increment_checks':all(s['binding_bt1']>s[b] for b in ['R91_frozen','binding_direct','binding_optimal_mixture','shuffle_bt1','reverse_bt1']),'robustly_adopted':False,'independent_confirmation':False,'source_training':False,'FINAL_access':False,'new_architecture':False,'no_follow_on_queued':True};dump('DECISION',d);print(d)
