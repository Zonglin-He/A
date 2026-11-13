"""Postseal method/component totals, all update rows and symmetric case selection."""
from target_core89 import *
P=B89/'prediction336_attempt001';tasks=read(H89/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');su=read(H89/'score_dev336_attempt001/TASK_SUMMARY.json')['ALL336'];rows=read(H89/'experiment/DEV336_PRIVATE.json');method=[];components=[];updates=[];case_rows=[];cases=[];selected=[];reasons={};ys={r['id_PRIVATE']:r['candidate_IoUs_PRIVATE'] for r in read(ROOT/'research/round55/score_dev336_attempt001/ALL_CANDIDATE_IOUS_PRIVATE.json')}
for arm in ARMS89:
 method.append({'platform':'balanced','arm':arm,'mIoU_percent':su['balanced_mIoU_percent'][arm]})
 for pl in ['drone','quad']:method.append({'platform':pl,'arm':arm,**su[pl][arm]})
comparisons=[('candidate_frozen','frozen_bi','source_prepared_interface'),('candidate_frozen','continue_frozen','same_source_budget_matching_continuation'),('candidate_frozen','query_frozen','query_only_source_prepared_control'),('candidate_tta1','candidate_frozen','parameter_update'),('candidate_tta1','candidate_mean3','same_information_three_read_control'),('candidate_tta1','R59','old_strong_no_update_control'),('candidate_tta3','candidate_tta1','secondary_three_steps'),('candidate_tta1','candidate_physical1','correspondence_plus_rank'),('candidate_tta1','candidate_no_correspondence1','local_correspondence')]+[('candidate_'+s,v+'_'+s,v) for v in VARIANTS89 for s in ['frozen','tta1']]
for a,b,name in comparisons:
 for pl in ['balanced','drone','quad']:
  tt=tasks if pl=='balanced' else [t for t in tasks if t['platform']==pl];d=np.array([t['IoU'][a]-t['IoU'][b] for t in tt]);components.append({'component':name,'arm':a,'reference':b,'platform':pl,'net_pp':float(d.mean()*100),'better_n':int((d>1e-10).sum()),'worse_n':int((d< -1e-10).sum()),'unchanged_n':int((abs(d)<=1e-10).sum()),'largest_positive_pp_of_total':float(max(0,d.max())*100/len(d)),'largest_negative_pp_of_total':float(min(0,d.min())*100/len(d))})
for r,t in zip(rows,tasks):
 for variant,tr in read(P/'records_PRIVATE'/(r['id']+'.json')).items():
  for st in tr:updates.append({'row':t['row'],'platform':r['platform'],'mode':variant,**{k:v for k,v in st.items() if k not in ['trials','parts_before','parts_after']},**{'before_'+k:v for k,v in st['parts_before'].items()},**{'after_'+k:v for k,v in st['parts_after'].items()}})
def take(ix,why):
 for i in ix:
  if i not in selected:selected.append(i)
  reasons.setdefault(i,[]).append(why)
for arm,ref in [('candidate_frozen','frozen_bi'),('candidate_tta1','candidate_frozen'),('candidate_tta1','candidate_mean3')]:
 d=np.array([t['IoU'][arm]-t['IoU'][ref] for t in tasks]);take([i for i in np.argsort(-d,kind='stable') if d[i]>1e-10][:2],'gain_'+arm+'_vs_'+ref);take([i for i in np.argsort(d,kind='stable') if d[i]<-1e-10][:2],'harm_'+arm+'_vs_'+ref)
v=np.array([t['IoU']['candidate_tta1'] for t in tasks]);take(np.argsort(-v,kind='stable')[:1],'absolute_high');take([i for i in np.argsort(v,kind='stable') if max(ys[tasks[i]['id_PRIVATE']])>.25 and v[i]<=.25][:1],'recoverable_failure')
for k,i in enumerate(selected):
 alias='CASE'+str(k+1).zfill(3);t=tasks[i];cases.append({'case':alias,'row':rows[i],'task':t,'reasons':reasons[i]});a=load(P/'samples_PRIVATE'/(rows[i]['id']+'.npz'));bb=a['source_boxes'];qa=t['queries']['candidate_frozen'];qb=t['queries']['candidate_tta1'];A=bb[qa];B=bb[qb];inter=np.maximum(0,np.minimum(A[:3]+A[3:]/2,B[:3]+B[3:]/2)-np.maximum(A[:3]-A[3:]/2,B[:3]-B[3:]/2)).prod();bi=float(inter/max(1e-12,A[3:].prod()+B[3:].prod()-inter));case_rows.append({'case':alias,'platform':t['platform'],'reasons':';'.join(reasons[i]),**{a+'_IoU':t['IoU'][a] for a in ARMS89},'frozen_TTA_selected_box_overlap':bi,'selected_center_shift':float(np.linalg.norm(A[:3]-B[:3])),'identity_not_inferred_from_box_IoU':True})
fixed=[]
for c in read(ROOT/'research/round88/experiment/GOOD_BAD_CASES_PRIVATE.json'):
 if c['case'] in ['CASE002','CASE004','CASE007']:
  i=next(i for i,r in enumerate(rows) if r['id']==c['row']['id']);fixed.append({'case':'R88_'+c['case'],'row':rows[i],'task':tasks[i],'reasons':['fixed_prior_case_followup']})
scene=[];sr=read(H89/'score_dev336_attempt001/ALL_SCENE_SUMMARIES_PRIVATE.json')
for pl in ['drone','quad']:
 vals=sorted({r['scene_PRIVATE'] for r in sr if r['platform']==pl})
 for s in [s for s in sr if s['platform']==pl]:
  for arm in ARMS89:scene.append({'platform':pl,'scene_alias':pl[0].upper()+'_S'+str(vals.index(s['scene_PRIVATE'])+1),'n':s['n'],'arm':arm,'mIoU_percent':s['mIoU_percent'][arm]})
for k,v in [('ALL_METHOD_ROWS',method),('COMPONENT_EFFECT_ROWS',components),('ALL_UPDATE_ROWS',updates),('GOOD_BAD_CASE_ROWS',case_rows),('ALL_SCENE_ROWS',scene),('FIXED_R88_CASE_ROWS',[{'case':c['case'],'platform':c['task']['platform'],**{a+'_IoU':c['task']['IoU'][a] for a in ARMS89}} for c in fixed])]:dump(H89/(k+'.json'),v)
dump(H89/'experiment/GOOD_BAD_CASES_PRIVATE.json',cases);dump(H89/'experiment/FIXED_R88_CASES_PRIVATE.json',fixed)
gains={a:su['balanced_mIoU_percent']['candidate_tta1']-su['balanced_mIoU_percent'][a] for a in ['frozen_bi','candidate_frozen','candidate_mean3','R59','wrong_space_tta1','role_swap_tta1']};sourcecomp=read(H89/'SOURCE_COMPONENT_ROWS.json');source={arm:float(np.mean([r[arm+'_IoU'] for r in sourcecomp])*100) for arm in ['prepared_full','frozen_R76','no_ref','shared_mix','query_only']};dump(H89/'SOURCE_VALIDATION_COMPONENT_SUMMARY.json',source)
dump(H89/'DECISION.json',{'state':'R89_COMPLETED_SOURCE_PREPARED_CANDIDATE_BINDING_DEVELOPMENT_EVALUATED','primary_arm':'candidate_tta1','primary_gains_pp':gains,'passes_matched_increment_checks':all(gains[a]>0 for a in ['frozen_bi','candidate_frozen','candidate_mean3','R59']),'robust_adoption':False,'independent_confirmation':False,'FINAL_access':False,'source_training':True,'new_external_vision_models':False,'no_follow_on_queued':True});print({'gains_pp':gains,'cases':len(cases),'source_VAL':source})
