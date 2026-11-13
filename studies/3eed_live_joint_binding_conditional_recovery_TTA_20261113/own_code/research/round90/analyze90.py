"""Sealed task totals, source prerequisites, full component effects and symmetric cases."""
from target90 import *
P=B90/'prediction336_attempt001';tasks=read(H90/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');su=read(H90/'score_dev336_attempt001/TASK_SUMMARY.json')['ALL336'];rows=read(H90/'experiment/DEV336_PRIVATE.json');methods=[];components=[];updates=[];part=[];ys={r['id_PRIVATE']:r['candidate_IoUs_PRIVATE'] for r in read(ROOT/'research/round55/score_dev336_attempt001/ALL_CANDIDATE_IOUS_PRIVATE.json')}
for a in ARMS90:
 methods.append({'platform':'balanced','arm':a,'mIoU_percent':su['balanced_mIoU_percent'][a]})
 for pl in ['drone','quad']:methods.append({'platform':pl,'arm':a,**su[pl][a]})
pairs=[('candidate_joint_frozen',b+'_frozen','source_preparation_'+b) for b in ['continue_joint','query_joint','candidate_headonly']]+[('candidate_joint_frozen','frozen_bi','new_source_prepared_structure')]+[('candidate_joint_tta1',b,'adaptation_or_control_'+b) for b in ['candidate_joint_frozen','candidate_joint_mean3','candidate_joint_feature3','R59','R89_continue_tta1','continue_joint_tta1','query_joint_tta1','candidate_headonly_tta1','candidate_joint_bypass1','candidate_joint_physical1','candidate_joint_shuffle1']]+[('candidate_joint_'+s,v+'_'+s,v) for v in VARIANTS90 for s in ['frozen','tta1']]+[('candidate_joint_tta3','candidate_joint_tta1','secondary_three_steps')]
for a,b,name in pairs:
 for pl in ['balanced','drone','quad']:
  tt=tasks if pl=='balanced' else [t for t in tasks if t['platform']==pl];v=np.array([t['IoU'][a]-t['IoU'][b] for t in tt]);components.append({'component':name,'arm':a,'reference':b,'platform':pl,'net_pp':float(v.mean()*100),'better_n':int((v>1e-10).sum()),'worse_n':int((v< -1e-10).sum()),'unchanged_n':int((abs(v)<=1e-10).sum()),'largest_positive_contribution_pp':float(max(0,v.max())*100/len(v)),'largest_negative_contribution_pp':float(min(0,v.min())*100/len(v))})
for r,t in zip(rows,tasks):
 rt=load(B89/'target_evidence336_attempt001/samples_PRIVATE'/(r['id']+'.npz'));known=bool(len(rt['token_roles']))
 for name,tr in read(P/'records_PRIVATE'/(r['id']+'.json')).items():
  for st in tr:updates.append({'row':t['row'],'platform':r['platform'],'mode':name,'spatial_roles_available':known,**{k:v for k,v in st.items() if k not in ['trials','parts_before','parts_after']},**{'before_'+k:v for k,v in st['parts_before'].items()},**{'after_'+k:v for k,v in st['parts_after'].items()}})
 part.append({'row':t['row'],'platform':r['platform'],'spatial_roles_available':known,'roles':len(rt['token_roles']),'reference_roles':max(0,len(rt['token_roles'])-1),'own_support_valid_n':int(rt['own_valid'].sum()),'original_candidate_oracle_IoU':float(max(ys[r['id']]))})
selected=[];reasons={}
def take(ix,reason):
 for i in ix:
  if i not in selected:selected.append(i)
  reasons.setdefault(i,[]).append(reason)
for arm,ref in [('candidate_joint_frozen','frozen_bi'),('candidate_joint_tta1','candidate_joint_frozen'),('candidate_joint_tta1','candidate_joint_feature3')]:
 v=np.array([t['IoU'][arm]-t['IoU'][ref] for t in tasks]);take([int(i) for i in np.argsort(-v,kind='stable') if v[i]>1e-10][:2],'gain_'+arm+'_vs_'+ref);take([int(i) for i in np.argsort(v,kind='stable') if v[i]<-1e-10][:2],'harm_'+arm+'_vs_'+ref)
v=np.array([t['IoU']['candidate_joint_tta1'] for t in tasks]);take(np.argsort(-v,kind='stable')[:1],'absolute_high');take([i for i in np.argsort(v,kind='stable') if max(ys[tasks[i]['id_PRIVATE']])>.25 and v[i]<=.25][:1],'recoverable_failure');cases=[];cr=[]
for j,i in enumerate(selected):
 t=tasks[i];alias='CASE'+str(j+1).zfill(3);cases.append({'case':alias,'row':rows[i],'task':t,'reasons':reasons[i]});cr.append({'case':alias,'platform':t['platform'],'reasons':';'.join(reasons[i]),**{a+'_IoU':t['IoU'][a] for a in ARMS90}})
fixed=[]
for c in read(H89/'experiment/GOOD_BAD_CASES_PRIVATE.json'):
 if c['case'] in ['CASE002','CASE003','CASE005','CASE008','CASE011']:
  i=next(i for i,r in enumerate(rows) if r['id']==c['row']['id']);fixed.append({'case':'R89_'+c['case'],'row':rows[i],'task':tasks[i],'reasons':['fixed_prior_failure_or_success']})
scene=[];sc=read(H90/'score_dev336_attempt001/ALL_SCENE_SUMMARIES_PRIVATE.json')
for pl in ['drone','quad']:
 vals=sorted({r['scene_PRIVATE'] for r in sc if r['platform']==pl})
 for s in [s for s in sc if s['platform']==pl]:
  for a in ARMS90:scene.append({'platform':pl,'scene_alias':pl[0].upper()+'_S'+str(vals.index(s['scene_PRIVATE'])+1),'n':s['n'],'arm':a,'mIoU_percent':s['mIoU_percent'][a]})
for k,v in [('ALL_METHOD_ROWS',methods),('COMPONENT_EFFECT_ROWS',components),('ALL_UPDATE_ROWS',updates),('SUPPORT_ROLE_ROWS',part),('GOOD_BAD_CASE_ROWS',cr),('ALL_SCENE_ROWS',scene),('FIXED_R89_CASE_ROWS',[{'case':c['case'],'platform':c['task']['platform'],**{a+'_IoU':c['task']['IoU'][a] for a in ARMS90}} for c in fixed])]:dump(H90/(k+'.json'),v)
dump(H90/'experiment/GOOD_BAD_CASES_PRIVATE.json',cases);dump(H90/'experiment/FIXED_R89_CASES_PRIVATE.json',fixed);ms=su['balanced_mIoU_percent'];gains={b:ms['candidate_joint_tta1']-ms[b] for b in ['frozen_bi','candidate_joint_frozen','candidate_joint_mean3','candidate_joint_feature3','continue_joint_tta1','R59','R89_continue_tta1','candidate_joint_bypass1','candidate_joint_physical1','candidate_joint_shuffle1']};source={r['arm']:r['mIoU_percent'] for r in read(H90/'SOURCE_AUX_TASK_SUMMARY_ROWS.json')};sg={b:source['candidate_joint_frozen']-source[b] for b in ['continue_joint_frozen','query_joint_frozen','candidate_headonly_frozen','no_ref_frozen','soft_pool_frozen']};d={'state':'R90_COMPLETED_JOINT_LEARNING_PATH_DEVELOPMENT_EVALUATED','primary_arm':'candidate_joint_tta1','primary_gains_pp':gains,'source_precondition_gains_pp':sg,'source_binding_prerequisite_established':all(sg[b]>0 for b in ['continue_joint_frozen','query_joint_frozen','candidate_headonly_frozen','no_ref_frozen']),'same_information_TTA_increment_positive':all(gains[b]>0 for b in ['candidate_joint_frozen','candidate_joint_mean3','candidate_joint_feature3']),'full_recipe_robustly_adopted':False,'independent_confirmation':False,'source_preparation':True,'postfusion_features_cached_as_constants':False,'live_shared_training_path_verified':True,'learned_router_updated_at_test':True,'FINAL_access':False,'no_follow_on_queued':True};dump(H90/'DECISION.json',d);print(d)
