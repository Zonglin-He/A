"""Whole-cohort components, scene clusters, sealed case rules, and source prerequisites."""
from targets91 import *
H=H91;P=B91/'prediction336_attempt001';tasks=read(H/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');su=read(H/'score_dev336_attempt001/TASK_SUMMARY.json')['ALL336'];arms=LEGACY91+NEW91;method=[];comp=[];scene=[];updates=[]
for a in arms:
 method.append({'platform':'balanced','arm':a,'mIoU_percent':su['balanced_mIoU_percent'][a],'Acc25_percent':float(np.mean([su[pl][a]['Acc25_percent'] for pl in ['drone','quad']])),'Acc50_percent':float(np.mean([su[pl][a]['Acc50_percent'] for pl in ['drone','quad']]))})
 for pl in ['drone','quad']:method.append({'platform':pl,'arm':a,**{k:su[pl][a][k] for k in ['n','mIoU_percent','Acc25_percent','Acc50_percent','unknown_n','fallback_n','no_parameter_update_n']}})
pairs=[]
for b in BASES91:
 n=b+'_full_disc';baseline='R76_frozen' if b=='R76' else 'R89_continue_frozen'
 pairs.extend([(n+'_frozen',baseline,'source_preparation'),(n+'_frozen',b+'_query_disc_frozen','spatial_vs_query'),(n+'_frozen',b+'_full_ce_frozen','source_competition_loss'),(n+'_frozen',b+'_no_ref_disc_frozen','reference_reading'),(n+'_frozen',b+'_no_local_disc_frozen','local_visual_evidence')])
 pairs.extend([(n+'_physical_bt',r,'adaptation_or_control') for r in [n+'_frozen',n+'_mean3',n+'_feature3',n+'_physical_raw',n+'_conditional_bt',n+'_wrong_physical_bt','R80_raw','R82_bt','R59',b+'_query_disc_physical_bt',b+'_full_ce_physical_bt',b+'_no_ref_disc_physical_bt',b+'_no_local_disc_physical_bt']]);pairs.append((n+'_physical_raw',n+'_wrong_physical_raw','matched_raw_correspondence'));pairs.append((n+'_conditional_raw',n+'_physical_raw','matched_raw_conditional_target'))
for a,b,name in pairs:
 for pl in ['balanced','drone','quad']:
  tt=tasks if pl=='balanced' else [t for t in tasks if t['platform']==pl];dv=np.array([t['IoU'][a]-t['IoU'][b] for t in tt]);comp.append({'component':name,'arm':a,'reference':b,'platform':pl,'n':len(tt),'net_pp':float(dv.mean()*100),'better_n':int((dv>1e-10).sum()),'worse_n':int((dv< -1e-10).sum()),'unchanged_n':int((abs(dv)<=1e-10).sum()),'largest_positive_contribution_pp':float(max(0,dv.max())*100/len(dv)),'largest_negative_contribution_pp':float(min(0,dv.min())*100/len(dv))})
for t,r in zip(tasks,read(H/'experiment/DEV336_PRIVATE.json')):
 for a,x in read(P/'records_PRIVATE'/(r['id']+'.json')).items():updates.append({'row':t['row'],'platform':r['platform'],'arm':a,**{k:v for k,v in x.items() if k not in ['trials','patch_key','feature_key']},'attempt_count':len(x['trials'])})
sc=read(H/'score_dev336_attempt001/ALL_SCENE_SUMMARIES_PRIVATE.json')
for pl in ['drone','quad']:
 ss=sorted({s['scene_PRIVATE'] for s in sc if s['platform']==pl})
 for s in [s for s in sc if s['platform']==pl]:
  for a in arms:scene.append({'platform':pl,'scene_alias':pl[0].upper()+'_S'+str(ss.index(s['scene_PRIVATE'])+1),'n':s['n'],'arm':a,'mIoU_percent':s['mIoU_percent'][a]})
ys={r['id_PRIVATE']:r['candidate_IoUs_PRIVATE'] for r in read(ROOT/'research/round55/score_dev336_attempt001/ALL_CANDIDATE_IOUS_PRIVATE.json')};rows=read(H/'experiment/DEV336_PRIVATE.json');selected=[];why={}
coverage=[];roleinfo=[]
for t,r in zip(tasks,rows):
 rt=load(B89/'target_evidence336_attempt001/samples_PRIVATE'/(r['id']+'.npz'));roleinfo.append({'row':t['row'],'platform':r['platform'],'oracle_IoU':float(max(ys[r['id']])),'role_count':len(rt['token_roles']),'reference_count':max(0,len(rt['token_roles'])-1),'spatial_missing':len(rt['token_roles'])==0,'has_own_point_support':bool(rt['own_support'].any())})
for name,test in [('all',lambda x:True),('oracle_above25',lambda x:x['oracle_IoU']>.25),('oracle_at_most25',lambda x:x['oracle_IoU']<=.25),('missing_spatial_roles',lambda x:x['spatial_missing']),('reference_available',lambda x:x['reference_count']>0)]:
 for pl in ['pooled','drone','quad']:
  ids={x['row'] for x in roleinfo if test(x) and (pl=='pooled' or x['platform']==pl)};tt=[t for t in tasks if t['row'] in ids]
  if not tt:continue
  for a,b in [('R76_full_disc_frozen','R76_frozen'),('R76_full_disc_physical_bt','R76_full_disc_frozen'),('R76_full_disc_physical_bt','R76_full_disc_feature3')]:
   dv=np.array([t['IoU'][a]-t['IoU'][b] for t in tt]);coverage.append({'cohort':name,'platform':pl,'n':len(tt),'arm':a,'reference':b,'net_pp':float(dv.mean()*100),'better_n':int((dv>1e-10).sum()),'worse_n':int((dv< -1e-10).sum()),'postseal_only_not_deployment_gate':True})
dump(H/'COVERAGE_ROLE_ROWS.json',roleinfo);dump(H/'COVERAGE_COMPONENT_ROWS.json',coverage)
def take(indices,reason):
 for i in indices:
  if i not in selected:selected.append(i)
  why.setdefault(i,[]).append(reason)
for a,b in [('R76_full_disc_frozen','R76_frozen'),('R76_full_disc_physical_bt','R76_full_disc_frozen'),('R76_full_disc_physical_bt','R76_full_disc_feature3')]:
 v=np.array([t['IoU'][a]-t['IoU'][b] for t in tasks]);take([int(i) for i in np.argsort(-v,kind='stable') if v[i]>1e-10][:2],'gain_'+a+'_vs_'+b);take([int(i) for i in np.argsort(v,kind='stable') if v[i]<-1e-10][:2],'harm_'+a+'_vs_'+b)
v=np.array([t['IoU']['R80_raw']-t['IoU']['R76_full_disc_physical_bt'] for t in tasks]);take([int(i) for i in np.argsort(-v,kind='stable') if v[i]>1e-10 and tasks[i]['IoU']['R80_raw']>tasks[i]['IoU']['R76_frozen']][:1],'old_physical_TTA_gain_lost');take(np.argsort([-t['IoU']['R76_full_disc_physical_bt'] for t in tasks])[:1],'absolute_high');take([i for i,t in enumerate(tasks) if max(ys[t['id_PRIVATE']])>.25 and t['IoU']['R76_full_disc_physical_bt']<=.25][:1],'recoverable_failure');cases=[{'case':'CASE'+str(j+1).zfill(3),'row':rows[i],'task':tasks[i],'reasons':why[i]} for j,i in enumerate(selected)];fixed=[]
for c in read(H90/'experiment/GOOD_BAD_CASES_PRIVATE.json'):
 if c['case'] in ['CASE002','CASE003','CASE004','CASE005','CASE009','CASE012']:
  i=next(i for i,r in enumerate(rows) if r['id']==c['row']['id']);fixed.append({'case':'R90_'+c['case'],'row':rows[i],'task':tasks[i],'reasons':['fixed_R90_case']})
dump(H/'experiment/GOOD_BAD_CASES_PRIVATE.json',cases);dump(H/'experiment/FIXED_R90_CASES_PRIVATE.json',fixed)
case_table=[{'case':c['case'],'platform':c['task']['platform'],'reasons':';'.join(c['reasons']),**{a+'_IoU':c['task']['IoU'][a] for a in arms}} for c in cases+fixed]
source=read(H/'score_source128_attempt001/TASK_SUMMARY.json')['ALL128'];source_rows0=[{'arm':a,**v} for a,v in source.items()];src_task=read(H/'score_source128_attempt001/ALL_TASK_ROWS_PRIVATE.json');src_flat=[{'row':r['row'],'arm':a,'IoU':r['IoU'][a],**r['statuses'][a]} for r in src_task for a in NEW91];history=read(B91/'trained_source_attempt001/HISTORY.json');safehist=[{k:v for k,v in x.items() if k not in ['state_path','state_sha256']} for x in history]
for k,v in [('ALL_METHOD_ROWS',method),('COMPONENT_EFFECT_ROWS',comp),('ALL_SCENE_ROWS',scene),('ALL_UPDATE_ROWS',updates),('GOOD_BAD_CASE_ROWS',case_table),('SOURCE_AUX_TASK_ROWS',src_flat),('SOURCE_AUX_SUMMARY_ROWS',source_rows0),('SOURCE_TRAINING_HISTORY_ROWS',safehist)]:dump(H/(k+'.json'),v)
s=su['balanced_mIoU_percent'];primary='R76_full_disc_physical_bt';gains={a:s[primary]-s[a] for a in ['R76_frozen','R76_full_disc_frozen','R76_full_disc_mean3','R76_full_disc_feature3','R76_query_disc_physical_bt','R76_full_ce_physical_bt','R76_no_ref_disc_physical_bt','R76_no_local_disc_physical_bt','R76_full_disc_conditional_bt','R76_full_disc_wrong_physical_bt','R80_raw','R82_bt','R59']};sel=read(B91/'trained_source_attempt001/SELECTED_SOURCE_ONLY.json')['models'];source_better={b:sel[b+'_full_disc']['source_val_mIoU_percent']>max(sel[b+'_query_disc']['source_val_mIoU_percent'],sel[b+'_full_ce']['source_val_mIoU_percent'],sel[b+'_no_ref_disc']['source_val_mIoU_percent'],sel[b+'_no_local_disc']['source_val_mIoU_percent']) for b in BASES91};decision={'state':'R91_COMPLETED_FIXED_STRONG_BASE_DISCRIMINATION_AND_PHYSICAL_TTA_DEVELOPMENT_EVALUATED','primary':primary,'gains_pp':gains,'source_discriminative_binding_prerequisite':source_better,'primary_above_both_same_information_zero_update_controls':gains['R76_full_disc_mean3']>0 and gains['R76_full_disc_feature3']>0,'full_recipe_robustly_adopted':False,'independent_confirmation':False,'original_fusion_preserved_during_source_preparation':True,'target_updates_only_original_102498_scope':True,'source_non_target_box_annotations_used':True,'reference_role_identity_GT_used':False,'no_target_GT_for_update_or_selection':True,'FINAL_access':False,'no_follow_on_queued':True};dump(H/'DECISION.json',decision);print(decision)
