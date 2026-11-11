"""Postseal target totals, all cases, source curves and physical-TTA diagnostics."""
from common76 import *
import collections
T=read(H/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');P=BASE/'target336_attempt001';methods=[];sr=[];frames=collections.defaultdict(list);ledger=[];scenevals=sorted({(x['platform'],x['scene_PRIVATE']) for x in T});sm={t:f'RECORDING_{i+1:02d}' for i,t in enumerate(scenevals)};manifest=target_rows();labels={x['id_PRIVATE']:x['candidate_IoUs_PRIVATE'] for x in read(ROOT/'research/round55/score_dev336_attempt001/ALL_CANDIDATE_IOUS_PRIVATE.json')}
for scope,tt in [('ALL336',T)]+[(s,[x for x in T if x['split']==s]) for s in sorted({x['split'] for x in T})]:
 for pl in ['balanced','drone','quad']:
  uu=tt if pl=='balanced' else [x for x in tt if x['platform']==pl]
  for k in ARMS:
   a=np.array([x['IoU'][k] for x in uu]);row={'scope':scope,'platform':pl,'method':k,'n':len(uu),'mIoU_percent':float(a.mean()*100),'Acc25_percent':float((a>.25).mean()*100),'Acc50_percent':float((a>.5).mean()*100)}
   for ref in ['source','independent_frozen','rgb_to_point_frozen','bidirectional_frozen','bidirectional_frozen_budget3','bidirectional_TTA3_wrong_correspondence','R59']:
    b=np.array([x['IoU'][ref] for x in uu]);d=a-b;row.update({f'gain_vs_{ref}_pp':float(d.mean()*100),f'improve_vs_{ref}':int((d>1e-10).sum()),f'worse_vs_{ref}':int((d< -1e-10).sum()),f'changes_vs_{ref}':sum(x['queries'][k]!=x['queries'][ref] for x in uu),f'rescue25_vs_{ref}':int(((a>.25)&(b<=.25)).sum()),f'break25_vs_{ref}':int(((a<=.25)&(b>.25)).sum()),f'largest_positive_contribution_vs_{ref}_pp':float(d.max()/len(uu)*100),f'remainder_after_largest_vs_{ref}_pp':float((d.sum()-d.max())/len(uu)*100)})
   methods.append(row)
for (pl,s),alias in sm.items():
 tt=[x for x in T if (x['platform'],x['scene_PRIVATE'])==(pl,s)]
 for k in ARMS:sr.append({'platform':pl,'recording_alias':alias,'method':k,'n':len(tt),'mIoU_percent':float(np.mean([x['IoU'][k] for x in tt])*100)})
for x,r in zip(T,manifest):
 frames[x['platform'],sha(r['imagepath']),sha(r['pointpath'])].append(x);y=np.array(labels[r['id']]);rec=read(P/'steps_PRIVATE'/(r['id']+'.json'));ledger.append({'row':x['row'],'platform':x['platform'],'split':x['split'],'source_failure25':x['IoU']['source']<=.25,'recoverable25_candidate':bool(y.max()>.25),'bidirectional_failure25':x['IoU']['bidirectional_frozen']<=.25,'source_max_candidate_IoU':float(y.max()),'visible_point_tokens':rec['visible_point_tokens'],'source_to_bidirectional_IoU_delta_pp':(x['IoU']['bidirectional_frozen']-x['IoU']['source'])*100,'TTA3_vs_bidirectional_delta_pp':(x['IoU']['bidirectional_TTA3']-x['IoU']['bidirectional_frozen'])*100,'reverse_off_vs_bidirectional_delta_pp':(x['IoU']['bidirectional_reverse_off']-x['IoU']['bidirectional_frozen'])*100,'wrong_RGB_vs_bidirectional_delta_pp':(x['IoU']['bidirectional_wrong_RGB']-x['IoU']['bidirectional_frozen'])*100})
fr=[]
for i,(key,tt) in enumerate(sorted(frames.items())):
 for k in ARMS:fr.append({'frame_alias':f'FRAME_{i+1:04d}','platform':key[0],'method':k,'descriptions':len(tt),'mean_IoU':float(np.mean([x['IoU'][k] for x in tt]))})
steps=read(H/'ALL_OPTIMIZER_STEP_ROWS.json');osum=[]
for pl in ['balanced','drone','quad']:
 for branch in ['correct','wrong_interaction_binding']:
  for t in [1,2,3]:
   a=[x for x in steps if x['branch']==branch and x['step']==t and (pl=='balanced' or x['platform']==pl)];osum.append({'platform':pl,'branch':branch,'step':t,'n':len(a),'accepted':sum(x['accepted'] for x in a),'loss_before_mean':float(np.mean([x['loss_before'] for x in a])),'loss_after_mean':float(np.mean([x['loss_after'] for x in a])),'aux_before_mean':float(np.mean([x['aux_before'] for x in a])),'aux_after_mean':float(np.mean([x['aux_after'] for x in a])),'keep_after_mean':float(np.mean([x['keep_after'] for x in a])),'mean_parameter_step_L2':float(np.mean([x['parameter_step_L2'] for x in a]))})
source=read(BASE/'trained_source_attempt002/HISTORY.json');old=read(BASE/'trained_source_attempt001/HISTORY_PROGRESS.json')
for x in source:x['attempt']='active002';x['used_for_final_source_selection']=True
for x in old:x['attempt']='interrupted001_unknown_RGB_bias';x['used_for_final_source_selection']=False
for n,v in [('ALL_METHOD_ROWS',methods),('ALL_SCENE_ROWS',sr),('ALL_FRAME_CLUSTER_ROWS',fr),('ALL_CASE_DIAGNOSTIC_ROWS',ledger),('OPTIMIZER_SUMMARY_ROWS',osum),('ALL_SOURCE_TRAINING_ROWS',source+old)]:dump(H/(n+'.json'),v)
dump(H/'COHORT_DIAGNOSTICS.json',{'state':'R76_EXPOSED_DEV336_SOURCE_PREPARED_TTA_DIAGNOSTIC','n':336,'recordings':len(scenevals),'frame_clusters':len(fr)//len(ARMS),'original_fixed_candidate_recoverable25':sum(x['recoverable25_candidate'] for x in ledger),'source_failure25_with_recoverable_candidate':sum(x['source_failure25'] and x['recoverable25_candidate'] for x in ledger),'bidirectional_failure25_with_recoverable_candidate':sum(x['bidirectional_failure25'] and x['recoverable25_candidate'] for x in ledger),'GT_only_postseal_diagnostic':True,'semantic_instance_identity_not_independently_certified':True})
print({'balanced':[(x['method'],x['mIoU_percent'],x['gain_vs_source_pp'],x['gain_vs_bidirectional_frozen_pp']) for x in methods if x['scope']=='ALL336' and x['platform']=='balanced'],'optimizer':osum[:6]})
