"""Postseal aggregate analysis, saved predictions and anonymous scalar outputs."""
from common74 import *
import collections
p=read(H/'experiment/ALL2304_PLAN_LOCK.json');P=ROOT/p['prediction_root'];E=ROOT/p['evidence_root'];F=ROOT/p['fit_root'];tasks=read(H/'score_all2304_attempt001/ALL_TASK_ROWS_PRIVATE.json');arms=p['all_task_arms'];rawrows=read(ROOT/'research/round69/input/ALL2304_PRIVATE.json');media=read(ROOT/'research/round69/input/SELECTED_MEDIA_HASHES_PRIVATE.json');print('media_schema',type(media).__name__)
if isinstance(media,list): media={x['id']:x for x in media}
methods=[];scene_rows=[];frames=collections.defaultdict(list);steps=[];coverage=[];auxsum=collections.defaultdict(lambda:collections.Counter());auxrows=[];diag=[]
scenes=sorted({(x['part'],x['platform'],x['scene_PRIVATE']) for x in tasks});sm={t:f'RECORDING_{i+1:02d}' for i,t in enumerate(scenes)}
for x in tasks:
 sid=x['id_PRIVATE'];ev=load(E/'samples_PRIVATE'/(sid+'.npz'));au=load(P/'aux_PRIVATE'/(sid+'.npz'));aux=read(P/'cases_PRIVATE'/(sid+'.json'));row=x['row'];a=read(E/'cases_PRIVATE'/(sid+'.json'));frame=(x['part'],x['platform'],media[sid]['image_sha256'],media[sid]['point_sha256']);frames[frame].append(x)
 coverage.append({'row':row,'part':x['part'],'platform':x['platform'],'phase':x['phase'],'regions':len(ev['region_boxes']),'RGB_negative_pairs':int(ev['RGB_negatives'].sum()),'geometry_negative_pairs':int(ev['geometry_negatives'].sum()),'RGB_noRGB_mask_differs':bool(np.any(ev['RGB_negatives']!=ev['geometry_negatives'])),'C_valid_regions':int(au['eval_negatives'].any(1).sum()),'C_unknown_regions':int((~au['eval_negatives'].any(1)).sum())})
 for k in ADAPT:
  r=read(F/'cases_PRIVATE'/(sid+'_'+k+'.json'));steps.append({'row':row,'part':x['part'],'platform':x['platform'],'phase':x['phase'],'method':k,'active':r['active'],'accepted':r['accepted'],'scale':r.get('scale',0),'loss_before':r.get('loss_before'),'loss_after':r.get('loss_after'),'parameter_change_L2':r.get('parameter_step_L2',0.),'gradient_norm':r.get('gradient_norm'),'trials':len(r.get('trials',[]))})
 for k in ['source']+ADAPT:
  for direction,ra,rb in [('AB_to_C',au[k+'_AB'],au[k+'_C']),('C_to_AB',au[k+'_C'],au[k+'_AB'])]:
   N=au['eval_negatives'];valid=N.any(1);sim=ra@rb.T;allowed=N|np.eye(len(N),dtype=bool);hit=int((np.where(allowed,sim,-np.inf).argmax(1)[valid]==np.flatnonzero(valid)).sum());gap=float((sim.diagonal()[valid]-np.where(N,sim,-np.inf).max(1)[valid]).sum());pos=float(sim.diagonal()[valid].sum());saved=aux[k][direction];assert hit==saved['hits'] and int(valid.sum())==saved['valid'];assert abs(gap-saved['gap_sum'])<1e-5 and abs(pos-saved['positive_cosine_sum'])<1e-5
   for scope in ['ALL2304',x['part'],x['phase']]:
    v=auxsum[(scope,k,direction)];v.update(n=1,valid=int(valid.sum()),hits=hit,unknown=int((~valid).sum()),gap=gap,pos=pos,macro_accuracy=(hit/valid.sum()) if valid.any() else 0,nonempty=int(valid.any()))
for (scope,k,d),v in auxsum.items():auxrows.append({'scope':scope,'method':k,'direction':d,'n':v['n'],'valid_regions':v['valid'],'unknown_regions':v['unknown'],'hits':v['hits'],'micro_accuracy_percent':v['hits']/v['valid']*100,'macro_nonempty_accuracy_percent':v['macro_accuracy']/v['nonempty']*100,'mean_gap':v['gap']/v['valid'],'mean_positive_cosine':v['pos']/v['valid']})
scopes=[('ALL2304',tasks),('VAL1280',[x for x in tasks if x['part']=='VAL1280']),('TRAIN1024',[x for x in tasks if x['part']=='TRAIN1024']),('PILOT64',[x for x in tasks if x['phase']=='PILOT64']),('REMAINDER2240',[x for x in tasks if x['phase']=='REMAINDER2240'])]
for scope,tt in scopes:
 for pl in ['balanced','drone','quad']:
  uu=tt if pl=='balanced' else [x for x in tt if x['platform']==pl]
  if pl=='balanced':assert sum(x['platform']=='drone' for x in uu)==len(uu)//2
  for k in arms:
   v=np.array([x['IoU'][k] for x in uu]);r={'scope':scope,'platform':pl,'method':k,'n':len(uu),'mIoU_percent':v.mean()*100,'Acc25_percent':(v>.25).mean()*100,'Acc50_percent':(v>.5).mean()*100}
   for ref in ['source','frozen_AB','frozen_AB_repr','frozen_region_consensus','R59','corr_shuffled','heldout_C_source']:
    b=np.array([x['IoU'][ref] for x in uu]);d=v-b;r.update({f'gain_vs_{ref}_pp':d.mean()*100,f'improve_vs_{ref}':int((d>1e-10).sum()),f'worse_vs_{ref}':int((d< -1e-10).sum()),f'query_change_vs_{ref}':sum(x['queries'][k]!=x['queries'][ref] for x in uu),f'rescue25_vs_{ref}':int(((v>.25)&(b<=.25)).sum()),f'break25_vs_{ref}':int(((v<=.25)&(b>.25)).sum())})
   methods.append(r)
  for k,ref in [('corr_RGB','frozen_AB'),('corr_RGB','frozen_AB_repr'),('corr_RGB','corr_shuffled'),('heldout_C_corr_RGB','heldout_C_source')]:
   d=np.array([x['IoU'][k]-x['IoU'][ref] for x in uu]);j=int(d.argmax());diag.append({'scope':scope,'platform':pl,'method':k,'reference':ref,'n':len(uu),'gain_pp':d.mean()*100,'largest_positive_row':uu[j]['row'],'largest_positive_contribution_pp':d[j]/len(uu)*100,'remainder_contribution_pp':(d.sum()-d[j])/len(uu)*100})
for t,alias in sm.items():
 uu=[x for x in tasks if (x['part'],x['platform'],x['scene_PRIVATE'])==t]
 for k in arms:scene_rows.append({'part':t[0],'platform':t[1],'recording_alias':alias,'n':len(uu),'method':k,'mIoU_percent':np.mean([x['IoU'][k] for x in uu])*100})
fr=[]
for i,(f,tt) in enumerate(sorted(frames.items())):
 for k in arms:fr.append({'frame_alias':f'FRAME_{i+1:05d}','part':f[0],'platform':f[1],'descriptions':len(tt),'method':k,'mean_IoU':float(np.mean([x['IoU'][k] for x in tt]))})
frame_summary=[]
for part in ['VAL1280','TRAIN1024']:
 for k in arms:
  z=[x for x in fr if x['part']==part and x['method']==k];by={pl:[x['mean_IoU'] for x in z if x['platform']==pl] for pl in ['drone','quad']};frame_summary.append({'part':part,'method':k,'frames':len(z),'drone_frame_mean_percent':np.mean(by['drone'])*100,'quad_frame_mean_percent':np.mean(by['quad'])*100,'balanced_frame_mean_percent':np.mean([np.mean(v) for v in by.values()])*100})
for name,val in [('ALL_METHOD_ROWS',methods),('ALL_SCENE_ROWS',scene_rows),('ALL_FRAME_CLUSTER_ROWS',fr),('FRAME_BALANCED_SUMMARY',frame_summary),('ALL_COVERAGE_ROWS',coverage),('ALL_OPTIMIZER_STEP_ROWS',steps),('REGION_RETRIEVAL_ROWS',auxrows),('LARGEST_CONTRIBUTIONS',diag)]:dump(H/(name+'.json'),val)
dump(H/'COHORT_DIAGNOSTICS.json',{'state':'R74_FULL_SAVED_COHORT_DESCRIPTIVE_ANALYSIS','unique_inputs':len(tasks),'recordings':len(scenes),'frame_clusters':len(fr)//len(arms),'regions':sum(x['regions'] for x in coverage),'RGB_noRGB_mask_different_inputs':sum(x['RGB_noRGB_mask_differs'] for x in coverage),'active_updates':sum(x['active'] for x in steps),'accepted_updates':sum(x['accepted'] for x in steps),'accepted_loss_increases':sum(x['accepted'] and x['loss_after']>x['loss_before'] for x in steps),'RGB_noRGB_task_differences':sum(x['queries']['corr_RGB']!=x['queries']['corr_noRGB'] for x in tasks),'C_RGB_noRGB_task_differences':sum(x['queries']['heldout_C_corr_RGB']!=x['queries']['heldout_C_corr_noRGB'] for x in tasks),'C_auxiliary_not_fitting':True,'historial_exposure_not_blind_test':True})
print(read(H/'COHORT_DIAGNOSTICS.json'));print([r for r in methods if r['scope']=='ALL2304' and r['platform']=='balanced'])
