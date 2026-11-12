"""Single post-seal larger-cohort IoU join, keeping cohorts and all negatives separate."""
from extension_common78 import *
import itertools
P=B78/'extension_prediction2304_attempt001';ps=read(P/'SEALED.json');audit=read(H78/'independent_extension_native_audit_attempt001/AUDIT_RESULT.json');assert audit['state'].startswith('PASS_') and audit['prediction_seal_sha256']==sha(P/'SEALED.json');old=ROOT/'research/round69/score_all2304_attempt001';ls=read(old/'SEALED.json');lf=old/'ALL_CANDIDATE_IOUS_PRIVATE.json';assert sha(lf)==ls['files'][lf.name];labels={r['id_PRIVATE']:r['candidate_IoUs_PRIVATE'] for r in read(lf)};cohort=read(ROOT/'research/round69/input/COHORT_BY_ID_PRIVATE.json');status={r['id']:r['arm_statuses'] for r in read(P/'READOUT_STATUS_PRIVATE.json')};arms=ps['arm_names'];tasks=[]
for i,r in enumerate(ext_rows()):
 f=P/'samples_PRIVATE'/(r['id']+'.npz');assert sha(f)==ps['files'][f.relative_to(P).as_posix()];z=load(f);qq=dict(zip(arms,map(int,z['queries'])));tasks.append({'row':f'E{i+1:05d}','id_PRIVATE':r['id'],'platform':r['platform'],'scene_PRIVATE':r['scene'],'part':cohort[r['id']],'split':'FIXED_R78_EXTENSION','queries':qq,'IoU':{a:float(labels[r['id']][qq[a]]) for a in arms},'statuses':status[r['id']]})
for name,n in [('VAL1280',1280),('TRAIN1024',1024)]:
 rr=[r for r in tasks if r['part']==name];assert len(rr)==n;summary={};scenes=[]
 for pl in ['drone','quad']:
  pr=[r for r in rr if r['platform']==pl];summary[pl]={}
  for a in arms:
   v=np.array([r['IoU'][a] for r in pr]);s={'n':len(pr),'mIoU_percent':float(v.mean()*100),'Acc25_percent':float((v>.25).mean()*100),'Acc50_percent':float((v>.5).mean()*100),'scene_clusters':len({r['scene_PRIVATE'] for r in pr})}
   for ref in ['source','frozen_ABC','R59','R76_bidirectional_frozen','joint0','joint1','joint_direct3','meta0','meta_direct3','meta_wrong1']:
    d=v-np.array([r['IoU'][ref] for r in pr]);s.update({f'gain_vs_{ref}_pp':float(d.mean()*100),f'improved_vs_{ref}':int((d>1e-10).sum()),f'worsened_vs_{ref}':int((d<-1e-10).sum()),f'query_changes_vs_{ref}':sum(r['queries'][a]!=r['queries'][ref] for r in pr)})
   for k in ['unknown','fallback','no_parameter_update']:s[k+'_n']=sum(r['statuses'][a][k] for r in pr)
   summary[pl][a]=s
  for sc in sorted({r['scene_PRIVATE'] for r in pr}):
   sr=[r for r in pr if r['scene_PRIVATE']==sc];scenes.append({'scope':'ALL'+str(n),'platform':pl,'scene_PRIVATE':sc,'n':len(sr),'mIoU_percent':{a:float(np.mean([r['IoU'][a] for r in sr])*100) for a in arms}})
 summary['balanced_mIoU_percent']={a:float(np.mean([summary[pl][a]['mIoU_percent'] for pl in ['drone','quad']])) for a in arms};summary['micro_mIoU_percent']={a:float(np.mean([r['IoU'][a] for r in rr])*100) for a in arms};neg=[{'row':r['row'],'platform':r['platform'],'part':r['part'],'split':r['split'],'arm':a,'reference':b,'delta_pp':100*(r['IoU'][a]-r['IoU'][b])} for r in rr for a,b in itertools.permutations(arms,2) if r['IoU'][a]-r['IoU'][b]<-1e-10];out=H78/('score_'+name.lower()+'_attempt001');assert not out.exists();out.mkdir();receipt={'state':'PASS_R78_FIXED_SOURCE_SELECTED_EXTENSION_POSTSEAL_SAVED_LABEL_JOIN','n':n,'arm_names':arms,'task_values':n*len(arms),'new_GT_scored_task_values':0,'reused_saved_task_values':n*len(arms),'ordered_negative_comparisons':len(neg),'single_saved_IoU_join':True,'new_raw_target_GT_or_geometry_calls':0,'source_selection_uses_target_labels':False,'prediction_seal_sha256':sha(P/'SEALED.json')}
 for filename,x in [('ALL_TASK_ROWS_PRIVATE',rr),('TASK_SUMMARY',{'ALL'+str(n):summary}),('ALL_SCENE_SUMMARIES_PRIVATE',scenes),('ALL_ORDERED_NEGATIVE_COMPARISONS',neg),('SCORING_RECEIPT',receipt),('SHARED_CALIBRATION_SELECTION',None)]:dump(out/(filename+'.json'),x)
 seal(out,receipt['state']);print(name,summary['balanced_mIoU_percent']);print(receipt)
