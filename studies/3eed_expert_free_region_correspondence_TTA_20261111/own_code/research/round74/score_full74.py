"""One postseal saved-IoU join; split into official cohort and pilot tables."""
from common74 import *
import itertools
assert read(H/'STATE.json')['CPU_scoring_authorized'];p=read(H/'experiment/ALL2304_PLAN_LOCK.json');pred=ROOT/p['prediction_root'];ps=read(pred/'SEALED.json');cert=read(H/'independent_union_audit_attempt001/AUDIT_RESULT.json');assert cert['state'].startswith('PASS_') and cert['prediction_seal_sha256']==sha(pred/'SEALED.json');ARMS=p['all_task_arms'];rr=read(ROOT/'research/round69/input/ALL2304_PRIVATE.json');co=read(ROOT/'research/round69/input/COHORT_BY_ID_PRIVATE.json');pilot={r['id'] for r in rows()};labelroot=ROOT/'research/round69/score_all2304_attempt001';ls=read(labelroot/'SEALED.json');lf=labelroot/'ALL_CANDIDATE_IOUS_PRIVATE.json';assert sha(lf)==ls['files'][lf.name];labels={x['id_PRIVATE']:x['candidate_IoUs_PRIVATE'] for x in read(lf)};status={x['id']:x['arm_statuses'] for x in read(pred/'READOUT_STATUS_PRIVATE.json')};tasks=[]
for i,r in enumerate(rr):
 sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(pred/rel)==ps['files'][rel];z=load(pred/rel);assert list(z['arm_names'])==ARMS;qq=dict(zip(ARMS,map(int,z['queries'])));vv=labels[sid];tasks.append({'row':'R'+str(i+1).zfill(5),'id_PRIVATE':sid,'platform':r['platform'],'scene_PRIVATE':r['scene'],'part':co[sid],'split':co[sid],'phase':'PILOT64' if sid in pilot else 'REMAINDER2240','queries':qq,'IoU':{k:float(vv[qq[k]]) for k in ARMS},'statuses':status[sid]})
allroot=H/'score_all2304_attempt001';assert not allroot.exists();allroot.mkdir();dump(allroot/'ALL_TASK_ROWS_PRIVATE.json',tasks)
for cohort,n,name in [('VAL1280',1280,'val1280'),('TRAIN1024',1024,'train1024'),('PILOT64',64,'pilot64')]:
 tt=[x for x in tasks if x['phase']==cohort] if cohort=='PILOT64' else [x for x in tasks if x['part']==cohort];assert len(tt)==n;summary={};scenes=[];negative=[]
 for pl in ['drone','quad']:
  uu=[x for x in tt if x['platform']==pl];assert len(uu)==n//2;summary[pl]={}
  for k in ARMS:
   a=np.array([x['IoU'][k] for x in uu]);rec={'n':len(uu),'mIoU_percent':float(a.mean()*100),'Acc25_percent':float((a>.25).mean()*100),'Acc50_percent':float((a>.5).mean()*100),'unknown_n':sum(x['statuses'][k]['unknown'] for x in uu),'fallback_n':sum(x['statuses'][k]['fallback'] for x in uu),'no_parameter_update_n':sum(x['statuses'][k]['no_parameter_update'] for x in uu)}
   for ref in ['source','frozen_AB','frozen_AB_repr','frozen_region_consensus','corr_shuffled','heldout_C_source']:
    d=a-np.array([x['IoU'][ref] for x in uu]);rec.update({f'gain_vs_{ref}_pp':float(d.mean()*100),f'improved_vs_{ref}':int((d>1e-10).sum()),f'worsened_vs_{ref}':int((d< -1e-10).sum()),f'query_changes_vs_{ref}':sum(x['queries'][k]!=x['queries'][ref] for x in uu)})
   summary[pl][k]=rec
  for scene in sorted({x['scene_PRIVATE'] for x in uu}):
   sub=[x for x in uu if x['scene_PRIVATE']==scene];scenes.append({'scope':'ALL'+str(n),'platform':pl,'scene_PRIVATE':scene,'n':len(sub),'mIoU_percent':{k:float(np.mean([x['IoU'][k] for x in sub])*100) for k in ARMS}})
 summary['balanced_mIoU_percent']={k:float(np.mean([summary[pl][k]['mIoU_percent'] for pl in ['drone','quad']])) for k in ARMS}
 for x in tt:
  for k,ref in itertools.permutations(ARMS,2):
   d=x['IoU'][k]-x['IoU'][ref]
   if d< -1e-10:negative.append({'row':x['row'],'platform':x['platform'],'part':x['part'],'split':x['split'],'arm':k,'reference':ref,'delta_pp':float(d*100)})
 root=H/('score_'+name+'_attempt001');assert not root.exists();root.mkdir();receipt={'state':'PASS_R74_FIXED_SCOPE_POSTSEAL_SAVED_A256_IOU_JOIN','n':n,'arm_names':ARMS,'task_values':n*len(ARMS),'new_GT_scored_task_values':0,'reused_saved_task_values':n*len(ARMS),'ordered_negative_comparisons':len(negative),'single_saved_IoU_join':True,'shared_join_all2304_pilot_subset_not_independent':True,'no_new_raw_GT_or_geometry_score':True,'prediction_seal_sha256':sha(pred/'SEALED.json')}
 for f,v in [('ALL_TASK_ROWS_PRIVATE',tt),('TASK_SUMMARY',{'ALL'+str(n):summary}),('ALL_SCENE_SUMMARIES_PRIVATE',scenes),('ALL_ORDERED_NEGATIVE_COMPARISONS',negative),('SCORING_RECEIPT',receipt),('SHARED_CALIBRATION_SELECTION',None)]:dump(root/(f+'.json'),v)
 seal(root,receipt['state']);print({'cohort':cohort,'balanced_mIoU_percent':summary['balanced_mIoU_percent'],'negatives':len(negative)})
dump(allroot/'SCORING_RECEIPT.json',{'state':'PASS_R74_ONE_FULL_POPULATION_SAVED_IOU_JOIN','unique_inputs':2304,'unique_readouts':2304*len(ARMS),'pilot_inputs_reused_from_prior_mechanism_test':64,'remaining_inputs':2240,'new_raw_GT_scoring_calls':0});seal(allroot,'R74_FULL_FIXED_POPULATION_SAVED_TASK_VALUES');print({'unique_n':2304,'arms':len(ARMS)})
