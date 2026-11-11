"""Postseal saved candidate-IoU joins; original labels/media never opened."""
from common71 import *
import itertools,time
assert read(H/'STATE.json')['CPU_scoring_authorized'];p=read(H/'experiment/SCORE_PLAN_LOCK.json');pin(p);pred=ROOT/p['prediction_root'];ps=read(pred/'SEALED.json');cert=read(H/'independent_audit_attempt001/AUDIT_RESULT.json');assert cert['state'].startswith('PASS_') and cert['prediction_seal_sha256']==sha(pred/'SEALED.json');old69=ROOT/'research/round69/score_all2304_attempt001';old55=ROOT/'research/round55/score_dev336_attempt001';labels={};prior={}
for old in [old69,old55]:
 ss=read(old/'SEALED.json')
 for name in ['ALL_CANDIDATE_IOUS_PRIVATE.json','ALL_TASK_ROWS_PRIVATE.json']:assert sha(old/name)==ss['files'][name]
 labels.update({x['id_PRIVATE']:x for x in read(old/'ALL_CANDIDATE_IOUS_PRIVATE.json')});prior.update({x['id_PRIVATE']:x for x in read(old/'ALL_TASK_ROWS_PRIVATE.json')})
old70=ROOT/'research/round70/score_dev336_attempt001';seal70=read(old70/'SEALED.json');assert sha(old70/'ALL_TASK_ROWS_PRIVATE.json')==seal70['files']['ALL_TASK_ROWS_PRIVATE.json'];prior.update({x['id_PRIVATE']:x for x in read(old70/'ALL_TASK_ROWS_PRIVATE.json')})
co=read(ROOT/p['cohort_map']);rr=rows()+devrows();status={x['id']:x['arm_statuses'] for x in read(pred/'READOUT_STATUS_PRIVATE.json')};tasks=[];start=time.perf_counter()
for i,r in enumerate(rr):
 sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(pred/rel)==ps['files'][rel];z=load(pred/rel);q=dict(zip(ARMS,map(int,z['queries'])));v=np.asarray(labels[sid]['candidate_IoUs_PRIVATE']);values={k:float(v[q[k]]) for k in ARMS};base=prior[sid]
 for k in ARMS[:3]:assert q[k]==base['queries'][k] and values[k]==base['IoU'][k]
 assert np.array_equal(z['selected_original_boxes'],z['source_boxes'][z['queries']]);part=co.get(sid,'DEV336_DIAGNOSTIC');tasks.append({'row':f'R{i+1:05d}','id_PRIVATE':sid,'platform':r['platform'],'scene_PRIVATE':r['scene'],'part':part,'split':part,'queries':q,'IoU':values,'statuses':status[sid]})
assert len(tasks)==2640
for cohort,n in [('VAL1280',1280),('TRAIN1024',1024),('DEV336_DIAGNOSTIC',336)]:
 tt=[x for x in tasks if x['part']==cohort];assert len(tt)==n;ss={};scenes=[];negative=[]
 for pl in ['drone','quad']:
  uu=[x for x in tt if x['platform']==pl];assert len(uu)==n//2;ss[pl]={}
  for k in ARMS:
   v=np.array([x['IoU'][k] for x in uu]);z={'n':len(uu),'mIoU_percent':float(v.mean()*100),'Acc25_percent':float((v>.25).mean()*100),'Acc50_percent':float((v>.5).mean()*100),'scene_clusters':len({x['scene_PRIVATE'] for x in uu}),'unknown_n':sum(x['statuses'][k]['unknown'] for x in uu),'fallback_n':sum(x['statuses'][k]['fallback'] for x in uu),'no_parameter_update_n':len(uu)}
   for ref in REFS:
    d=v-np.array([x['IoU'][ref] for x in uu]);z.update({f'gain_vs_{ref}_pp':float(d.mean()*100),f'improved_vs_{ref}':int((d>1e-10).sum()),f'worsened_vs_{ref}':int((d< -1e-10).sum()),f'query_changes_vs_{ref}':sum(x['queries'][k]!=x['queries'][ref] for x in uu)})
   ss[pl][k]=z
  for scene in sorted({x['scene_PRIVATE'] for x in uu}):
   sr=[x for x in uu if x['scene_PRIVATE']==scene];scenes.append({'scope':'ALL'+str(n),'platform':pl,'scene_PRIVATE':scene,'n':len(sr),'mIoU_percent':{k:float(np.mean([x['IoU'][k] for x in sr])*100) for k in ARMS}})
 ss['balanced_mIoU_percent']={k:float((ss['drone'][k]['mIoU_percent']+ss['quad'][k]['mIoU_percent'])/2) for k in ARMS};ss['micro_mIoU_percent']={k:float(np.mean([x['IoU'][k] for x in tt])*100) for k in ARMS}
 for x in tt:
  for k,ref in itertools.product(ARMS,repeat=2):
   d=x['IoU'][k]-x['IoU'][ref]
   if k!=ref and d< -1e-10:negative.append({'row':x['row'],'part':x['part'],'split':x['split'],'platform':x['platform'],'arm':k,'reference':ref,'delta_pp':float(d*100)})
 name={'VAL1280':'val1280','TRAIN1024':'train1024','DEV336_DIAGNOSTIC':'dev336'}[cohort];out=H/('score_'+name+'_attempt001');assert not out.exists();out.mkdir()
 for f,v in [('ALL_TASK_ROWS_PRIVATE',tt),('TASK_SUMMARY',{'ALL'+str(n):ss}),('ALL_SCENE_SUMMARIES_PRIVATE',scenes),('ALL_ORDERED_NEGATIVE_COMPARISONS',negative),('SHARED_CALIBRATION_SELECTION',None)]:dump(out/(f+'.json'),v)
 rec={'state':'PASS_R71_POSTSEAL_FIXED_CLIP_SAVED_CANDIDATE_IOU_JOIN','n':n,'cohort':cohort,'arm_names':ARMS,'task_values':n*5,'new_GT_scored_task_values':0,'reused_saved_task_values':n*5,'ordered_negative_comparisons':len(negative),'single_GT_scoring_pass':False,'single_saved_IoU_join':True,'new_raw_GT_scoring_passes':0,'no_new_original_GT_or_geometry_score':True,'all_unknown_fallback_and_no_update_rows_retained':True,'prediction_seal_sha256':sha(pred/'SEALED.json'),'independent_audit_sha256':sha(H/'independent_audit_attempt001/AUDIT_RESULT.json')};dump(out/'SCORING_RECEIPT.json',rec);seal(out,rec['state']);print(rec)
print({'n':len(tasks),'seconds':time.perf_counter()-start})
