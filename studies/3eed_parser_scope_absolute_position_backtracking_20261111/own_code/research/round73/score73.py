from common73 import *
import itertools,collections
assert read(H/'STATE.json')['CPU_scoring_authorized'];p=read(H/'experiment/RUN_PLAN_LOCK.json');pred=ROOT/p['prediction_root'];cert=read(H/'independent_audit_attempt001/AUDIT_RESULT.json');assert cert['state'].startswith('PASS_') and cert['prediction_seal_sha256']==sha(pred/'SEALED.json');ss=read(pred/'SEALED.json');labels={}
for root in ['research/round69/score_all2304_attempt001']:
 root=ROOT/root;seal0=read(root/'SEALED.json');f=root/'ALL_CANDIDATE_IOUS_PRIVATE.json';assert sha(f)==seal0['files'][f.name];labels.update({x['id_PRIVATE']:x['candidate_IoUs_PRIVATE'] for x in read(f)})
status={x['id']:x['arm_statuses'] for x in read(pred/'READOUT_STATUS_PRIVATE.json')};tasks=[];co=cohort_map()
for i,r in enumerate(read(H/'input/MECHANISM64_PRIVATE.json')):
 sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(pred/rel)==ss['files'][rel];z=load(pred/rel);q=dict(zip(ARMS,map(int,z['queries'])));iou=np.array(labels[sid]);tasks.append({'row':'R'+str(i+1).zfill(5),'id_PRIVATE':sid,'platform':r['platform'],'scene_PRIVATE':r['scene'],'part':co[sid],'split':co[sid],'queries':q,'IoU':{k:float(iou[q[k]]) for k in ARMS},'statuses':status[sid]})
summary={};scenes=[];neg=[]
for pl in ['drone','quad']:
 rr=[x for x in tasks if x['platform']==pl];summary[pl]={}
 for k in ARMS:
  values=np.array([x['IoU'][k] for x in rr]);summary[pl][k]={'n':len(rr),'mIoU_percent':float(values.mean()*100),'Acc25_percent':float((values>.25).mean()*100),'Acc50_percent':float((values>.5).mean()*100),'query_changes_vs_source':sum(x['queries'][k]!=x['queries']['source'] for x in rr),'improved_vs_source':sum(x['IoU'][k]>x['IoU']['source']+1e-10 for x in rr),'worsened_vs_source':sum(x['IoU'][k]<x['IoU']['source']-1e-10 for x in rr)}
 for scene in sorted({x['scene_PRIVATE'] for x in rr}):
  sub=[x for x in rr if x['scene_PRIVATE']==scene];scenes.append({'scope':'ALL64','platform':pl,'scene_PRIVATE':scene,'n':len(sub),'mIoU_percent':{k:float(np.mean([x['IoU'][k] for x in sub])*100) for k in ARMS}})
summary['balanced_mIoU_percent']={k:float(np.mean([summary[pl][k]['mIoU_percent'] for pl in ['drone','quad']])) for k in ARMS}
for x in tasks:
 for k,ref in itertools.permutations(ARMS,2):
  d=x['IoU'][k]-x['IoU'][ref]
  if d< -1e-10:neg.append({'row':x['row'],'platform':x['platform'],'part':x['part'],'split':x['split'],'arm':k,'reference':ref,'delta_pp':float(d*100)})
out=H/'score_mech64_attempt001';assert not out.exists();out.mkdir();receipt={'state':'PASS_R73_SEALED_MECHANISM64_SAVED_IOU_JOIN','n':64,'arm_names':ARMS,'task_values':64*len(ARMS),'new_GT_scored_task_values':0,'reused_saved_task_values':64*len(ARMS),'ordered_negative_comparisons':len(neg),'single_saved_IoU_join':True,'no_new_raw_GT_or_geometry_score':True,'prediction_seal_sha256':sha(pred/'SEALED.json')}
for f,v in [('ALL_TASK_ROWS_PRIVATE',tasks),('TASK_SUMMARY',{'ALL64':summary}),('ALL_SCENE_SUMMARIES_PRIVATE',scenes),('ALL_ORDERED_NEGATIVE_COMPARISONS',neg),('SCORING_RECEIPT',receipt),('SHARED_CALIBRATION_SELECTION',None)]:dump(out/(f+'.json'),v)
seal(out,receipt['state']);print(summary['balanced_mIoU_percent']);print(receipt)
