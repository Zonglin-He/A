"""One postseal GT evaluation for fixed methods; no access from adaptation."""
import sys,time,itertools,ijson
from common69 import *
sys.path.insert(0,str(ROOT/'research/round5/audit'));from independent_core_v2 import transform_gt,candidate_ious
H=ROOT/'research/round69';assert read(H/'STATE.json')['CPU_scoring_authorized'];p=read(H/'experiment/SCORE_PLAN_LOCK.json');pin(p);pred=ROOT/p['prediction_root'];ps=read(pred/'SEALED.json');cert=read(H/'independent_audit_attempt001/AUDIT_RESULT.json');assert cert['state'].startswith('PASS_') and cert['prediction_seal_sha256']==sha(pred/'SEALED.json');rows=read(ROOT/p['manifest']);idx={x['id']:x for x in read(ROOT/p['scoring_index'])};cohort=read(ROOT/p['cohort_map']);status={x['id']:x['arm_statuses'] for x in read(pred/'READOUT_STATUS_PRIVATE.json')};out=H/'score_all2304_attempt001';assert not out.exists();out.mkdir();tasks=[];allious=[];start=time.perf_counter()
for i,r in enumerate(rows):
 sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(pred/rel)==ps['files'][rel];z=load(pred/rel);ix=idx[sid];assert sha(ix['meta_path'])==ix['metadata_container_sha256']
 with Path(ix['meta_path']).open('rb') as f:obj=next(o for j,o in enumerate(ijson.items(f,'ground_info.item',use_float=True)) if j==ix['object_index'])
 assert ' '.join(obj['caption'].replace(',',' ,').split())==r['caption'];gt=transform_gt(np.asarray(obj['bbox_3d'],dtype=np.float64),r['platform']);boxes=z['source_boxes'].copy();boxes[:,3:]=np.maximum(boxes[:,3:],1e-6);ious=candidate_ious(boxes,gt)[0];q=dict(zip(ARMS,map(int,z['queries'])));values={a:float(ious[q[a]]) for a in ARMS};tasks.append({'row':f'R{i+1:05d}','id_PRIVATE':sid,'platform':r['platform'],'scene_PRIVATE':r['scene'],'part':cohort[sid],'split':cohort[sid],'queries':q,'IoU':values,'statuses':status[sid]});allious.append({'id_PRIVATE':sid,'candidate_IoUs_PRIVATE':ious.tolist()})
 if(i+1)%64==0:print('SCORE69',i+1,len(rows),flush=True)
# Separate cohort summaries; never combine into a favorable-method selector.
summary={};scenes=[];neg=[]
for co in ['VAL1280','TRAIN1024']:
 tt=[t for t in tasks if t['part']==co];ss={};n=len(tt)
 for pl in ['drone','quad']:
  rr=[t for t in tt if t['platform']==pl];ss[pl]={}
  for a in ARMS:
   v=np.array([t['IoU'][a] for t in rr]);ss[pl][a]={'n':len(rr),'mIoU_percent':float(v.mean()*100),'Acc25_percent':float((v>.25).mean()*100),'Acc50_percent':float((v>.5).mean()*100),'scene_clusters':len({t['scene_PRIVATE'] for t in rr})}
   for ref in REFS:
    dd=v-np.array([t['IoU'][ref] for t in rr]);ss[pl][a].update({f'gain_vs_{ref}_pp':float(dd.mean()*100),f'improved_vs_{ref}':int((dd>1e-10).sum()),f'worsened_vs_{ref}':int((dd< -1e-10).sum())})
  for sc in sorted({t['scene_PRIVATE'] for t in rr}):
   aa=[t for t in rr if t['scene_PRIVATE']==sc];scenes.append({'scope':co,'platform':pl,'scene_PRIVATE':sc,'n':len(aa),'mIoU_percent':{a:float(np.mean([t['IoU'][a] for t in aa])*100) for a in ARMS}})
 ss['balanced_mIoU_percent']={a:(ss['drone'][a]['mIoU_percent']+ss['quad'][a]['mIoU_percent'])/2 for a in ARMS};summary[co]=ss
for t in tasks:
 for a,ref in itertools.product(ARMS,repeat=2):
  delta=t['IoU'][a]-t['IoU'][ref]
  if a!=ref and delta< -1e-10:neg.append({'row':t['row'],'part':t['part'],'split':t['split'],'platform':t['platform'],'arm':a,'reference':ref,'delta_pp':delta*100})
for name,value in [('ALL_TASK_ROWS_PRIVATE',tasks),('ALL_CANDIDATE_IOUS_PRIVATE',allious),('TASK_SUMMARY',summary),('ALL_SCENE_SUMMARIES_PRIVATE',scenes),('ALL_ORDERED_NEGATIVE_COMPARISONS',neg),('SHARED_CALIBRATION_SELECTION',None)]:dump(out/(name+'.json'),value)
receipt={'state':'PASS_R69_SINGLE_POSTSEAL_FIXED_METHOD_TARGET_GT_SCORE','n':len(rows),'arm_names':ARMS,'task_values':len(rows)*len(ARMS),'new_GT_scored_task_values':len(rows)*len(ARMS),'reused_saved_task_values':0,'ordered_negative_comparisons':len(neg),'single_GT_scoring_pass':True,'all_candidate_IoUs_cached_for_postseal_diagnostics':True,'no_label_used_to_change_methods_or_parameters':True,'prediction_seal_sha256':sha(pred/'SEALED.json'),'seconds':time.perf_counter()-start};dump(out/'SCORING_RECEIPT.json',receipt);seal(out,receipt['state']);print(receipt)
