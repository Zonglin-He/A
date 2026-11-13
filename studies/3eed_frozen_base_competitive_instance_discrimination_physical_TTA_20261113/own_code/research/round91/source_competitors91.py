"""Source annotation-supported distinct-instance negatives; no target access."""
from core91 import *
sys.path.insert(0,str(ROOT/'research/round5/audit'));from independent_core_v2 import candidate_ious,transform_gt,yaw_matrix
from obb_iou import oriented_iou
import time
O=B91/'source_competitors640_attempt001';(O/'samples_PRIVATE').mkdir(parents=True,exist_ok=False);start=time.perf_counter();allstats=[];max_target_error=0.
def gt_overlap(a,b):return oriented_iou(a[:3],a[3:6],yaw_matrix(a[6]),b[:3],b[3:6],yaw_matrix(b[6]))
for split in ['TRAIN512','VAL128']:
 for nr,r in enumerate(source_rows(split)):
  assert r['platform']=='waymo';z=prefix(r);boxes=z['boxes'].copy();boxes[:,3:]=np.maximum(boxes[:,3:],1e-6);meta=read(r['metapath']);target=meta['ground_info'][r['object_index']];cl=str(target['class']).strip().lower();gt=transform_gt(target['bbox_3d'],'waymo',r['suffix']);raw=[(x['bbox_3d'],str(x['class']).strip().lower()) for x in meta['ground_info']]
  for x in meta['ground_info']:raw.extend((o['bbox_3d_other'],str(o['class_other']).strip().lower()) for o in x.get('others',[]))
  objects=[];seen=set()
  for b,c in raw:
   if c!=cl:continue
   b=transform_gt(b,'waymo',r['suffix']);key=tuple(np.round(b.astype(float),5))
   if key in seen:continue
   seen.add(key)
   if gt_overlap(gt,b)<.05:objects.append(b)
  y=load(ROOT/'research/round25/experiment/source_labels_attempt002/samples_PRIVATE'/(r['id']+'.npz'))['ious'];fresh=candidate_ious(boxes,gt)[0];max_target_error=max(max_target_error,float(abs(y-fresh).max()));assert max_target_error<1e-12
  rt=load(B89/'source_features640_attempt001/routes_PRIVATE'/(r['id']+'.npz'));own=rt['own_support'].astype(bool);i=int(y.argmax());positive=bool(y[i]>.25 and own[i].sum()>=2);other=np.stack([candidate_ious(boxes,b)[0] for b in objects],1) if objects else np.zeros((256,0));assign=np.full(256,-1,np.int64)
  if positive and len(objects):
   best=other.argmax(1);top=other[np.arange(256),best];altern=other.copy();altern[np.arange(256),best]=-1;second=np.maximum(y,altern.max(1));lo=boxes[:,:3]-boxes[:,3:]/2;hi=boxes[:,:3]+boxes[:,3:]/2;iv=np.maximum(0,np.minimum(hi,hi[i])-np.maximum(lo,lo[i])).prod(1);vv=boxes[:,3:].prod(1);iou=iv/np.maximum(1e-12,vv+vv[i]-iv);inter=(own&own[i]).sum(1);union=(own|own[i]).sum(1);jac=inter/np.maximum(union,1);ok=(top>.25)&(top-second>=.1)&(y<.05)&(own.sum(1)>=2)&(jac<=.1)&(iou<=.1);assign[ok]=best[ok]
  np.savez_compressed(O/'samples_PRIVATE'/(r['id']+'.npz'),positive_query_PRIVATE=np.array(i),positive_valid=np.array(positive),competitor_assignment_PRIVATE=assign)
  allstats.append({'row':split+'_'+str(nr+1).zfill(4),'split':split,'same_class_disjoint_annotated_objects':len(objects),'positive_supported':positive,'eligible_negative_candidates':int((assign>=0).sum()),'eligible_distinct_negative_objects':len(set(assign[assign>=0].tolist())),'has_pair':bool((assign>=0).any()),'target_oracle_above25':bool(y[i]>.25)})
  if (nr+1)%128==0:print('COMPETITORS91',split,nr+1,flush=True)
dump(H91/'SOURCE_COMPETITOR_COVERAGE_ROWS.json',allstats);summary={s:{'n':sum(r['split']==s for r in allstats),'pair_eligible_n':sum(r['split']==s and r['has_pair'] for r in allstats),'positive_supported_n':sum(r['split']==s and r['positive_supported'] for r in allstats),'same_class_annotated_object_n':sum(r['split']==s and r['same_class_disjoint_annotated_objects']>0 for r in allstats)} for s in ['TRAIN512','VAL128']};dump(O/'COUNTS.json',{'source_only':True,'target_inputs_or_labels_read':False,'source_additional_non_target_box_annotations_used':True,'reference_role_identity_GT_used':False,'target_IoU_recalculation_max_error':max_target_error,'source_coverage':summary,'seconds':time.perf_counter()-start});seal(O,'R91_SOURCE_ANNOTATION_SUPPORTED_COMPETITOR_POOLS_BEFORE_TRAINING',n=640);print(read(O/'COUNTS.json'))
