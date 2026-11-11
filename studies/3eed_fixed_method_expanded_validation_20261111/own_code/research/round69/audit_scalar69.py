"""Independent yaw-box polygon geometry, caption join and every task scalar."""
import math,ijson,itertools
from common69 import *
H=ROOT/'research/round69';p=read(H/'experiment/SCALAR_PLAN_LOCK.json');pin(p);tasks=read(H/'score_all2304_attempt001/ALL_TASK_ROWS_PRIVATE.json');labels=read(H/'score_all2304_attempt001/ALL_CANDIDATE_IOUS_PRIVATE.json');rows=read(ROOT/p['manifest']);idx={x['id']:x for x in read(ROOT/p['scoring_index'])};error=0.
def area(poly):
 if len(poly)<3:return 0.
 z=np.asarray(poly,float);return abs(float(np.dot(z[:,0],np.roll(z[:,1],-1))-np.dot(z[:,1],np.roll(z[:,0],-1))))/2
def cross(a,b):return a[0]*b[1]-a[1]*b[0]
def clip(subject,boundary):
 pp=[np.array(a,float) for a in subject]
 for a,b in zip(boundary,np.roll(boundary,-1,axis=0)):
  out=[]
  if not pp:break
  prev=pp[-1];vp=cross(b-a,prev-a)
  for cur in pp:
   vc=cross(b-a,cur-a)
   if (vc>=0)!=(vp>=0):
    denom=cross(cur-prev,b-a)
    if denom!=0:out.append(prev+cross(a-prev,b-a)/denom*(cur-prev))
   if vc>=0:out.append(cur)
   prev=cur;vp=vc
  pp=out
 return area(pp)
def all_ious(boxes,raw,platform):
 target=np.asarray(raw,np.float64).copy()
 if platform=='drone':target[2]+=1.8
 t=target[:7].astype(np.float32).astype(float);yaw=t[6];c=math.cos(yaw);s=math.sin(yaw);corners=np.array([[-1,-1],[1,-1],[1,1],[-1,1]],float)*t[3:5]/2;bound=corners@np.array([[c,s],[-s,c]])+t[:2];result=[]
 for b in boxes.astype(float):
  vertical=max(0.,min(b[2]+b[5]/2,t[2]+t[5]/2)-max(b[2]-b[5]/2,t[2]-t[5]/2))
  if vertical==0:result.append(0.);continue
  q=np.array([[-1,-1],[1,-1],[1,1],[-1,1]],float)*b[3:5]/2+b[:2]
  if (q.max(0)<bound.min(0)).any() or(bound.max(0)<q.min(0)).any():result.append(0.);continue
  inter=clip(q,bound)*vertical;union=np.prod(b[3:6])+np.prod(t[3:6])-inter;result.append(inter/union if union>0 else 0.)
 return np.array(result)
for i,(r,t,l) in enumerate(zip(rows,tasks,labels)):
 assert r['id']==t['id_PRIVATE']==l['id_PRIVATE'];ix=idx[r['id']];assert sha(ix['meta_path'])==ix['metadata_container_sha256']
 with Path(ix['meta_path']).open('rb') as f:obj=next(o for j,o in enumerate(ijson.items(f,'ground_info.item')) if j==ix['object_index'])
 assert ' '.join(obj['caption'].replace(',',' ,').split())==r['caption'];z=load(ROOT/p['prediction_root']/'samples_PRIVATE'/(r['id']+'.npz'));boxes=z['source_boxes'].copy();boxes[:,3:]=np.maximum(boxes[:,3:],1e-6);ious=all_ious(boxes,obj['bbox_3d'],r['platform']);err=float(abs(ious-np.array(l['candidate_IoUs_PRIVATE'])).max());error=max(error,err);assert err<1e-7,('geometry',i,err)
 for threshold in [.25,.5]:assert np.array_equal(ious>threshold,np.asarray(l['candidate_IoUs_PRIVATE'])>threshold),('threshold_geometry',i,threshold)
 for a in ARMS:assert t['IoU'][a]==l['candidate_IoUs_PRIVATE'][t['queries'][a]]
 if(i+1)%128==0:print('AUDIT_GEOMETRY69',i+1,len(rows),error,flush=True)
# Cohort tables are derived from this one sealed scoring pass, not rescored.
score=H/'score_all2304_attempt001';summary=read(score/'TASK_SUMMARY.json');scene=read(score/'ALL_SCENE_SUMMARIES_PRIVATE.json');allneg=read(score/'ALL_ORDERED_NEGATIVE_COMPARISONS.json')
for co in ['VAL1280','TRAIN1024']:
 tt=[t for t in tasks if t['part']==co]
 for pl in ['drone','quad']:
  rr=[t for t in tt if t['platform']==pl]
  for a in ARMS:
   v=np.array([t['IoU'][a] for t in rr]);actual=summary[co][pl][a];assert abs(v.mean()*100-actual['mIoU_percent'])<1e-11;assert abs((v>.25).mean()*100-actual['Acc25_percent'])<1e-11;assert abs((v>.5).mean()*100-actual['Acc50_percent'])<1e-11
 for sc in [s for s in scene if s['scope']==co]:
  rr=[t for t in tt if t['platform']==sc['platform'] and t['scene_PRIVATE']==sc['scene_PRIVATE']]
  for a in ARMS:assert abs(np.mean([t['IoU'][a] for t in rr])*100-sc['mIoU_percent'][a])<1e-11
 neg=[]
 for t in tt:
  for a,ref in itertools.product(ARMS,repeat=2):
   delta=t['IoU'][a]-t['IoU'][ref]
   if a!=ref and delta< -1e-10:neg.append({'row':t['row'],'part':t['part'],'split':t['split'],'platform':t['platform'],'arm':a,'reference':ref,'delta_pp':delta*100})
 assert neg==[r for r in allneg if r['part']==co];out=H/('score_'+co.lower()+'_derived001');assert not out.exists();out.mkdir();ls={l['id_PRIVATE']:l for l in labels}
 for name,value in [('ALL_TASK_ROWS_PRIVATE',tt),('ALL_CANDIDATE_IOUS_PRIVATE',[ls[t['id_PRIVATE']] for t in tt]),('TASK_SUMMARY',{co:summary[co]}),('ALL_SCENE_SUMMARIES_PRIVATE',[r for r in scene if r['scope']==co]),('ALL_ORDERED_NEGATIVE_COMPARISONS',neg),('SHARED_CALIBRATION_SELECTION',None)]:dump(out/(name+'.json'),value)
 receipt={'state':'PASS_R69_DERIVED_COHORT_FROM_SINGLE_ALL2304_GT_PASS','n':len(tt),'arm_names':ARMS,'task_values':len(tt)*len(ARMS),'new_GT_scored_task_values':len(tt)*len(ARMS),'reused_saved_task_values':0,'ordered_negative_comparisons':len(neg),'single_GT_scoring_pass':True,'derived_without_extra_GT_scoring':True,'parent_score_seal_sha256':sha(score/'SEALED.json')};dump(out/'SCORING_RECEIPT.json',receipt);seal(out,receipt['state'])
out=H/'independent_scalar_audit_attempt001';assert not out.exists();out.mkdir();dump(out/'AUDIT_RESULT.json',{'state':'PASS_R69_INDEPENDENT_POLYGON_GEOMETRY_AND_ALL_TASK_SCENE_NEGATIVE_SCALARS','n':len(rows),'arms':len(ARMS),'max_geometry_IoU_error':error,'candidate_IoUs_independently_checked':len(rows)*256,'no_method_or_hyperparameter_changed':True,'scoring_seal_sha256':sha(score/'SEALED.json')});print({'n':len(rows),'max_geometry_error':error})
