"""Seal 208 cached-domain predictions; never imports a task scorer or GT."""
from common47 import *
def main():
 check_plan()
 for folder in [STRICT,P3,PRE]:verify(folder)
 OUT.mkdir(exist_ok=False)
 for n in ['cases_PRIVATE','predictions_PRIVATE']:(OUT/n).mkdir()
 for r in rows():
  case=prepared(r);z=arrays(r);old=load_npz(STRICT/'predictions_PRIVATE'/(r['id']+'.npz'));assert old['arm_names'].tolist()==STRICT_ARMS
  calls={'3B':read(P3/'calls_PRIVATE'/(r['id']+'_bbox.json'))['result'],'7B':read(STRICT/'calls_PRIVATE'/(r['id']+'_bbox.json'))}
  reads={};details={}
  for cap,c in calls.items():
   rd,de=readouts_for(cap,c,case,z);reads.update(rd);details[cap]=de
  q=np.r_[old['queries'],[reads[a]['query'] for a in NEW_ARMS]].astype(np.int64);boxes=z['source_boxes'][q]
  assert np.array_equal(boxes[:21],old['boxes']) and np.array_equal(q[:21],old['queries'])
  atomic_npz(OUT/'predictions_PRIVATE'/(r['id']+'.npz'),arm_names=np.array(ARMS),queries=q,boxes=boxes)
  atomic(OUT/'cases_PRIVATE'/(r['id']+'.json'),{'id_PRIVATE':r['id'],'alias':r['alias'],'platform':r['platform'],'scene_PRIVATE':r['scene'],'details':details,'readouts':reads,'raw_refs':{c:{'file':str((P3/'calls_PRIVATE'/(r['id']+'_bbox.json') if c=='3B' else STRICT/'calls_PRIVATE'/(r['id']+'_bbox.json')).relative_to(ROOT)),'sha256':sha(P3/'calls_PRIVATE'/(r['id']+'_bbox.json') if c=='3B' else STRICT/'calls_PRIVATE'/(r['id']+'_bbox.json'))} for c in calls}})
 files={str(f.relative_to(OUT)):sha(f) for f in sorted(OUT.rglob('*')) if f.is_file() and f.name!='SEALED.json'}
 atomic(OUT/'SEALED.json',{'state':'R47_CACHED_GEOMETRY_DOMAIN_PREDICTIONS_BEFORE_TASK_SCORING','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'n':208,'arms':29,'new_model_GPU_VLM_optimizer_calls':0,'files':files,'plan_sha256':sha(EXP/'PLAN_LOCK.json'),'implementation_sha256':sha(EXP/'IMPLEMENTATION_LOCK.json'),'root_authorization_sha256':sha(EXP/'ROOT_EXECUTION_AUTHORIZATION.json'),'strict21_unchanged':True})
 print('R47_ALL208_29ARMS_SEALED',sha(OUT/'SEALED.json'))
if __name__=='__main__':main()
