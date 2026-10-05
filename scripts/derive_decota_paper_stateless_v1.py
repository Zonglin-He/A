"""Derive Source/DINO-Refine from sealed exact shared inputs, CPU only."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_paper_common_v1 import *
def run():
 import numpy as np
 from vg_tta.decota_paper_controls_v1 import dino_refine
 verify();assert read(BASE/'GLOBAL_PREDICTION_BARRIER.json')['status']=='sealed'
 out=BASE/'table1_stateless';lf=out/'RUNTIME_LOCK.json'
 if not lf.exists():write(lf,dict(pins={f:sha(ROOT/f) for f in ['scripts/derive_decota_paper_stateless_v1.py','vg_tta/decota_paper_controls_v1.py']},main_runtime_sha256=sha(BASE/'RUNTIME_LOCK.json'),main_barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),GT_read=False,time=time.time()))
 lock=read(lf);assert lock['main_runtime_sha256']==sha(BASE/'RUNTIME_LOCK.json') and lock['main_barrier_sha256']==sha(BASE/'GLOBAL_PREDICTION_BARRIER.json')
 for f,h in lock['pins'].items():assert sha(ROOT/f)==h
 if (out/'GLOBAL_PREDICTION_BARRIER.json').exists():
  receipt=read(out/'GLOBAL_PREDICTION_BARRIER.json');assert receipt['status']=='sealed' and receipt['unique_readouts_per_arm']==13785
  for job,h in receipt['jobs'].items():assert sha(out/job/'PREDICTION_BARRIER.json')==h
  return
 total=0;barriers={}
 for job,ds,_ in JOBS:
  plan=read(BASE/ds/'PLAN.json');files={};done=0
  for row in plan['rows']:
   f=BASE/job/'inputs/clean'/f'{row["ordinal"]:05}.npz';a,md,rc=load_npz(f)
   assert md['GT_read'] is False and md['dataset']==ds and md['condition']=='clean' and md['parent']==row['ordinal']
   dest=out/job/f'{row["ordinal"]:05}.npz'
   if dest.exists():aa,z,c=load_npz(dest);assert z['input_sha256']==rc['sha256'] and z['control_lock_sha256']==sha(lf)
   else:
    refined,control=dino_refine(a['native_boxes'],md['frame_ids'],md['expert'])
    z=dict(parent=row['ordinal'],dataset=ds,input_sha256=rc['sha256'],control_lock_sha256=sha(lf),interval=md['interval'],frame_ids=md['frame_ids'],GT_read=False,
     control=control,shared_DINO_cache=True,new_DINO_calls=0,new_model_forwards=0,logical_orders=list(plan['orders']),native_stateful=False,
     shared_uncached_DINO_calls=md['expert']['new_DINO'],shared_expert_seconds=sum(x['receipt']['seconds'] for x in md['expert']['observations']))
    c=save_npz(dest,dict(Source=a['native_boxes'],DINO_Refine=refined),z)
   files[str(dest.relative_to(BASE))]=c['sha256'];done+=1
   if done%100==0:status(out/'STATUS.json',dict(status='deriving',job=job,done=done,total=plan['queries'],GT_read=False,time=time.time()));print('STATELESS_PROGRESS',job,done,plan['queries'],flush=True)
  barrier=out/job/'PREDICTION_BARRIER.json'
  if barrier.exists():
   sealed=read(barrier);assert sealed['status']=='sealed' and sealed['files']==files and sealed['unique_readouts']==done
  else:write(barrier,dict(status='sealed',arms=['Source Only','DINO-Refine'],unique_readouts=done,logical_rows_per_arm=done*3,files=files,GT_read=False,time=time.time()))
  barriers[job]=sha(barrier);total+=done
 assert total==13785
 write(out/'GLOBAL_PREDICTION_BARRIER.json',dict(status='sealed',jobs=barriers,unique_readouts_per_arm=13785,logical_rows_per_arm=41355,new_DINO_calls=0,new_model_forwards=0,GT_read=False,time=time.time()))
 status(out/'STATUS.json',dict(status='sealed_pending_remaining_Table1_arms',GT_read=False,time=time.time()))
 print('TABLE1_STATELESS_SEALED',13785,41355,flush=True)
if __name__=='__main__':run()
