"""Target-trained frozen reference on identical complete Table1 input rosters.

This is a separately labeled supervised reference, never a TTA competitor or
a mathematical upper bound. No DINO, gradient update or evaluation label read.
"""
import os,sys,time,collections,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_paper_common_v1 import *
OUT=BASE/'table1_target_reference'

def lock():
 verify();lf=OUT/'RUNTIME_LOCK.json'
 if not lf.exists():
  pins=['scripts/run_decota_paper_reference_v1.py','vg_tta/native_baselines_paper_v1.py']
  write(lf,dict(pins={f:sha(ROOT/f) for f in pins},main_runtime_sha256=sha(BASE/'RUNTIME_LOCK.json'),GT_read=False,time=time.time()))
 cfg=read(lf);assert cfg['main_runtime_sha256']==sha(BASE/'RUNTIME_LOCK.json')
 for p,h in cfg['pins'].items():assert sha(ROOT/p)==h
 return cfg

def run(ds):
 import torch
 from scripts.run_decota_paper_main_v1 import gpu,model_for,read_row,frames_for
 from methods.decota_final_simplified_v1.backbone import make_batch
 from methods.decota_final_simplified_v1.tensors import state_hash
 from vg_tta.native_baselines_paper_v1 import live_output
 assert ds in DATASETS
 lock();dest=OUT/ds;barrier=dest/'PREDICTION_BARRIER.json'
 if barrier.exists():
  rc=read(barrier);assert rc['status']=='sealed' and rc['unique_predictions']==read(BASE/ds/'PLAN.json')['queries']
  for p,h in rc['files'].items():assert sha(BASE/p)==h
  return
 lease=gpu();source='hcstvg2' if ds=='hc2' else 'vidstg';model=model_for(source);mh=state_hash(model.state_dict())
 cache=collections.OrderedDict();plan=read(BASE/ds/'PLAN.json');files={};tick=time.time();checks=[]
 try:
  for row in plan['rows']:
   budget();parent=row['ordinal'];row=read_row(ds,parent);f=dest/f'{parent:05}.npz'
   if f.exists():
    a,md,rc=load_npz(f);assert md['reference_lock_sha256']==sha(OUT/'RUNTIME_LOCK.json') and md['input_metadata_sha256']==digest(row['input'])
   else:
    frames,ids,reuse=frames_for(ds,row,cache);batch=make_batch(frames,ids,row['input'],model)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
    with torch.no_grad():native,pred=live_output(model,batch,ids,row['parses']['subject'])
    torch.cuda.synchronize();seconds=time.perf_counter()-begin
    assert torch.isfinite(pred['boxes']).all() and len(pred['boxes'])==len(ids) and 0<=pred['indices'][0]<pred['indices'][1]<len(ids)
    if parent<2:
     with torch.no_grad():_,again=live_output(model,batch,ids,row['parses']['subject'])
     assert torch.equal(again['boxes'],pred['boxes']) and again['indices']==pred['indices']
     checks.append(dict(parent=parent,repeat_bitwise=True,finite=True,GT_read=False))
    md=dict(method='Target-trained reference',dataset=ds,source_checkpoint=source,parent=parent,frame_ids=ids,interval=pred['physical_interval'],indices=pred['indices'],
     input_metadata_sha256=digest(row['input']),source_model_state_sha256=mh,reference_lock_sha256=sha(OUT/'RUNTIME_LOCK.json'),logical_orders=list(plan['orders']),
     supervised_source_training=True,not_a_mathematical_upper_bound=True,GT_read=False,
     compute=dict(model_seconds=seconds,new_model_forwards=2,new_DINO_calls=0,backwards=0,CUDA_peak_allocated=torch.cuda.max_memory_allocated(),CUDA_peak_reserved=torch.cuda.max_memory_reserved(),decoded_frame_reuse=reuse))
    rc=save_npz(f,dict(Reference=pred['boxes'].cpu().numpy()),md);del frames,batch,native,pred
   files[str(f.relative_to(BASE))]=rc['sha256']
   if (parent+1)%10==0:
    status(dest/'STATUS.json',dict(status='running',pid=os.getpid(),done=parent+1,total=plan['queries'],GT_read=False,seconds=time.time()-tick,time=time.time()))
    print('TARGET_REFERENCE_PROGRESS',ds,parent+1,plan['queries'],flush=True)
  assert state_hash(model.state_dict())==mh and len(files)==plan['queries']
  write(barrier,dict(status='sealed',method='Target-trained reference',dataset=ds,unique_predictions=plan['queries'],logical_rows=plan['queries']*3,files=files,GT_read=False,time=time.time()))
  write(dest/'SMOKE_AND_RESTORATION.json',dict(status='pass',first_two_queries=checks,full_model_unchanged=True,GT_read=False,time=time.time()))
  status(dest/'STATUS.json',dict(status='sealed_pending_all_Table1',done=plan['queries'],GT_read=False,time=time.time()))
 finally:lease.close()

if __name__=='__main__':
 try:run(sys.argv[1])
 except BaseException:
  fd=OUT/'failures'/str(time.time_ns());fd.mkdir(parents=True,exist_ok=True);(fd/'traceback.txt').write_text(traceback.format_exc())
  status(OUT/'WORKER_FAILURE.json',dict(status='failed',dataset=sys.argv[1],pid=os.getpid(),failure=str(fd),GT_read=False,time=time.time()));raise
