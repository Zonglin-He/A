"""Exact saved TENT continuation after the user-requested OPD intermission.

Original code, predictions, scientific config and runtime locks stay immutable.
Restore the latest complete optimizer sample and replay only the saved tail,
requiring bitwise parameter and Before/After equality before adding arrivals.
"""
import sys,os,time,collections,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_paper_common_v1 import *
from scripts.run_decota_paper_baselines_v1 import lock,native_input,flat,OUT
PAUSE=BASE/'user_opd_pause_20261006'

def run(checkpoint_only=False):
 import torch,numpy as np
 from scripts.run_decota_paper_main_v1 import gpu,model_for,read_row
 from vg_tta.tastvg_baseline_expansion import decoder_layernorm_scope
 from vg_tta.native_baselines_paper_v1 import live_output
 from vg_tta.decota_paper_baselines_20261005_v1 import OnlineBaseline
 from scripts.decota_matrix_common_v1 import load,save
 from methods.decota_final_simplified_v1.tensors import state_hash
 lock();verify();manifest=read(PAUSE/'PREFIX_MANIFEST.json')
 for k,h in manifest['files'].items():
  if not k.endswith('STATUS.json'):assert sha(BASE/k)==h,k
 dest=OUT/'TENT_vidstg';plan=read(BASE/'vidstg/PLAN.json');names_order=list(plan['orders'])
 complete=[];files={};cursor=None
 for order,seq in plan['orders'].items():
  n=0;previous=None
  for at,parent in enumerate(seq):
   f=dest/order/f'{at:05}.npz'
   if not f.with_suffix('.json').exists():
    assert not f.exists(),'Unreceipted original payload: preserve and inspect'
    assert not any((dest/order).glob(f'{at+1:05}.npz'));break
   a,md,rc=load_npz(f)
   assert md['parent']==parent and md['arrival']==at and md['order']==order and md['previous_payload_sha256']==previous
   assert not md['GT_read'] and md['method']=='TENT'
   previous=rc['sha256'];files[str(f.relative_to(BASE))]=previous;n+=1
  if n==len(seq):complete.append(order)
  elif cursor is None:cursor=(order,n,previous)
  else:assert n==0,'Unexpected saved outputs after incomplete order'
 assert len(files)==sum(k.endswith('.npz') for k in manifest['files'])
 assert cursor is not None
 lease=gpu();model=model_for('hcstvg2');mh=state_hash(model.state_dict());names,params,scope=decoder_layernorm_scope(model)
 opt=OnlineBaseline(names,params,'TENT');cache=collections.OrderedDict();tick=time.time();order,start,previous=cursor
 checkpoint=PAUSE/'EXACT_RESUME_STATE.pt';receipt=PAUSE/'EXACT_RESUME_RECEIPT.json'
 try:
  if receipt.exists():
   r=read(receipt);assert r['status']=='pass' and r['checkpoint_sha256']==sha(checkpoint)
   z=load(checkpoint);assert z['order']==order and z['next_arrival']==start and z['previous_payload_sha256']==previous
   opt.load_state_dict(z['state']);replayed=r['replayed_saved_arrivals']
  else:
   candidates=sorted((dest/'state_samples'/order).glob('*.pt'))
   candidates=[f for f in candidates if int(f.stem)<start]
   replay_start=0;sample=None
   if candidates:
    sample=candidates[-1];sample_at=int(sample.stem)
    a,md,rc=load_npz(dest/order/f'{sample_at:05}.npz')
    assert md['optimizer_sample']['sha256']==sha(sample)
    opt.load_state_dict(load(sample));assert np.array_equal(flat(params),a['state_after'])
    assert opt.arrivals==sample_at+1;replay_start=sample_at+1
   for at in range(replay_start,start):
    parent=plan['orders'][order][at];a,md,rc=load_npz(dest/order/f'{at:05}.npz')
    row=read_row('vidstg',parent);batch,ids,subject,_=native_input(model,'vidstg',row,cache)
    assert np.array_equal(flat(params),a['state_before'])
    pre,post,ev=opt.arrive(lambda:live_output(model,batch,ids,subject)[0],lambda:live_output(model,batch,ids,subject)[1])
    assert np.array_equal(flat(params),a['state_after']),('state replay',at)
    assert np.array_equal(pre['boxes'].cpu().numpy(),a['Before']),('Before replay',at)
    assert np.array_equal(post['boxes'].cpu().numpy(),a['After']),('After replay',at)
    assert ev['arrival']==at+1 and pre['physical_interval']==md['interval_before'] and post['physical_interval']==md['interval_after']
    del batch,pre,post
    if (at-replay_start)%10==0:print('SAVED_TAIL_REPLAY',at,start,flush=True)
   replayed=start-replay_start
   save(checkpoint,dict(state=opt.state_dict(),order=order,next_arrival=start,previous_payload_sha256=previous))
   write(receipt,dict(status='pass',saved_predictions=len(files),complete_orders=complete,order=order,next_arrival=start,
    checkpoint_sha256=sha(checkpoint),source_optimizer_sample=None if sample is None else str(sample.relative_to(BASE)),
    replayed_saved_arrivals=replayed,bitwise_parameters_before_after=True,bitwise_predictions_before_after=True,
    original_files_unchanged=True,GT_read=False,time=time.time()))
  if checkpoint_only:return
  count=len(files);started=False
  for current,seq in plan['orders'].items():
   if current in complete:continue
   first=start if current==order else 0
   if current!=order:opt.reset();previous=None
   for at in range(first,len(seq)):
    budget();parent=seq[at];f=dest/current/f'{at:05}.npz';assert not f.exists()
    row=read_row('vidstg',parent);batch,ids,subject,reuse=native_input(model,'vidstg',row,cache)
    before=flat(params);prehash=digest(before.tolist());torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
    pre,post,audit=opt.arrive(lambda:live_output(model,batch,ids,subject)[0],lambda:live_output(model,batch,ids,subject)[1]);torch.cuda.synchronize();seconds=time.perf_counter()-begin
    after=flat(params);md=dict(method='TENT',dataset='vidstg',source_checkpoint='hcstvg2',parent=parent,order=current,arrival=at,frame_ids=ids,
     interval_before=pre['physical_interval'],interval_after=post['physical_interval'],indices_before=pre['indices'],indices_after=post['indices'],
     previous_payload_sha256=previous,parameter_before_sha256=prehash,parameter_after_sha256=digest(after.tolist()),input_metadata_sha256=digest(row['input']),
     port_lock_sha256=sha(OUT/'RUNTIME_LOCK.json'),audit={k:v for k,v in audit.items() if k!='update_evidence'},
     compute=dict(model_seconds=seconds,CUDA_peak_allocated=torch.cuda.max_memory_allocated(),CUDA_peak_reserved=torch.cuda.max_memory_reserved(),new_DINO_calls=0,new_backbone_forwards=2*(2+audit['counts']['forward_closures']),decoded_frame_reuse=reuse),
     engineering_resume_receipt_sha256=sha(receipt),GT_read=False)
    if at%100==0:
     sample=dest/'state_samples'/current/f'{at:05}.pt';save(sample,opt.state_dict());md['optimizer_sample']=dict(path=str(sample.relative_to(BASE)),sha256=sha(sample))
    rc=save_npz(f,dict(Before=pre['boxes'].cpu().numpy(),After=post['boxes'].cpu().numpy(),state_before=before,state_after=after),md)
    previous=rc['sha256'];files[str(f.relative_to(BASE))]=previous;count+=1;del batch,pre,post,before,after
    if count%10==0:
     status(dest/'STATUS.json',dict(status='online_running',method='TENT',pid=os.getpid(),done=count,total=plan['queries']*3,order=current,GT_read=False,time=time.time()));print('BASELINE_RESUMED_PROGRESS',count,plan['queries']*3,flush=True)
  opt.reset();assert state_hash(model.state_dict())==mh and count==plan['queries']*3
  write(dest/'PREDICTION_BARRIER.json',dict(status='sealed',method='TENT',dataset='vidstg',cells=count,files=files,GT_read=False,time=time.time()))
  status(dest/'STATUS.json',dict(status='completed_sealed_pending_all_Table1',done=count,total=count,GT_read=False,time=time.time()))
 finally:lease.close()

if __name__=='__main__':run('--checkpoint-only' in sys.argv)
