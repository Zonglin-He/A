"""Actual fixed256 propagation, full raw masks checkpointed before validation."""
from pilot_common_v1 import *
import traceback,gc
def run():
 import torch,numpy as np,importlib.util
 from stvg.runtime import setup,lease
 from pilot_gate_v1 import gate
 from model_adapter_v1 import build_cpu,strict_fill
 spec=importlib.util.spec_from_file_location('_unchanged_EdgeTAM_output_adapter',S/'adapter.py');adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
 plan=validate();assert gate()is not None;assert not(D/'pilot_v1/worker.json').exists();setup();torch.set_num_threads(1);parent.gt_guard();lock=lease();who=identity();atomic(D/'pilot_v1/worker.json',who);start=time.perf_counter();model=None;counts=dict(attempted_queries=0,committed_queries=0,new_Source=0,new_original_expert=0,new_optimizer=0)
 try:
  resources(start);tick=time.perf_counter();model,load=build_cpu();model=model.cuda().eval().requires_grad_(False);torch.cuda.synchronize();model_load=time.perf_counter()-tick;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  import sam2.sam2_video_predictor as module
  import sam2.utils.misc as misc
  import sam2._C as extension
  # Explicit real CUDA postprocessing contract, separate from video queries.
  tick=time.perf_counter();mask=torch.zeros(1,1,4,4,dtype=torch.uint8,device='cuda');mask[0,0,0,0:2]=1;mask[0,0,3,3]=1;labels,areas=extension.get_connected_componnets(mask);torch.cuda.synchronize();assert areas[0,0,0,0].item()==areas[0,0,0,1].item()==2 and areas[0,0,3,3].item()==1;kernel_contract=time.perf_counter()-tick
  original_fill=module.fill_holes_in_mask_scores;kernel_calls=[0];strict=strict_fill(misc)
  def checked_fill(*args,**kwargs):kernel_calls[0]+=1;return strict(*args,**kwargs)
  module.fill_holes_in_mask_scores=checked_fill
  encoder_calls=[0]
  def observe_encoder(module,args,out):encoder_calls[0]+=1
  encoder_handle=model.image_encoder.register_forward_hook(observe_encoder)
  atomic(D/'pilot_v1/model_load.json',dict(worker=who,seconds=model_load,load=load,device=torch.cuda.get_device_name(),precision='FP32_no_autocast_no_TF32',actual_CUDA_kernel_contract_passed=True,kernel_contract_seconds=kernel_contract,kernel_contract_not_video_forward=True))
  for ds in ['hc2','vidstg']:
   folder=D/'pilot_v1'/ds;files={};domain_start=time.perf_counter()
   for index,row in enumerate(rows(ds),1):
    resources(start);key=row['id'];attempt=folder/'attempts'/(key+'.json');assert not attempt.exists();tick=time.perf_counter();decode_start=time.perf_counter();frames,pos,native,arrays,proof=payload(ds,row);decode=time.perf_counter()-decode_start;N,H,W,_=frames.shape;scratch=folder/'active'/key;rawfiles=[];packed={};raw_boxes={};checkpoint_IO=[0.];stages=dict(init_preprocess_and_frame0_seconds=0.,eight_prompt_request_seconds=0.,propagation_generator_seconds=0.);torch.cuda.reset_peak_memory_stats();resident=torch.cuda.memory_allocated();kc=kernel_calls[0];ec=encoder_calls[0]
    atomic(attempt,dict(status='about_to_run_one_EdgeTAM_query',worker=who,input_binding=proof,N=N,height=H,width=W,GT_read=False));counts['attempted_queries']+=1
    def checkpoint(i,scores):
     torch.cuda.synchronize();t=time.perf_counter();p=scratch/(str(int(i))+'.pt');save(p,dict(scores=scores.detach().cpu(),frame_index=int(i),input_binding=proof,GT_read=False));e=dict(path=str(p.relative_to(D)),sha256=sha(p));atomic(p.with_suffix('.json'),dict(e,before_finite_shape_coverage_box_processing=True));rawfiles.append(e);checkpoint_IO[0]+=time.perf_counter()-t
     assert scores.shape==(1,1,H,W)and torch.isfinite(scores).all();binary=(scores[0,0]>0).detach().cpu().numpy();assert int(i)not in packed;packed[int(i)]=np.packbits(binary.reshape(-1));raw_boxes[int(i)]=adapter.mask_box(binary)
    originals={name:getattr(model,name)for name in ['init_state','add_new_points_or_box','propagate_in_video']}
    def init(*a,**kw):
     t=time.perf_counter();out=originals['init_state'](*a,**kw);torch.cuda.synchronize();stages['init_preprocess_and_frame0_seconds']+=time.perf_counter()-t;return out
    def prompt(*a,**kw):
     t=time.perf_counter();out=originals['add_new_points_or_box'](*a,**kw);torch.cuda.synchronize();stages['eight_prompt_request_seconds']+=time.perf_counter()-t;return out
    def propagate(*a,**kw):
     it=iter(originals['propagate_in_video'](*a,**kw))
     while True:
      t=time.perf_counter()
      try:out=next(it)
      except StopIteration:break
      torch.cuda.synchronize();stages['propagation_generator_seconds']+=time.perf_counter()-t;yield out
    model.init_state=init;model.add_new_points_or_box=prompt;model.propagate_in_video=propagate;request=time.perf_counter()
    try:boxes,info=adapter.propagate_boxes(model,module,frames,pos,native,checkpoint)
    finally:
     for name in originals:delattr(model,name)
    request=time.perf_counter()-request;assert set(packed)==set(range(N))and len(rawfiles)==N and kernel_calls[0]>kc;support=dict(packed_masks=np.stack([packed[i]for i in range(N)]),native_shape=[N,H,W],native_boxes=native.copy(),positions8=np.asarray(pos),GT_read=False);sp=folder/'support'/(key+'.pt');save(sp,support);cold=torch.load(sp,map_location='cpu',weights_only=False,mmap=True);recovered=np.stack([adapter.mask_box(np.unpackbits(p)[:H*W].reshape(H,W).astype(bool))for p in cold['packed_masks']]);assert np.array_equal(recovered,info['raw_mask_boxes']);again=adapter.keep_observed(recovered,native,pos);assert np.array_equal(again,boxes);arrays['EdgeTAM']=boxes;arrays['EdgeTAM_interval']=arrays['Ours_interval'].copy();out=folder/'predictions'/(key+'.npz');out.parent.mkdir(parents=True,exist_ok=True)
    with out.with_suffix('.tmp').open('wb')as f:np.savez_compressed(f,**arrays)
    os.replace(out.with_suffix('.tmp'),out)
    with np.load(out)as z:assert set(z.files)==set(arrays)and all(np.array_equal(v,z[k])for k,v in arrays.items())
    info={k:v for k,v in info.items()if k!='raw_mask_boxes'};rec=dict(path=str(out.relative_to(D)),sha256=sha(out),support_path=str(sp.relative_to(D)),support_sha256=sha(sp),input_binding=proof,raw_mask_checkpoints=rawfiles,all_raw_masks_checkpointed_before_validation=True,all_packed_mask_boxes_cold_exact=True,protocol=info,worker=who,new_Source=0,new_original_expert=0,new_tracking_queries=1,actual_image_encoder_calls=encoder_calls[0]-ec,actual_strict_postprocess_calls=kernel_calls[0]-kc,decode_and_provenance_seconds=decode,request_wall_including_checkpoints_seconds=request,raw_checkpoint_IO_seconds=checkpoint_IO[0],nested_stage_seconds=stages,conditional_query_wall_seconds=time.perf_counter()-tick,resident_GPU_bytes=resident,peak_query_GPU_bytes=torch.cuda.max_memory_allocated(),GT_read=False,fingerprint=plan['fingerprint']);rp=folder/'receipts'/(key+'.json');atomic(rp,rec);e=dict(rec,receipt_path=str(rp.relative_to(D)),receipt_sha256=sha(rp));files[key]=e;counts['committed_queries']+=1;atomic(attempt,dict(status='committed',receipt_sha256=sha(rp)))
    if index==2:
     waiting=folder/'bridge_waiting.json';atomic(waiting,dict(status='waiting_root_first_two_actual_EdgeTAM_queries',files=list(files.values()),worker=who,GT_read=False));ack=folder/'root_actual_bridge.json'
     while not ack.exists():resources(start);time.sleep(10)
     q=read(ack);assert q['status']=='passed_first_two_actual_EdgeTAM_queries'and q['waiting_sha256']==sha(waiting)
    # Raw float logits are transient only after exact packed-support cold boxes
    # and committed prediction/receipt. Original caches are never disposed.
    for e in rawfiles:
     p=D/e['path'];assert p.resolve().is_relative_to((folder/'active').resolve())and sha(p)==e['sha256'];p.unlink()
    atomic(scratch/'disposed.json',dict(raw_mask_SHA_records=rawfiles,receipt_sha256=sha(rp),prediction_sha256=rec['sha256'],support_sha256=rec['support_sha256'],all_mask_boxes_cold_exact=True,arbitrary_logit_or_model_replay_available=False));atomic(folder/'progress.json',dict(completed=index,total=128,counts=counts,GT_read=False));del frames,arrays,support,cold,packed,boxes,again,recovered;gc.collect();torch.cuda.empty_cache()
   atomic(folder/'sealed.json',dict(queries=128,files=files,worker=who,elapsed_wall_seconds=time.perf_counter()-domain_start,GT_read=False,fingerprint=plan['fingerprint']))
  encoder_handle.remove();module.fill_holes_in_mask_scores=original_fill;atomic(D/'pilot_v1/state.json',dict(status='all256_actual_EdgeTAM_queries_sealed_pending_independent_audit_score_root',counts=counts,worker=who,seconds=time.perf_counter()-start))
 except BaseException:atomic(D/'pilot_v1/failure_preserved.json',dict(error=traceback.format_exc(),counts=counts,worker=who,elapsed_wall_seconds=time.perf_counter()-start));raise
 finally:model=None;gc.collect();torch.cuda.empty_cache();lock.close()
if __name__=='__main__':run()
