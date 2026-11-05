"""Single actual SAM2 GPU owner, lossless masks and pre-validation raw evidence."""
from common import *
import argparse,gc,subprocess,traceback
import model as adapter
def run(stage):
 plan=check();folder=D/stage;assert not(folder/'worker.json').exists();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
 from stvg.runtime import lease,setup
 foreign();lock=lease();setup();torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;sys.addaudithook(inference_guard);atomic(folder/'worker.json',identity());start=time.perf_counter();predictor=None;counts=dict(new_SAM2_queries=0,reused_pilot_queries=0,new_Source=0,new_ReferDINO=0,new_optimizer=0)
 try:
  resource(start);tick=time.perf_counter();predictor,loadproof=adapter.build_cpu();predictor=predictor.cuda().eval().requires_grad_(False);torch.cuda.synchronize();module,misc,ext=adapter.modules();loadsecs=time.perf_counter()-tick
  # Exact pinned official CUDA postprocessor; reject its optional silent fallback.
  test=torch.zeros((1,1,4,4),dtype=torch.uint8,device='cuda');test[0,0,0,:2]=1;test[0,0,3,3]=1;_,areas=ext.get_connected_componnets(test);torch.cuda.synchronize();assert int(areas[0,0,0,0])==int(areas[0,0,0,1])==2 and int(areas[0,0,3,3])==1
  postcalls=[0];encoder=[0]
  def strict_fill(mask,max_area):
   assert max_area>0;postcalls[0]+=1;labels,areas=misc.get_connected_components(mask<=0);return torch.where((labels>0)&(areas<=max_area),0.1,mask)
  module.fill_holes_in_mask_scores=strict_fill
  handle=predictor.image_encoder.register_forward_hook(lambda m,a,o:encoder.__setitem__(0,encoder[0]+1))
  atomic(folder/'model_load.json',dict(worker=identity(),seconds=loadsecs,proof=loadproof,official_CUDA_component_kernel_actual_test_passed=True,actual_GPU=torch.cuda.get_device_name(),precision='FP32 resident weights, BF16 CUDA autocast, TF32 enabled as pinned official notebook',new_Source=0,new_ReferDINO=0))
  for ds in COUNTS:
   dsstart=time.perf_counter();files={};newkeys=[];old=fixed(D/'pilot'/ds/'sealed.json')['files']if stage=='full'else{}
   for row in rows(stage,ds):
    resource(dsstart);key=row['id']
    if key in old:
     e=old[key]
     for p,h in [('path','sha256'),('mask_path','mask_sha256'),('receipt_path','receipt_sha256')]:assert sha(D/e[p])==e[h]
     files[key]=e;counts['reused_pilot_queries']+=1;continue
    foreign();tick=time.perf_counter();frames,a,binding=data(ds,row);pos=a['positions8'].astype(int).tolist();prompts=a['Source'][pos];n,h,w,_=frames.shape;active=folder/ds/'raw_active'/key;assert not active.exists();active.mkdir(parents=True)
    atomic(active/'attempt.json',dict(before_SAM2=True,input_binding=binding,worker=identity(),GT_read=False));raw=[];packed={};empty=[];checkpoint_seconds=[0.];ec=encoder[0];pc=postcalls[0]
    original=module.load_video_frames;preprocess={}
    # Source decoded RGB already checked against fixed video SHA; observer sees
    # the actual tensors fed to the official image encoder, without alteration.
    from studies.review_strengthening_v1.sam2_input_adapter_v1 import lossless_rgb_loader
    def observed_loader(*args,**kwargs):
     images,hh,ww=lossless_rgb_loader(*args,**kwargs);preprocess['eight_preprocessed_sha256']=hashlib.sha256(images[pos].contiguous().numpy().tobytes()).hexdigest();return images,hh,ww
    def checkpoint(i,scores):
     torch.cuda.synchronize();t=time.perf_counter();cpu=scores.detach().cpu();path=active/f'{int(i):04}.pt';save(path,dict(frame_index=int(i),scores=cpu,input_binding=binding,GT_read=False));raw.append(dict(path=str(path.relative_to(D)),sha256=sha(path)));checkpoint_seconds[0]+=time.perf_counter()-t
     assert cpu.shape==(1,1,h,w)and torch.isfinite(cpu).all()and int(i)not in packed;mask=(cpu[0,0]>0).numpy();packed[int(i)]=np.packbits(mask.reshape(-1));
     if not mask.any():empty.append(int(i))
     return cpu
    # Adapter scopes the loader itself. Wrap its unchanged pure loader function
    # only to record the actual normalized eight conditioning tensors.
    import studies.review_strengthening_v1.sam2_input_adapter_v1 as loader_module
    pure_loader=loader_module.lossless_rgb_loader;loader_module.lossless_rgb_loader=observed_loader
    try:
     torch.cuda.synchronize();request=time.perf_counter()
     with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):boxes=adapter.propagate(predictor,module,frames,pos,prompts,checkpoint)
     torch.cuda.synchronize();request=time.perf_counter()-request
    finally:loader_module.lossless_rgb_loader=pure_loader
    assert set(packed)==set(range(n))and len(raw)==n and encoder[0]>ec and postcalls[0]>pc
    mp=folder/ds/'masks'/(key+'.npz');mp.parent.mkdir(parents=True,exist_ok=True)
    with mp.open('xb')as f:np.savez_compressed(f,packed=np.stack([packed[i]for i in range(n)]),native_shape=np.array([n,h,w]),positions8=a['positions8'])
    with np.load(mp)as z:
     assert np.array_equal(z['native_shape'],[n,h,w]);cold=[]
     for i,x in enumerate(z['packed']):
      mask=np.unpackbits(x)[:h*w].reshape(h,w).astype(bool);assert np.array_equal(x,packed[i]);cold.append(adapter.mask_box(mask))
     assert np.array_equal(np.stack(cold),boxes)
    a['SAM2Native']=boxes.copy();a['SAM2TR']=boxes.copy();a['SAM2Native_interval']=a['Source_interval'].copy();a['SAM2TR_interval']=a['SourceTR_interval'].copy();out=folder/ds/'predictions'/(key+'.npz');out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('xb')as f:np.savez_compressed(f,**a)
    with np.load(out)as z:assert set(z.files)==set(a)and all(np.array_equal(v,z[k])for k,v in a.items())
    rec=dict(path=str(out.relative_to(D)),sha256=sha(out),mask_path=str(mp.relative_to(D)),mask_sha256=sha(mp),input_binding=binding,actual_preprocessing=preprocess,raw_float_checkpoints=raw,raw_before_validation=True,all_native_frames=n,empty_frames=len(empty),empty_conditioning_frames=sum(i in set(pos)for i in empty),observed_mask_boxes_not_overwritten=True,Source_boxes_only_prompt=True,ReferDINO_input=False,new_Source=0,new_ReferDINO=0,new_optimizer=0,new_SAM2_queries=1,actual_image_encoder_calls=encoder[0]-ec,actual_CUDA_postprocessor_calls=postcalls[0]-pc,request_wall_including_checkpoint_seconds=request,checkpoint_IO_seconds=checkpoint_seconds[0],active_query_wall_seconds=time.perf_counter()-tick,worker=identity(),GT_read=False,plan_fingerprint=plan['fingerprint'])
    rp=folder/ds/'receipts'/(key+'.json');atomic(rp,rec);files[key]=dict(rec,receipt_path=str(rp.relative_to(D)),receipt_sha256=sha(rp));newkeys.append(key);counts['new_SAM2_queries']+=1
    if len(newkeys)==2:
     bridge=D/stage/ds/'bridge_input.json';atomic(bridge,dict(keys=newkeys.copy(),worker=identity(),plan_fingerprint=plan['fingerprint'],auto_actual_independent_CPU_bridge=True))
     log=folder/ds/'first2_bridge.log';bt=time.perf_counter()
     with log.open('xb')as f:child=subprocess.Popen([str(PYTHON),'-B',str(S/'report.py'),'bridge','--stage',stage,'--dataset',ds],cwd=R,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},stdout=f,stderr=subprocess.STDOUT);who=identity(child.pid);code=child.wait()
     atomic(folder/ds/'bridge_actual_exit.json',dict(worker=who,actual_subprocess_wait=True,exit_code=code,elapsed_wall_seconds=time.perf_counter()-bt,log_sha256=sha(log)));assert code==0,'Preserve all raw paid SAM2 state on bridge failure'
     assert read(folder/ds/'actual_bridge.json')['plan_fingerprint']==plan['fingerprint']
    # Keep first paid request until the actual independent first-two bridge.
    disposal=[files[k]for k in newkeys[:2]]if len(newkeys)==2 else[files[key]]if len(newkeys)>2 else[]
    for e in disposal:
     for rawrec in e['raw_float_checkpoints']:
      path=D/rawrec['path'];assert path.resolve().is_relative_to((folder/ds/'raw_active').resolve())and sha(path)==rawrec['sha256'];path.unlink()
     proofpath=(D/e['raw_float_checkpoints'][0]['path']).parent/'disposed.json';atomic(proofpath,dict(raw_SHA_records=e['raw_float_checkpoints'],prediction_sha256=e['sha256'],packed_mask_sha256=e['mask_sha256'],cold_mask_and_boxes_exact=True,first2_independent_bridge_sha256=sha(folder/ds/'actual_bridge.json'),only_new_own_transient_raw_disposed=True))
    atomic(folder/'progress.json',dict(dataset=ds,completed_in_domain=len(files),total_in_domain=len(rows(stage,ds)),counts=counts,GT_read=False,worker=identity()));del frames,a,packed,boxes,cold;gc.collect();torch.cuda.empty_cache()
   atomic(folder/ds/'sealed.json',dict(queries=len(files),files=files,stage=stage,GT_read=False,plan_fingerprint=plan['fingerprint'],worker=identity(),elapsed_wall_seconds=time.perf_counter()-dsstart));assert len(files)==len(rows(stage,ds))
  handle.remove();atomic(folder/'GPU_complete.json',dict(status='all_stage_domains_SAM2_predictions_sealed',counts=counts,worker=identity(),elapsed_wall_seconds=time.perf_counter()-start));print(stage,'SAM2 all domains sealed',flush=True)
 except BaseException:atomic(folder/'failure.json',dict(error=traceback.format_exc(),counts=counts,worker=identity(),all_paid_state_preserved=True));raise
 finally:
  predictor=None;gc.collect();torch.cuda.empty_cache();lock.close()
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--stage',choices=['pilot','full'],required=True);x=a.parse_args();run(x.stage)
