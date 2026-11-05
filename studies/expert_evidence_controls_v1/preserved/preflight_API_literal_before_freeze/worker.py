"""Only active GPU owner for original8 ReferDINO mask propagation."""
from common import *
import model as adapter
import gc,traceback
def run():
 plan=check();folder=D/'propagation';assert not(folder/'worker.json').exists()and os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
 from stvg.runtime import setup
 from studies.review_strengthening_v1.sam2_input_adapter_v1 import selected_prompts
 gate=foreign();lock=lease();setup();torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
 atomic(folder/'worker.json',dict(**identity(),original_owner_remains_stopped=gate));start=time.perf_counter();predictor=None;counts=dict(new_SAM2_queries=0,new_Source=0,new_ReferDINO=0,new_optimizer=0)
 try:
  resource(start);tick=time.perf_counter();predictor,proof=adapter.build_cpu();predictor=predictor.cuda().eval().requires_grad_(False);torch.cuda.synchronize();module,misc,ext=adapter.modules();load=time.perf_counter()-tick
  test=torch.zeros((1,1,4,4),dtype=torch.uint8,device='cuda');test[0,0,0,:2]=1;test[0,0,3,3]=1;_,areas=ext.get_connected_componnets(test);torch.cuda.synchronize();assert int(areas[0,0,0,0])==int(areas[0,0,0,1])==2 and int(areas[0,0,3,3])==1
  pc=[0];ec=[0]
  def strict_fill(mask,max_area):
   assert max_area>0;pc[0]+=1;labels,areas=misc.get_connected_components(mask<=0);return torch.where((labels>0)&(areas<=max_area),.1,mask)
  module.fill_holes_in_mask_scores=strict_fill
  handle=predictor.image_encoder.register_forward_hook(lambda m,a,o:ec.__setitem__(0,ec[0]+1))
  atomic(folder/'model_load.json',dict(worker=identity(),seconds=load,proof=proof,actual_GPU=torch.cuda.get_device_name(),actual_component_contract_test=True,experimental_not_deployment=True,old_stopped_GPU_context_excluded_from_active_pipeline=True))
  for ds in ['hc2','vidstg']:
   dsstart=time.perf_counter();files={};first=[]
   for row in rows(ds):
    resource(dsstart);foreign();tick=time.perf_counter();key=row['id'];frames,a,teacher,binding=data(ds,row);pos=a['positions8'].tolist();n,h,w,_=frames.shape;prompts=selected_prompts(teacher,h,w);binding['mask_prompt_sha256']=hashlib.sha256(prompts.tobytes()).hexdigest()
    active=folder/ds/'raw_active'/key;assert not active.exists();active.mkdir(parents=True);atomic(active/'attempt.json',dict(before_SAM2=True,input_binding=binding,worker=identity(),GT_read=False))
    packed={};raw=[];empty=[];checkpoint_seconds=[0.];preprocess={};ec0,pc0=ec[0],pc[0]
    from studies.review_strengthening_v1 import sam2_input_adapter_v1 as loader_module
    pure_loader=loader_module.lossless_rgb_loader
    def observed_loader(*args,**kwargs):
     images,hh,ww=pure_loader(*args,**kwargs);preprocess['eight_preprocessed_sha256']=hashlib.sha256(images[pos].contiguous().numpy().tobytes()).hexdigest();return images,hh,ww
    def checkpoint(i,scores):
     torch.cuda.synchronize();t=time.perf_counter();cpu=scores.detach().cpu();path=active/f'{int(i):04}.pt';save(path,dict(frame_index=int(i),scores=cpu,input_binding=binding,GT_read=False));raw.append(dict(path=str(path.relative_to(D)),sha256=sha(path)));checkpoint_seconds[0]+=time.perf_counter()-t
     assert cpu.shape==(1,1,h,w)and torch.isfinite(cpu).all()and int(i)not in packed;mask=(cpu[0,0]>0).numpy();packed[int(i)]=np.packbits(mask.reshape(-1));
     if not mask.any():empty.append(int(i))
     return cpu
    loader_module.lossless_rgb_loader=observed_loader
    try:
     torch.cuda.synchronize();request=time.perf_counter()
     with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):boxes=adapter.propagate(predictor,module,frames,pos,prompts,checkpoint)
     torch.cuda.synchronize();request=time.perf_counter()-request
    finally:loader_module.lossless_rgb_loader=pure_loader
    assert set(packed)==set(range(n))and len(raw)==n and ec[0]>ec0 and pc[0]>pc0
    mp=folder/ds/'masks'/(key+'.npz');mp.parent.mkdir(parents=True,exist_ok=True)
    with mp.open('xb')as f:np.savez_compressed(f,packed=np.stack([packed[i]for i in range(n)]),native_shape=np.array([n,h,w]),positions8=a['positions8'])
    with np.load(mp)as z:
     assert np.array_equal(z['native_shape'],[n,h,w]);cold=[]
     for i,x in enumerate(z['packed']):assert np.array_equal(x,packed[i]);cold.append(adapter.mask_box(np.unpackbits(x)[:h*w].reshape(h,w).astype(bool)))
     assert np.array_equal(np.stack(cold),boxes)
    arrays=dict(Full=a['Ours'],SAM2MatchedMask=boxes,shared_interval=a['Ours_interval'],frame_ids=a['frame_ids'],positions8=a['positions8']);out=folder/ds/'predictions'/(key+'.npz');out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('xb')as f:np.savez_compressed(f,**arrays)
    with np.load(out)as z:assert set(z.files)==set(arrays)and all(np.array_equal(v,z[k])for k,v in arrays.items())
    rec=dict(path=str(out.relative_to(D)),sha256=sha(out),mask_path=str(mp.relative_to(D)),mask_sha256=sha(mp),input_binding=binding,actual_preprocessing=preprocess,raw_float_checkpoints=raw,raw_before_validation=True,all_native_frames=n,empty_frames=len(empty),empty_conditioning_frames=sum(i in set(pos)for i in empty),empty_input_masks=int((~prompts.reshape(8,-1).any(1)).sum()),observed_mask_boxes_not_overwritten=True,ReferDINO_mask_input=True,Source_prompt_input=False,new_Source=0,new_ReferDINO=0,new_optimizer=0,new_SAM2_queries=1,actual_image_encoder_calls=ec[0]-ec0,actual_CUDA_postprocessor_calls=pc[0]-pc0,request_wall_including_checkpoint_seconds=request,checkpoint_IO_seconds=checkpoint_seconds[0],active_query_wall_seconds=time.perf_counter()-tick,worker=identity(),GT_read=False,plan_fingerprint=plan['fingerprint'])
    rp=folder/ds/'receipts'/(key+'.json');atomic(rp,rec);files[key]=dict(rec,receipt_path=str(rp.relative_to(D)),receipt_sha256=sha(rp));first.append(key);counts['new_SAM2_queries']+=1
    if len(first)==2:
     atomic(folder/ds/'bridge_input.json',dict(keys=first.copy(),worker=identity(),plan_fingerprint=plan['fingerprint']))
     log=folder/ds/'first2_bridge.log';bt=time.perf_counter()
     with log.open('xb')as f:
      child=subprocess.Popen([str(PYTHON),'-B',str(S/'report.py'),'bridge','--dataset',ds],cwd=R,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},stdout=f,stderr=subprocess.STDOUT);who=identity(child.pid);code=child.wait()
     atomic(folder/ds/'bridge_actual_exit.json',dict(worker=who,actual_subprocess_wait=True,exit_code=code,elapsed_wall_seconds=time.perf_counter()-bt,log_sha256=sha(log)));assert code==0,'Preserve all paid raw on bridge failure'
     assert read(folder/ds/'actual_bridge.json')['plan_fingerprint']==plan['fingerprint']
    disposal=[files[k]for k in first[:2]]if len(first)==2 else[files[key]]if len(first)>2 else[]
    for e in disposal:
     for v in e['raw_float_checkpoints']:
      path=D/v['path'];assert path.resolve().is_relative_to((folder/ds/'raw_active').resolve())and sha(path)==v['sha256'];path.unlink()
     atomic((D/e['raw_float_checkpoints'][0]['path']).parent/'disposed.json',dict(only_new_own_transient_raw=True,raw_SHA_records=e['raw_float_checkpoints'],packed_mask_sha256=e['mask_sha256'],prediction_sha256=e['sha256'],actual_first2_bridge_sha256=sha(folder/ds/'actual_bridge.json'),cold_mask_boxes_exact=True))
    atomic(folder/'progress.json',dict(dataset=ds,completed_in_domain=len(files),total_in_domain=128,counts=counts,worker=identity(),GT_read=False));del frames,a,teacher,prompts,packed,boxes,cold;gc.collect();torch.cuda.empty_cache()
   assert len(files)==128;atomic(folder/ds/'sealed.json',dict(queries=128,files=files,GT_read=False,plan_fingerprint=plan['fingerprint'],worker=identity(),elapsed_wall_seconds=time.perf_counter()-dsstart))
  handle.remove();atomic(folder/'GPU_complete.json',dict(status='both128_mask_prompt_domains_sealed',counts=counts,worker=identity(),elapsed_wall_seconds=time.perf_counter()-start));print('matched expert GPU complete',flush=True)
 except BaseException:atomic(folder/'failure.json',dict(error=traceback.format_exc(),counts=counts,worker=identity(),all_paid_state_preserved=True));raise
 finally:predictor=None;gc.collect();torch.cuda.empty_cache();lock.close()
if __name__=='__main__':run()
