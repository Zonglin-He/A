from common_v1 import *
import adapter_v1 as adapter
import argparse,gc,traceback,subprocess

def foreign_check():
 out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
 entries=[]
 for line in out.splitlines():
  pid,name=line.split(',',1);pid=int(pid);exe=Path(f'/proc/{pid}/exe').resolve();assert str(exe)=='/opt/todesk/bin/ToDesk_Session',(pid,str(exe));entries.append(dict(pid=pid,exe=str(exe)))
 return entries

def run(expert):
 plan=validate();start=time.perf_counter();folder=D/expert;assert not(folder/'worker.json').exists();resources(start);guard();torch.set_num_threads(1);torch.manual_seed(20261026);np.random.seed(20261026);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 from stvg.runtime import lease
 foreign=foreign_check();lock=lease();who=identity();atomic(folder/'worker.json',who);atomic(folder/'GPU_gate.json',dict(allowed_only_exact_ToDesk=foreign,global_gpu_lock_acquired=True,worker=who));counts=dict(attempted_new_expert=0,successful_new_expert=0,committed=0,new_Source=0,new_original_ReferDINO=0,new_optimizer=0);model=None
 try:
  tick=time.perf_counter();model,tok,load_audit=adapter.construct(expert);model=model.cuda().eval().requires_grad_(False);torch.cuda.synchronize();atomic(folder/'model_load.json',dict(worker=who,seconds=time.perf_counter()-tick,checkpoint_sha256=plan['input_sha256'][str((D/'vendor'/('referformer_r101_refytvos.pth'if expert=='referformer_r101'else'mttr_refytvos_window12.pth.tar')).relative_to(R))],load_audit=load_audit,device=torch.cuda.get_device_name(),torch_version=torch.__version__,dtype='FP32',TF32=False,instances=1))
  kernel=[0];original_kernel=None
  if expert=='referformer_r101':
   import MultiScaleDeformableAttention as extension
   original_kernel=extension.ms_deform_attn_forward
   def observed(*a,**kw):assert a[0].is_cuda;kernel[0]+=1;return original_kernel(*a,**kw)
   extension.ms_deform_attn_forward=observed
  for ds in ['hc2','vidstg']:
   dest=folder/ds;files={};domstart=time.perf_counter()
   for index,row in enumerate(rows(ds),1):
    resources(start);qt=time.perf_counter();key=row['id'];attempt=dest/'attempts'/(key+'.json');assert not attempt.exists();cap,raw,frames,current,refs,proof=read_inputs(ds,row)
    expr=' '.join(row['metadata']['caption'].lower().split());pre=time.perf_counter();cpu=adapter.preprocess(expert,frames[cap['positions8']]);shape=list(cpu.shape);textids=tok(expr,return_tensors='pt')['input_ids'];assert textids.shape[-1]<=tok.model_max_length;preprocess=time.perf_counter()-pre
    atomic(attempt,dict(status='before_one_actual_expert_forward',input_binding=proof,shape=shape,text_token_sha256=hashlib.sha256(textids.numpy().tobytes()).hexdigest(),expert=expert,worker=who,GT_read=False));torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();fw=time.perf_counter();kc=kernel[0];counts['attempted_new_expert']+=1
    with torch.inference_mode():outputs=adapter.forward(model,expert,cpu,expr)
    torch.cuda.synchronize();forward=time.perf_counter()-fw;counts['successful_new_expert']+=1
    # Full paid raw state lands BEFORE selection or any expert geometry checks.
    cp=dest/'raw_expert_outputs'/(key+'.pt');rawtick=time.perf_counter();coldout=cpu_tree(outputs);save(cp,dict(outputs=coldout,input_binding=proof,input_shape=shape,original_shape=list(frames.shape[1:3]),processed_input_sha256=hashlib.sha256(cpu.numpy().tobytes()).hexdigest(),text_token_sha256=hashlib.sha256(textids.numpy().tobytes()).hexdigest(),expert=expert,worker=who,GT_read=False));rawsha=sha(cp);rawio=time.perf_counter()-rawtick
    if expert=='referformer_r101':assert kernel[0]>kc
    pp=time.perf_counter();teacher=adapter.select(expert,coldout,shape,frames.shape[1:3]);post=time.perf_counter()-pp;teacher.update(model_checkpoint_sha256=read(folder/'model_load.json')['checkpoint_sha256'],positions8=cap['positions8'],frame_ids=cap['frame_ids'],text_sha256=proof['caption_sha256'],query_forwards=1,evidence_binding_verified=True);tp=dest/'expert_payloads'/(key+'.pt');save(tp,teacher)
    finish=time.perf_counter();arrays,compact=build(cap,raw,teacher,ds);finish=time.perf_counter()-finish
    record=dict(input_binding=proof,raw_expert_output=dict(path=str(cp.relative_to(D)),sha256=rawsha,before_selected_query_validation=True,before_ROI_or_geometry=True),expert_payload=dict(path=str(tp.relative_to(D)),sha256=sha(tp),before_both_inference_arms=True),expert=expert,worker=who,new_expert_queries=1,new_expert_observations=8,new_Source=0,new_original_expert=0,new_optimizer=0,GT_read=False,official_CUDA_kernel_calls=kernel[0]-kc,empty_mask_to_full_frame_boxes=teacher.get('empty_box_conversion',0),cost=dict(preprocess_seconds=preprocess,transfer_and_forward_seconds=forward,full_raw_checkpoint_seconds=rawio,postprocess_seconds=post,finish_seconds=finish,query_wall_before_commit_seconds=time.perf_counter()-qt),peak_allocated_GPU_bytes=torch.cuda.max_memory_allocated(),peak_reserved_GPU_bytes=torch.cuda.max_memory_reserved(),plan_fingerprint=plan['fingerprint'])
    e=commit(dest,key,arrays,compact,record);files[key]=e;counts['committed']+=1;atomic(attempt,dict(status='committed',receipt_sha256=e['receipt_sha256'],worker=who));atomic(dest/'progress.json',dict(completed=index,total=128,counts=counts,worker=who,GT_read=False))
    if index==2:
     wait=dest/'bridge_waiting.json';atomic(wait,dict(status='waiting_root_actual_first_two_expert_payload_and_fullmath_bridge',files=list(files.values()),model_load_sha256=sha(folder/'model_load.json'),worker=who));atomic(folder/'state.json',dict(status='waiting_root_bridge',dataset=ds,worker=who))
     ack=dest/'root_actual_bridge.json'
     while not ack.exists():resources(start);time.sleep(5)
     ack=read(ack);assert ack['waiting_sha256']==sha(wait)and ack['status']=='passed_actual_two_new_expert_full_input_selection_Epool_geometry_posterior_finalDP_cold_exact'
    atomic(folder/'state.json',dict(status='predicting',dataset=ds,completed=index,worker=who));del cap,raw,frames,current,refs,cpu,textids,outputs,coldout,teacher,arrays,compact;gc.collect();torch.cuda.empty_cache()
   atomic(dest/'sealed.json',dict(queries=128,files=files,plan_fingerprint=plan['fingerprint'],GT_read=False,worker=who,elapsed_wall_seconds=time.perf_counter()-domstart))
  atomic(folder/'GPU_complete.json',dict(status='both128domains_sealed',counts=counts,worker=who,elapsed_wall_seconds=time.perf_counter()-start));print(expert,'all256 new expert queries sealed',flush=True)
 except BaseException:atomic(folder/'failure.json',dict(error=traceback.format_exc(),counts=counts,worker=who,elapsed_wall_seconds=time.perf_counter()-start,paid_outputs_attempts_preserved=True));raise
 finally:
  if expert=='referformer_r101'and 'original_kernel'in locals()and original_kernel is not None:extension.ms_deform_attn_forward=original_kernel
  model=None;gc.collect();torch.cuda.empty_cache();lock.close()
def reuse(expert):
 validate();guard();torch.set_num_threads(1);assert expert in REUSE;start=time.perf_counter()
 for ds in ['hc2','vidstg']:
  folder=D/expert/ds;assert not(folder/'sealed.json').exists();files={}
  for index,row in enumerate(rows(ds),1):
   resources(start);tick=time.perf_counter();arrays,c,proof=cached_reuse_derive(expert,ds,row);source=proof
   files[row['id']]=commit(folder,row['id'],arrays,c,dict(input_binding=proof,expert=expert,inherited_expert=source,new_expert_queries=0,new_Source=0,new_original_expert=0,new_optimizer=0,GT_read=False,worker=identity(),cached_derive_wall_seconds=time.perf_counter()-tick));atomic(folder/'progress.json',dict(completed=index,total=128,GT_read=False))
  atomic(folder/'sealed.json',dict(queries=128,files=files,GT_read=False,worker=identity(),existing_Current_arrays_exact=True))
 print(expert,'cached256 sealed without any new expert/Source/GT',flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['GPU','reuse']);a.add_argument('--expert',required=True,choices=EXPERTS+REUSE);q=a.parse_args();(run if q.stage=='GPU'else reuse)(q.expert)
