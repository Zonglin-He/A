"""One gated expert forward/query on256retained Source grids, no GT scoring."""
from eg_common_v1 import *
import argparse,traceback,gc,copy
@lru_cache(None)
def validate_pilot():
 validate_math();p=read(D/'pilot_plan_v1.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for field in ['code_sha256','input_sha256']:
  for f,h in p[field].items():assert sha(R/f)==h,f
 lock=read(D/'source_lock.json');assert lock['commit']==p['source_commit']
 for f,h in lock['files_sha256'].items():assert sha(D/'vendor/ReferFormer'/f)==h,f
 assert sha(D/'vendor/r50_refytvos_finetune.pth')==p['checkpoint_sha256']
 for f,e in read(D/'CPU_model_plan.json')['text_assets']['files'].items():assert sha(D/'vendor/roberta-base'/f)==e['sha256'],f
 doctor=read(D/'runtime_doctor_receipt.json');assert sha(R/'runtime/referdino/cuda_build/MultiScaleDeformableAttention.so')==doctor['compiled_extension_sha256']
 for f,h in doctor['dependency_overlay_files_sha256'].items():assert sha(D/'runtime/python'/f)==h,f
 return p
def save(path,value):
 import torch
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');assert not path.exists();torch.save(value,tmp);os.replace(tmp,path)
def payload(ds,row):
 import torch,numpy as np
 from stvg.decode import decode
 from stvg.teacher_bridge import evidence_key
 cap,parent=rc.selected.suite().load_parent(ds,row);entry=fixed(R/'outputs/map_paper_completion_v1/raw_reuse_index.json')[ds][row['id']];assert sha(entry['path'])==entry['sha256'];raw=torch.load(entry['path'],map_location='cpu',weights_only=False,mmap=True)
 assert not raw['GT_read']and raw['parent_capture_sha256']==parent['capture_sha256']and raw['frame_ids']==cap['frame_ids']
 frames,ids=decode(row['metadata'],ds);assert ids==cap['frame_ids'];pos=cap['positions8'];key,ctx=evidence_key(frames[pos],row['metadata']['caption'],positions=pos,frame_ids=ids,fingerprint=fixed(rc.RUN/'plan.json')['provider_runtime_fingerprint'],dataset=ds)
 roots=[rc.PROVIDER/'outputs/boxfixed_full_v1/teacher_cache',R.parent/'stvg-mainline/outputs/tastvg_paper_v1/teacher_cache'];hits=[root for root in roots if(root/(key+'.json')).exists()];assert hits;root=hits[0];meta=read(root/(key+'.json'));assert meta['key']==key and meta['context']==ctx and sha(root/(key+'.pt'))==meta['sha256'];oldteacher=torch.load(root/(key+'.pt'),map_location='cpu',weights_only=False,mmap=True)
 assert np.array_equal(oldteacher['boxes'].numpy(),np.asarray(cap['geometry']['native_boxes']))and oldteacher['native_same_track_all8']
 pe,ce=rc.paths(ds,row['id'])
 with np.load(rc.RUN/ds/pe['path'])as z,np.load(rc.selected.D/ce['path'])as cur:reference=dict(CurrentSimple=z['Interpolation'].copy(),CurrentOurs=cur['FullMAP'].copy(),CurrentOurs_interval=cur['FullMAP_interval'].copy())
 proof=dict(parent=parent,retained_raw_path=entry['path'],retained_raw_sha256=entry['sha256'],current_prediction_sha256=ce['sha256'],current_receipt_sha256=ce['receipt_sha256'],current_teacher_payload_path=str(root/(key+'.pt')),current_teacher_payload_sha256=meta['sha256'],current_teacher_receipt_path=str(root/(key+'.json')),current_teacher_receipt_sha256=sha(root/(key+'.json')),current_teacher_key=key,actual_eight_RGB_sha256=hashlib.sha256(frames[pos].tobytes()).hexdigest(),caption_sha256=hashlib.sha256(' '.join(row['metadata']['caption'].lower().split()).encode()).hexdigest(),positions8=pos,frame_ids=ids,new_Source=0,new_original_expert=0,GT_read=False)
 return cap,raw,frames,oldteacher,reference,proof
def cpu_tree(x):
 import torch
 if torch.is_tensor(x):return x.detach().cpu()
 if isinstance(x,dict):return{k:cpu_tree(v)for k,v in x.items()}
 if isinstance(x,(list,tuple)):return type(x)(cpu_tree(v)for v in x)
 return copy.deepcopy(x)
def require_release():
 import eg_pilot_gate_v1 as gate
 value=gate.gate();assert value is not None,'No model construction until actual predecessor exit/full audits/root publication';return value
def run():
 import torch,numpy as np
 from PIL import Image
 import torchvision.transforms as T
 from stvg.runtime import setup,lease
 import eg_math_core_v1 as math
 # Load uniquely named local adapters before the official model changes sys.path.
 import importlib.util
 spec=importlib.util.spec_from_file_location('_referring_expert_output_interface_v1',S/'interface.py');adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
 spec=importlib.util.spec_from_file_location('_referring_expert_model_builder_v1',S/'referformer_adapter.py');builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
 plan=validate_pilot();require_release();assert read(D/'math_preflight_v1_receipt.json')['status']=='passed_four_actual_raw_grid_full_expert_chain_rebuild';dest=D/'pilot_v1';assert not(dest/'worker.json').exists();setup();torch.set_num_threads(1);rc.gt_guard();leasefile=lease();who=identity();atomic(dest/'worker.json',who);counts=dict(attempted_new_expert=0,successful_new_expert=0,committed_queries=0,new_Source=0,new_original_expert=0,new_optimizer=0);start=time.perf_counter();model=None
 try:
  resources(start);tick=time.perf_counter();model,tokenizer,load_audit=builder.build_cpu();assert Path(sys.modules['models'].__file__).resolve().is_relative_to((D/'vendor/ReferFormer').resolve());model=model.cuda().eval().requires_grad_(False);torch.cuda.synchronize();model_load=time.perf_counter()-tick
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.reset_peak_memory_stats();atomic(dest/'model_load.json',dict(worker=who,seconds=model_load,load_audit=load_audit,device_name=torch.cuda.get_device_name(),torch_version=torch.__version__,compute_dtype='float32',TF32=False,model_instances=1))
  from util.misc import nested_tensor_from_videos_list
  import MultiScaleDeformableAttention as extension
  original_forward=extension.ms_deform_attn_forward;kernel_calls=[0]
  def observed_kernel(*args,**kw):
   assert args[0].is_cuda;kernel_calls[0]+=1;return original_forward(*args,**kw)
  extension.ms_deform_attn_forward=observed_kernel
  transform=T.Compose([T.Resize(360),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
  for ds in rc.COUNTS:
   files={};domain_start=time.perf_counter()
   for index,row in enumerate(pilot_rows(ds),1):
    resources(start);key=row['id'];folder=dest/ds;attempt=folder/'attempts'/(key+'.json');assert not attempt.exists();tick=time.perf_counter();cap,raw,frames,oldteacher,ref,proof=payload(ds,row);compat_start=time.perf_counter();before,_=math.build(cap,raw,oldteacher,ds)
    assert np.array_equal(before['Ours'],ref['CurrentOurs'])and np.array_equal(before['Ours_interval'],ref['CurrentOurs_interval'])and np.array_equal(before['Simple'],ref['CurrentSimple']);compatibility=time.perf_counter()-compat_start
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();resident_bytes=torch.cuda.memory_allocated();request=time.perf_counter();prep=time.perf_counter();expr=' '.join(row['metadata']['caption'].lower().split());cpu_inputs=torch.stack([transform(Image.fromarray(f))for f in frames[cap['positions8']]]);shape=list(cpu_inputs.shape);text_ids=tokenizer(expr,return_tensors='pt')['input_ids'];assert text_ids.shape[-1]<=tokenizer.model_max_length;CPU_preprocess=time.perf_counter()-prep
    atomic(attempt,dict(status='about_to_call_new_expert_once',worker=who,input_binding=proof,input_shape=shape,token_count=int(text_ids.shape[-1]),text_token_sha256=hashlib.sha256(text_ids.numpy().tobytes()).hexdigest(),GT_read=False));torch.cuda.synchronize();pre=time.perf_counter();images=cpu_inputs.cuda();samples=nested_tensor_from_videos_list([images]);assert not samples.mask.any();target=dict(size=torch.tensor(shape[-2:],device='cuda'));torch.cuda.synchronize();fw=time.perf_counter();kc=kernel_calls[0];counts['attempted_new_expert']+=1
    with torch.inference_mode():outputs=model(samples,[expr],[target])
    torch.cuda.synchronize();forward_end=time.perf_counter();counts['successful_new_expert']+=1
    # Paid full raw output lands before selection/output assertions or pooling.
    cp=folder/'raw_expert_outputs'/(key+'.pt');raw_tick=time.perf_counter();cpu_outputs=cpu_tree(outputs);save(cp,dict(outputs=cpu_outputs,input_binding=proof,input_shape=shape,worker=who,GT_read=False));raw_record=dict(path=str(cp.relative_to(D)),sha256=sha(cp),before_selected_query_validation=True,before_ROI_or_geometry=True);raw_io=time.perf_counter()-raw_tick
    assert kernel_calls[0]>kc,'No silent alternative CUDA operator path';post_tick=time.perf_counter();teacher=adapter.selected_eight(cpu_outputs,shape);teacher.update(text_sha256=proof['caption_sha256'],model_checkpoint_sha256=plan['checkpoint_sha256'],model_source_commit=plan['source_commit'],positions8=cap['positions8'],frame_ids=cap['frame_ids'],evidence_binding_verified=True,query_forwards=1)
    post=time.perf_counter()-post_tick;teacher['cost']=dict(CPU_RGB_preprocessing_and_tokenization_seconds=CPU_preprocess,transfer_and_nested_tensor_seconds=fw-pre,forward_seconds=forward_end-fw,selected_postprocess_seconds=post,raw_output_checkpoint_seconds=raw_io,request_wall_including_CPU_preprocess_transfer_checkpoint_seconds=time.perf_counter()-request);tp=folder/'expert_payloads'/(key+'.pt');save(tp,teacher);teacher_record=dict(path=str(tp.relative_to(D)),sha256=sha(tp),before_both_inference_arms=True)
    finish_tick=time.perf_counter();arrays,compact=math.build(cap,raw,teacher,ds);arrays.update(ref);arrays['CurrentSimple_interval']=arrays['Source_interval'].copy();finish=time.perf_counter()-finish_tick;compact_path=folder/'compact'/(key+'.pt');save(compact_path,compact);cold=torch.load(compact_path,map_location='cpu',weights_only=False,mmap=True);again=math.replay(cold);assert all(np.array_equal(v,again[k])for k,v in arrays.items()if k in again)
    pred=folder/'predictions'/(key+'.npz');pred.parent.mkdir(parents=True,exist_ok=True)
    with pred.with_suffix('.tmp').open('wb')as f:np.savez_compressed(f,**arrays)
    os.replace(pred.with_suffix('.tmp'),pred)
    with np.load(pred)as z:assert set(z.files)==set(arrays)and all(np.array_equal(v,z[k])for k,v in arrays.items())
    receipt=dict(path=str(pred.relative_to(D)),sha256=sha(pred),compact_path=str(compact_path.relative_to(D)),compact_sha256=sha(compact_path),expert_payload=teacher_record,raw_expert_output=raw_record,input_binding=proof,original_expert_all_chain_exact=True,new_expert_queries=1,new_expert_observations=8,new_Source=0,new_original_expert=0,new_optimizer=0,all_cold_arrays_exact=True,Source_event_for_both_simple=True,own_new_expert_event_for_Ours=True,official_CUDA_kernel_calls=kernel_calls[0]-kc,expert_cost=teacher['cost'],selected_full_chain_seconds=finish,old_compatibility_seconds=compatibility,conditional_query_wall_seconds=time.perf_counter()-tick,resident_model_GPU_bytes=resident_bytes,query_peak_GPU_bytes=torch.cuda.max_memory_allocated(),worker=who,GT_read=False,pilot_fingerprint=plan['fingerprint']);rp=folder/'receipts'/(key+'.json');atomic(rp,receipt);files[key]=dict(receipt,receipt_path=str(rp.relative_to(D)),receipt_sha256=sha(rp));counts['committed_queries']+=1;atomic(attempt,dict(status='committed',receipt_sha256=sha(rp),worker=who))
    atomic(folder/'progress.json',dict(completed=index,total=128,counts=counts,GT_read=False,worker=who))
    if index==2:
     waiting=folder/'bridge_waiting.json';atomic(waiting,dict(status='waiting_root_actual_first_two_new_expert_bridge',worker=who,files=list(files.values()),model_load_sha256=sha(dest/'model_load.json'),GT_read=False))
     ack=folder/'root_actual_bridge.json'
     while not ack.exists():resources(start);time.sleep(10)
     a=read(ack);assert a['status']=='passed_actual_first_two_new_expert_bridge'and a['waiting_sha256']==sha(waiting)
    del cap,raw,frames,oldteacher,ref,before,cpu_inputs,images,samples,outputs,cpu_outputs,teacher,compact,cold,arrays,again;gc.collect();torch.cuda.empty_cache()
   assert len(files)==128;atomic(folder/'sealed.json',dict(status='fixed128_pilot_predictions_committed_no_GT_scoring',queries=128,files=files,elapsed_wall_seconds=time.perf_counter()-domain_start,worker=who,GT_read=False,pilot_fingerprint=plan['fingerprint']))
  extension.ms_deform_attn_forward=original_forward;atomic(dest/'state.json',dict(status='all256_pilot_committed_pending_independent_audit_root_cost_feasibility_and_full_release',counts=counts,worker=who,elapsed_wall_seconds=time.perf_counter()-start,full_experiment_complete=False))
 except BaseException:atomic(dest/'failure_preserved.json',dict(error=traceback.format_exc(),counts=counts,worker=who,elapsed_wall_seconds=time.perf_counter()-start));raise
 finally:
  model=None;gc.collect();torch.cuda.empty_cache();leasefile.close()
if __name__=='__main__':run()
