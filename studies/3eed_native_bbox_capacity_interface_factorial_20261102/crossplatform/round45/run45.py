"""R45 frozen7B nativebbox; never run without root execution authorization."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from common45 import *
from dataclasses import replace
from PIL import Image
from association import DEFAULT_CONFIG
from local_engine37 import LocalFrozenVLM
from common39 import messages_bbox

def native_call(engine,reference_processor,r,case,expected,key,phase,planhash,ledger):
 rawpath=OUT/'raw_answers_PRIVATE'/(key+'.json');tokenpath=OUT/'token_ids_PRIVATE'/(key+'.json');inputpath=OUT/'input_bindings_PRIVATE'/(key+'.json');resultpath=OUT/'calls_PRIVATE'/(key+'.json')
 if rawpath.exists() or tokenpath.exists() or resultpath.exists():raise RuntimeError('Existing generation retained; no implicit replay or retry')
 t=engine.torch;im=Image.open(r['imagepath']).convert('RGB');assert list(im.size)==case['image_size_wh'];rgbhash=array_sha(np.asarray(im));assert [rgbhash]==expected['images_array_sha256']
 pre=engine.processor.image_processor(images=[im],return_tensors='pt');grid=pre['image_grid_thw'].tolist();assert grid==expected['image_grid_thw'] and len(grid)==1 and grid[0][0]==1
 W,H=grid[0][2]*14,grid[0][1]*14;vt=grid[0][1]*grid[0][2]//4;assert 256<=vt<=1024
 prompt=engine.processor.apply_chat_template(messages_bbox(r['caption'],[W,H]),tokenize=False,add_generation_prompt=True);ph=hashlib.sha256(prompt.encode()).hexdigest();qh=hashlib.sha256(r['caption'].encode()).hexdigest();assert ph==expected['serialized_prompt_sha256'] and qh==expected['query_sha256']
 cpu_inputs=engine.processor(text=[prompt],images=[im],return_tensors='pt',padding=False,truncation=False)
 ref_inputs=reference_processor(text=[prompt],images=[im],return_tensors='pt',padding=False,truncation=False)
 assert set(cpu_inputs)==set(ref_inputs)
 assert all(t.equal(cpu_inputs[k],ref_inputs[k]) for k in cpu_inputs),'3B/7B CPU input tensors differ'
 input_ids_sha=array_sha(cpu_inputs['input_ids'].numpy());pixel_values_sha=array_sha(cpu_inputs['pixel_values'].numpy());del ref_inputs
 inputs=cpu_inputs.to('cuda');assert inputs['image_grid_thw'].tolist()==grid and int(inputs['input_ids'].shape[1])==expected['input_tokens'] and int(inputs['input_ids'].shape[1])<=8192
 cfg=engine.model.generation_config;assert cfg.repetition_penalty==1.05 and cfg.bos_token_id==151643 and cfg.pad_token_id==151643 and cfg.eos_token_id==[151645,151643]
 bind={'alias':r['alias'],'key':key,'phase':phase,'plan_sha256':planhash,'image_grid_thw':grid,'model_resized_wh':[W,H],'original_wh':case['image_size_wh'],'image_visual_tokens':[vt],'input_tokens':int(inputs['input_ids'].shape[1]),'serialized_prompt_sha256':ph,'query_sha256':qh,'images_array_sha256':[rgbhash],'max_new_tokens':96,'eos_token_ids':cfg.eos_token_id,'actual_repetition_penalty':1.05,'same_input_as_cached3B':True,'3B7B_reconstructed_CPU_input_tensors_exact':True,'input_ids_array_sha256':input_ids_sha,'pixel_values_array_sha256':pixel_values_sha,'memory_before_generate':memory_receipt(t)}
 atomic(inputpath,bind);events=ledger.read_text().splitlines() if ledger.exists() else [];assert len(events)<208
 event={'attempt_index':len(events)+1,'key':key,'alias':r['alias'],'phase':phase,'started_unix':time.time(),'raw_target':str(rawpath.relative_to(OUT))}
 with ledger.open('a') as f:f.write(json.dumps(event)+'\n');f.flush();os.fsync(f.fileno())
 t.cuda.synchronize();start=time.perf_counter()
 with t.inference_mode():generated=engine.model.generate(**inputs,do_sample=False,num_beams=1,max_new_tokens=96,use_cache=True,repetition_penalty=1.05)
 tail=generated[0,inputs['input_ids'].shape[1]:]
 # Persist generated IDs even if token decoding itself should fail.
 tokens=tail.detach().cpu().tolist();atomic(tokenpath,{'generated_token_ids':tokens})
 raw=engine.processor.tokenizer.decode(tail,skip_special_tokens=True,clean_up_tokenization_spaces=False)
 capture_decoded_raw(rawpath,raw)
 # No model output is parsed/hashed before the raw write above.
 t.cuda.synchronize();bind.update(seconds_generate_including_durable_output=time.perf_counter()-start,memory_after_generate=memory_receipt(t))
 return annotate_saved(rawpath,bind,tokenpath,resultpath)

def main():
 verification_started=time.perf_counter();plan,auth,download=check_authorization();verify(ROOT/'crossplatform/round37/prepared208_attempt001');verify(P3);verify(MATCHED);verify(POINT7)
 assert not OUT.exists(),'Use a new registered attempt rather than overwrite/auto-resume'
 verification_seconds=time.perf_counter()-verification_started
 OUT.mkdir(parents=True)
 atomic(OUT/'INPUT_WEIGHT_VERIFY_COST.json',{'seconds_before_GPUwall_budget':verification_seconds,'includes_all5_weight_SHA256_and_source_cache_seal_verification':True,'not_in2700sec_model_load_and_generation_budget':True})
 for n in ['raw_answers_PRIVATE','token_ids_PRIVATE','input_bindings_PRIVATE','calls_PRIVATE','predictions_PRIVATE']:(OUT/n).mkdir()
 binding={'plan_sha256':sha(EXP/'PLAN_LOCK.json'),'implementation_sha256':sha(EXP/'IMPLEMENTATION_LOCK.json'),'execution_authorization_sha256':sha(EXP/'EXECUTION_AUTHORIZATION.json'),'complete_download_manifest_sha256':sha(MODEL/'DOWNLOAD_MANIFEST_ATTEMPT002.json'),'source3B_seal_sha256':sha(P3/'SEALED.json'),'matched3B_seal_sha256':sha(MATCHED/'SEALED.json'),'n':208,'synthetic':0,'max_calls':208};atomic(OUT/'RUN_BINDING.json',binding)
 # Resource-only prerequisite, deliberately not an accuracy or method gate.
 mem=memory_receipt();available=int(mem['system_memory']['MemAvailable'].split()[0])*1024;atomic(OUT/'PRELOAD_RESOURCE_RECEIPT.json',mem)
 if available<plan['minimum_MemAvailable_bytes']:
  atomic(OUT/'STOPPED_RESOURCE_PREREQUISITE.json',{'state':'INSUFFICIENT_MemAvailable_NO_MODEL_LOAD','available_bytes':available,'required_bytes':plan['minimum_MemAvailable_bytes'],'not_scientific7B_negative':True});raise RuntimeError('Resource-only insufficient host memory; no precision change')
 started=start_wall_budget(2700);engine=None;ledger=OUT/'GENERATION_BUDGET.jsonl'
 try:
  startload=time.perf_counter();engine=LocalFrozenVLM(MODEL,replace(DEFAULT_CONFIG,max_new_tokens=96));t=engine.torch
  assert not engine.model.training and not any(p.requires_grad for p in engine.model.parameters())
  assert all(str(p.dtype)=='torch.bfloat16' for p in engine.model.parameters())
  atomic(OUT/'MODEL_LOAD_RECEIPT.json',{'seconds_load':time.perf_counter()-startload,'memory_includes_load_peak_before_reset':memory_receipt(t),'actual_parameter_count':sum(p.numel() for p in engine.model.parameters()),'loading_info':engine.loading_info,'model_generation_config':engine.model.generation_config.to_dict(),'same_effective_decode_as3B':True,'versions':{x:__import__('importlib.metadata',fromlist=['version']).version(x) for x in ['torch','transformers','safetensors']},'local_only_BF16_SDPA_no_remote_code':True});t.cuda.reset_peak_memory_stats()
   # Processor-only CPU reconstruction, never load a second model.
  from transformers import AutoProcessor,AutoTokenizer
  reference_dir=Path(plan['reference3B_processor_dir'])
  for name,h in plan['reference3B_processor_files_sha256'].items():assert sha(reference_dir/name)==h
  reference_processor=AutoProcessor.from_pretrained(reference_dir,local_files_only=True,trust_remote_code=False,use_fast=False,min_pixels=200704,max_pixels=802816)
  reference_processor.tokenizer=AutoTokenizer.from_pretrained(reference_dir,local_files_only=True,trust_remote_code=False,use_fast=True)
  allrows=rows();assert len(allrows)==208;pre=select_preflight(allrows,read(ROOT/'research/round37/experiment/PREFLIGHT12_PRIVATE.json'));assert [r['id'] for r in pre]==plan['preflight2_ids_PRIVATE'];order=pre+[r for r in allrows if r['id'] not in {x['id'] for x in pre}];records=[]
  for j,r in enumerate(order):
   case=prepared(r);z=arrays(r);old=read(P3/'calls_PRIVATE'/(r['id']+'_bbox.json'))['result'];phase='preflight_reused' if j<2 else 'full';result=native_call(engine,reference_processor,r,case,old,r['id']+'_bbox',phase,binding['plan_sha256'],ledger)
   assignment=iou_assignment(result,case,z);reads={NEW_ARMS[0]:bbox_readout(result,case,z,False),NEW_ARMS[1]:bbox_readout(result,case,z,True),NEW_ARMS[2]:iou_readout(assignment,case,z,False),NEW_ARMS[3]:iou_readout(assignment,case,z,True)};cached=load_npz(MATCHED/'predictions_PRIVATE'/(r['id']+'.npz'));assert cached['arm_names'].tolist()==CACHED_ARMS;point=load_npz(POINT7/'predictions_PRIVATE'/(r['id']+'.npz'));pq=dict(zip(point['arm_names'].tolist(),map(int,point['queries'])));qq=list(map(int,cached['queries']))+[pq[n] for n in POINT_ARMS]+[reads[n]['query'] for n in NEW_ARMS];q=np.array(qq,dtype=np.int64);assert len(q)==21
   atomic_npz(OUT/'predictions_PRIVATE'/(r['id']+'.npz'),arm_names=np.array(ARMS),queries=q,boxes=z['source_boxes'][q]);record={'alias':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'readouts':reads,'assignment':assignment,'call_file':str((OUT/'calls_PRIVATE'/(r['id']+'_bbox.json')).relative_to(OUT))};atomic(OUT/'predictions_PRIVATE'/(r['id']+'.json'),record);records.append(record)
   if j==1:atomic(OUT/'PREFLIGHT2_RECEIPT.json',{'state':'SOFTWARE_AND_MODEL_INPUT_CONTRACTS_PASSED','n':2,'reused_no_extra_calls':True,'ids_PRIVATE':[x['id'] for x in pre],'parsed_or_null':sum(read(OUT/x['call_file'])['parsed']['status'] in ['parsed','explicit_null'] for x in records),'invalid_box':sum(read(OUT/x['call_file'])['parsed']['status']=='invalid_box' for x in records),'parse_status_counts':{status:sum(read(OUT/x['call_file'])['parsed']['status']==status for x in records) for status in ['parsed','explicit_null','invalid_box','parse_failure']},'mapped':sum(x['readouts'][NEW_ARMS[0]]['mapped'] for x in records),'unknowns_not_technical_failure_gate':True,'no_task_GT':True})
   if (j+1)%4==0:print('R45',j+1,'seconds',round(time.perf_counter()-started,1),flush=True)
  assert engine.verify_unchanged();assert len(ledger.read_text().splitlines())==208 and len(list((OUT/'raw_answers_PRIVATE').glob('*.json')))==208
  atomic(OUT/'ALL_READOUTS_PRIVATE.json',records);atomic(OUT/'FINAL_RESOURCE_RECEIPT.json',{'GPU_wall_seconds_including_load_and_generation':time.perf_counter()-started,'memory_generate_peak_after_load_reset':memory_receipt(t),'same_exact208_inputs_as3B':True,'frozen_unchanged':True,'no_TTA_or_task_GT':True})
  seal_outputs(OUT,{'n':208,'synthetic_calls':0,'total_generation_attempts':208,'same_input_capacity_only':True,'no_TTA_or_task_GT':True,**binding});signal.setitimer(signal.ITIMER_REAL,0);print('R45_COMPLETE_SEALED_BEFORE_TASK_SCORING',flush=True)
 except BaseException as e:
  atomic(OUT/('FAILURE_'+str(time.time_ns())+'.json'),{'state':'FAILED_OR_INTERRUPTED_PRESERVED','error_type':type(e).__name__,'message':str(e),'ledger_attempts':len(ledger.read_text().splitlines()) if ledger.exists() else 0,'seconds_since_load_budget_start':time.perf_counter()-started,'no_automatic_new_calls_or_precision_resolution_changes':True,'raw_and_tokens_preserved':True,'memory':memory_receipt(engine.torch if engine else sys.modules.get('torch'))});raise
if __name__=='__main__':main()
