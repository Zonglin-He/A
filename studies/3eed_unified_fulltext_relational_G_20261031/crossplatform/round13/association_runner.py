"""R13 root-executed frozen VLM:116cases x3queries x2numberings, no GT access.
This module is import-safe; model/GPU use occurs only behind actual hashed plan.
No downloads, optimizer, 3D regression, generated-code execution or retry repair.
"""
from pathlib import Path
import importlib.metadata
import argparse,hashlib,json,os,sys,time,traceback
import numpy as np
from association import (AssociationConfig,DEFAULT_CONFIG,asdict,array_sha,map_masks_to_query_groups,numbered_candidates,render_numbered_panel,build_messages,parse_choice,readout_choice,consensus_readout,first_noun_text)
from input_io import B,read,sha,verify_seal,load_case
from camera_projection import raw_to_model_points
QUERIES=('full','noun','deranged');ARMS=('source','full_order0_raw','full_two_numbering_consensus','noun_two_numbering_consensus','deranged_two_numbering_consensus')

def serial(x):
 if isinstance(x,dict):return {str(k):serial(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [serial(v) for v in x]
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 return x

def atomic_json(path,obj):
 path=Path(path);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(serial(obj),indent=2,allow_nan=False)+'\n');os.replace(tmp,path)

def model_snap(model):
 import torch
 return {'training':model.training,'params':[(n,tuple(p.shape),str(p.dtype),p._version,p.requires_grad) for n,p in model.named_parameters()],'buffers':[(n,tuple(p.shape),str(p.dtype),p._version,array_sha(p.detach().cpu().contiguous().view(torch.uint8).numpy())) for n,p in model.named_buffers()]}

class FrozenVLM:
 def __init__(self,model_dir,config):
  import torch
  from transformers import AutoProcessor,AutoTokenizer,Qwen2_5_VLForConditionalGeneration
  self.torch=torch;self.config=config;torch.manual_seed(config.seed)
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
  if not torch.cuda.is_available():raise RuntimeError('Root GPU release requires actual CUDA; no silent CPU/remote fallback')
  self.processor=AutoProcessor.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False,use_fast=False,min_pixels=config.min_pixels,max_pixels=config.max_pixels)
  self.processor.tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False,use_fast=True)
  self.model,info=Qwen2_5_VLForConditionalGeneration.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False,use_safetensors=True,torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa',output_loading_info=True)
  if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):raise RuntimeError('Non-strict model load: '+json.dumps(info))
  self.loading_info=info;self.model.eval().requires_grad_(False);self.snap=model_snap(self.model);assert not self.snap['training'] and not any(p[-1] for p in self.snap['params']);self.torch.cuda.synchronize()
 def predict(self,images,query,candidates):
  t=self.torch;text=self.processor.apply_chat_template(build_messages(query,[c['candidate_id'] for c in candidates]),tokenize=False,add_generation_prompt=True)
  inputs=self.processor(text=[text],images=images,return_tensors='pt',padding=False,truncation=False).to('cuda')
  if inputs['input_ids'].shape[1]>8192:raise RuntimeError('Full prompt exceeds locked8192token engineering cap; never silently truncate')
  grid=inputs['image_grid_thw'].tolist();assert len(grid)==2 and all(g[0]==1 for g in grid)
  assert all(g[1]*g[2]//4<=1024 for g in grid),'visualtoken cap'
  start=time.perf_counter();t.cuda.synchronize()
  with t.inference_mode():out=self.model.generate(**inputs,do_sample=False,max_new_tokens=self.config.max_new_tokens,use_cache=True)
  t.cuda.synchronize();tail=out[0,inputs['input_ids'].shape[1]:];raw=self.processor.tokenizer.decode(tail,skip_special_tokens=True,clean_up_tokenization_spaces=False);eos=self.model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
  return {'raw_answer_PRIVATE':raw,'parsed':parse_choice(raw,candidates),'input_token_n':int(inputs['input_ids'].shape[1]),'image_grid_thw':grid,'image_visualtoken_n':[g[0]*g[1]*g[2]//4 for g in grid],'generated_token_n':len(tail),'ended_with_eos':any(int(x) in eos for x in tail),'max_new_tokens':self.config.max_new_tokens,'query_sha256':hashlib.sha256(query.encode()).hexdigest(),'serialized_prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),'seconds_model_generate':time.perf_counter()-start}
 def verify_unchanged(self):
  after=model_snap(self.model)
  if after!=self.snap:raise RuntimeError('Frozen VLM parameter/buffer/mode state changed')
  return True

def main(a):
 started=time.perf_counter();assert sha(a.plan)==a.plan_sha256;plan=read(a.plan)
 assert plan['stage']=='R13_frozen_full_description_association' and plan['GT_read_permitted'] is False and plan['model_execution_permitted'] and plan['GPU_execution_permitted'] and plan['source_and_mask_caches_sealed']
 assert plan['expected_samples']==116 and plan['formal_cases_complete_all3x2_before_GT'] and plan['formal_preflight_policy']=='rerun_all116_in_separate_output'
 assert plan['model_dtype']=='bfloat16' and plan['attn_implementation']=='sdpa' and plan['association_config']==asdict(DEFAULT_CONFIG)
 for rel,h in plan['program_sha256'].items():assert sha(B/rel)==h,rel
 assert plan['program_sha256']['crossplatform/round13/association_runner.py']==sha(__file__)
 data={}
 for k in ('manifest','case_index','frame_index'):
  path=Path(plan[k]);assert sha(path)==plan[k+'_sha256'];data[k]=read(path)
 rows=data['manifest'];cases={r['id']:r for r in data['case_index']};index=data['frame_index'];assert len(rows)==len(cases)==116 and len({r['id'] for r in rows})==116
 inp={r['id']:r for r in rows}
 for row in rows:
  c=cases[row['id']];donor=inp[c['shuffled_donor_id_PRIVATE']];assert donor['id']!=row['id'] and donor['scene']==row['scene'] and donor['platform']==row['platform'] and donor['caption']==c['shuffled_caption']
 for rel,h in plan['cache_seals_sha256'].items():assert sha(B/rel)==h;verify_seal((B/rel).parent)
 modeldir=Path(plan['model_dir']);assert read(modeldir/'config.json')['model_type']=='qwen2_5_vl';mi=read(modeldir/'model.safetensors.index.json');required=set(mi['weight_map'].values())|{'config.json','model.safetensors.index.json','preprocessor_config.json','tokenizer_config.json','tokenizer.json','vocab.json','merges.txt','chat_template.json','generation_config.json'}
 assert required<=set(plan['model_files_sha256']) and all(sha(modeldir/n)==h for n,h in plan['model_files_sha256'].items()),'all smallfiles and safetensors actualhash required before model load'
 limit=a.limit
 if limit is not None:assert limit==2 and plan['allow_engineering_limit2'];rows=rows[:2]
 scope='engineering_inputonly_first2_12calls_not_formal116' if limit else 'formal116_rerun_separate_output_696calls'
 out=Path(a.output);binding={'state':'R13_association_run_bindings','scope':scope,'plan_sha256':sha(a.plan),'program_sha256':plan['program_sha256'],'manifest_sha256':plan['manifest_sha256'],'model_files_sha256':plan['model_files_sha256'],'queries':list(QUERIES),'two_numberings':2,'n_cases':len(rows)}
 if out.exists():
  assert a.resume and not (out/'SEALED.json').exists() and read(out/'RUN_BINDINGS.json')==binding,'Only samebound unfinished run can resume'
 else:out.mkdir(parents=True);atomic_json(out/'RUN_BINDINGS.json',binding)
 callroot=out/'calls';callroot.mkdir(exist_ok=True);trroot=out/'cases';trroot.mkdir(exist_ok=True);journal=out/'calls.jsonl';engine=None;total_existing_calls=0
 try:
  loadstart=time.perf_counter();engine=FrozenVLM(modeldir,DEFAULT_CONFIG);loadseconds=time.perf_counter()-loadstart;peak0=int(engine.torch.cuda.max_memory_allocated());atomic_json(out/'MODEL_LOAD_RECEIPT.json',{'loading_info':engine.loading_info,'seconds':loadseconds,'allocated_bytes_after_load':int(engine.torch.cuda.memory_allocated()),'peak_allocated_bytes_after_load':peak0,'modelweights_files_rehashed':True,'local_only_safetensors_no_remote_code':True,'frozen_eval_no_updates':True,'formal_scope':scope})
  for row in rows:
   case=cases[row['id']];alias=case['case_alias'];frame,c,bindings=load_case(row,case,index);boxes=c['boxes_raw6'][0];scores=c['text_scores'][0];q0=int(c['query_index']);modelp=np.array(frame['raw_points'],np.float32,copy=True)
   if row['platform']=='drone':modelp[:,2]+=np.float32(1.8)
   elif row['platform']!='quad':raise ValueError('R13 targetplatform contract')
   mapping=map_masks_to_query_groups(boxes_raw6=boxes,text_scores=scores,q0=q0,points_model=modelp,camera_spec=frame['camera'],masks_bool=frame['instances']['masks_bool'],mask_boxes_xyxy=frame['instances']['boxes_xyxy'])
   candidates=[numbered_candidates(row['id'],frame['instances']['masks_bool'],frame['instances']['boxes_xyxy'],permutation=p) for p in range(2)]
   from PIL import Image
   original=Image.fromarray(frame['rgb']);panels=[render_numbered_panel(frame['rgb'],cand) for cand in candidates];noun,nounreceipt=first_noun_text(row['caption']);queries={'full':row['caption'],'noun':noun,'deranged':case['shuffled_caption']};calls={}
   paneldir=out/'panels_PRIVATE';paneldir.mkdir(exist_ok=True)
   for k,panel in enumerate(panels):
    panelpath=paneldir/(alias+'__order'+str(k)+'.png')
    if not panelpath.exists():panel.save(panelpath)
   for cond in QUERIES:
    calls[cond]=[]
    for order in range(2):
     cp=callroot/(alias+'__'+cond+'__order'+str(order)+'.json');cbinding={'case_alias':alias,'case_id_PRIVATE':row['id'],'condition':cond,'numbering':order,'plan_sha256':sha(a.plan),'bindings':bindings,'original_RGB_array_sha256':array_sha(frame['rgb']),'panel_RGB_array_sha256':array_sha(np.asarray(panels[order])),'query_sha256':hashlib.sha256(queries[cond].encode()).hexdigest(),'candidate_id_to_mask_index':{c['candidate_id']:c['mask_index'] for c in candidates[order]}}
     if cp.exists():record=read(cp);assert record['binding']==cbinding;total_existing_calls+=1
     else:
      result=engine.predict([original,panels[order]],queries[cond],candidates[order]);record={'state':'complete_frozen_VLM_single_call_no_GT_or_update','binding':cbinding,'result':result};atomic_json(cp,record)
      with journal.open('a') as f:f.write(json.dumps({'path':str(cp.relative_to(out)),'sha256':sha(cp)})+'\n');f.flush();os.fsync(f.fileno())
     calls[cond].append(record['result'])
   results={cond:consensus_readout([x['parsed'] for x in calls[cond]],candidates,mapping,q0,boxes) for cond in QUERIES};raw0=readout_choice(calls['full'][0]['parsed'],candidates[0],mapping,q0,boxes)
   armq=np.asarray([q0,raw0['q_selected'],results['full']['q_selected'],results['noun']['q_selected'],results['deranged']['q_selected']],np.int64);armboxes=boxes[armq].copy();assert np.array_equal(armboxes,boxes[armq])
   npz=trroot/(alias+'.npz');tmp=npz.with_name(npz.name+'.tmp')
   with tmp.open('wb') as f:np.savez_compressed(f,sample_id=np.asarray(row['id']),case_alias=np.asarray(alias),role=np.asarray(case['role']),arm_names=np.asarray(ARMS),arm_query_index=armq,arm_boxes_raw6=armboxes,source_q0=np.asarray(q0),source_boxes_raw6=boxes,source_text_scores256=scores)
   os.replace(tmp,npz);atomic_json(trroot/(alias+'.json'),{'state':'complete6calls_and_originalquery_readout_no_GT','id_PRIVATE':row['id'],'case_alias':alias,'role':case['role'],'platform':row['platform'],'scene_PRIVATE':row['scene'],'bindings':bindings,'q0':q0,'noun_control':nounreceipt,'deranged_same_caption_as_full':queries['deranged']==queries['full'],'mapping':mapping,'candidate_panels':candidates,'full_order0_raw':raw0,'consensus':results,'calls':calls,'no_3Dbox_regression_or_parameter_updates':True})
   if sum((trroot/(cases[r['id']]['case_alias']+'.json')).exists() for r in rows)%5==0 or limit:print(json.dumps({'event':'association_progress','case_alias':alias,'completed_cases':sum((trroot/(cases[r['id']]['case_alias']+'.json')).exists() for r in rows),'expected_cases':len(rows)}),flush=True)
   atomic_json(out/'PROGRESS.json',{'state':'unfinished_no_GTrelease','scope':scope,'completed_cases':sum((trroot/(cases[r['id']]['case_alias']+'.json')).exists() for r in rows),'expected_cases':len(rows),'expected_calls':len(rows)*6,'last_case_alias':alias})
  assert engine.verify_unchanged();files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ('SEALED.json',) and not p.name.endswith('.tmp')}
  assert len(list(callroot.glob('*.json')))==len(rows)*6 and len(list(trroot.glob('*.npz')))==len(rows)
  seal={'state':'sealed_R13_frozen_association_first2_engineering_no_GT' if limit else 'sealed_R13_all116_frozen_association6calls_before_GT','n_samples':len(rows),'n_queries':3,'numberings':2,'n_model_calls':len(rows)*6,'arm_names':list(ARMS),'plan_sha256':sha(a.plan),'manifest_sha256':plan['manifest_sha256'],'case_index_sha256':plan['case_index_sha256'],'frame_index_sha256':plan['frame_index_sha256'],'program_sha256':plan['program_sha256'],'cache_seals_sha256':plan['cache_seals_sha256'],'model_files_sha256':plan['model_files_sha256'],'model_frozen_parameters_buffers_unchanged':True,'GT_or_optimizer_or_neural_update_executed':False,'only_original_source_query_boxes_selected':True,'formal_preflight_policy':plan['formal_preflight_policy'],'scope':scope,'resumed_existing_call_n':total_existing_calls,'seconds_this_process':time.perf_counter()-started,'peak_allocated_bytes':int(engine.torch.cuda.max_memory_allocated()),'runtime_packages':{n:importlib.metadata.version(n) for n in ('torch','transformers','numpy','Pillow','safetensors')},'files':files};atomic_json(out/'SEALED.json',seal);print(json.dumps({'state':seal['state'],'n':len(rows),'calls':len(rows)*6,'seal_sha256':sha(out/'SEALED.json')}))
 except Exception as error:
  failure={'state':'failed_R13_association_attempt_preserved_no_GT','scope':scope,'type':type(error).__name__,'message':str(error),'traceback_PRIVATE':traceback.format_exc(),'plan_sha256':sha(a.plan),'no_failure_repair_or_retry_performed':True,'seconds':time.perf_counter()-started};stamp=time.time_ns();atomic_json(out/('FAILURE_'+str(stamp)+'.json'),failure);raise
if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--plan-sha256',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--limit',type=int);ap.add_argument('--resume',action='store_true');main(ap.parse_args())
