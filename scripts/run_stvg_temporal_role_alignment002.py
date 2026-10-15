"""Frozen R1 role evidence and original cached512 temporal adaptation, no GT."""
from pathlib import Path
import sys,copy,time,gc
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_temporal_role_alignment002 import *
from scripts.stvg_soft_ep_coordinate_common_v1 import save_pt

def teacher(phase):
 import torch
 from transformers import AutoProcessor
 from scripts import run_stvg_native_instance_teacher_storage002 as t
 from scripts import stvg_table1_temporal_adapter_v1 as ta
 check_runtime();sys.addaudithook(guard);assert phase in ['P1','P2']
 if phase=='P2':assert read(BASE/'P1/CPU_COMPLETION.json')['p2_eligible'] is True
 t.verify_local_model_receipt();t.MAX_MEMORY={0:'20GiB','cpu':'4GiB'}
 _,model,torch,device,mapping,load_s,peak=t._load_qwen(ROOT/t.CHECKPOINT_REL,BASE/phase/'offload');model.generation_config.use_cache=True
 processor=AutoProcessor.from_pretrained(ROOT/t.CHECKPOINT_REL,local_files_only=True,trust_remote_code=False);ta.configure_original_processor(processor);assert not model.training and not any(p.requires_grad for p in model.parameters())
 jobs={j['job_id']:j for j in read(BASE/phase/'TEACHER_JOB_LOCK.json')['records']};files=[];calls=caption_calls=0;start=time.perf_counter()
 for rec in [r for r in roster()if is_question(r)]:
  cid=rec['probe_alias'];case,cp=load_case(rec);old=p0_responses(rec);q=query(rec);new=[];clips=None
  if phase=='P2'and old:
   from scripts import run_stvg_table1_native_instance_adapter_v1 as nat
   from scripts import run_stvg_table1_teacher_v1 as oldt
   request=nat.canonicalize(case,q,BASE/'P2/canonical_cache');write(BASE/'P2/requests'/(cid+'.json'),request)
   eligible,reason=oldt.temporal_clip_eligibility(case,request);assert eligible,reason
   rgb=oldt.rgb_for_candidates(case,request,case['temporal_candidates']);clips=ta.prepare_candidate_clips(case,case['temporal_candidates'],decoded_rgb_by_physical_id=rgb,frame_map=request['frame_map']);del rgb
   for candidate,prior in zip(case['temporal_candidates'],old):
    meta=clips[candidate['canonical_id']]['metadata'];assert meta['marked_RGB_sha256']==prior['metadata']['marked_RGB_sha256']and meta['sampled_physical_frame_ids']==prior['metadata']['sampled_physical_frame_ids']
  for i,candidate in enumerate(case['temporal_candidates']):
   jid=cid+'_C'+str(i);job=jobs.get(jid)
   if job is None:new.append(None);continue
   prior=old[i];assert job['query_sha256']==digest(q)and job['caption_sha256']==digest(prior['raw_caption'])and job['source_case_sha256']==case['source_case_sha256']
   before=time.perf_counter()
   if phase=='P1':
    prompt=alignment_prompt(q,prior['raw_caption']);assert digest(prompt)==job['alignment_text_sha256'];batch,chatsha=ta._processor_batch(processor,prompt,device=device);label_ids=ta._abc_token_ids(processor)
    with torch.inference_mode():
     output=model(**batch,use_cache=False);logits=output.logits[0,-1,label_ids].float();probs=torch.softmax(logits,dim=0)
    row=dict(candidate_id=candidate['canonical_id'],caption_sha256=prior['caption_sha256'],query_sha256=digest(q),caption_status=prior['caption_status'],output_truncated_at_cap=prior['output_truncated_at_cap'],label_logits=logits.detach().cpu().tolist(),probabilities=probs.detach().cpu().tolist(),reward=float(probs[0]-probs[1]),input_tokens=int(batch['input_ids'].shape[1]),alignment_text_sha256=digest(prompt),alignment_prompt_sha256=chatsha,metadata=copy.deepcopy(prior['metadata']),GT_read=False)
    del batch,output,logits,probs
   else:
    previous_caption,previous_align=ta.caption_prompt,ta.alignment_prompt
    try:
     ta.caption_prompt=lambda:RELATION_CAPTION;ta.alignment_prompt=alignment_prompt
     row=ta.run_hot_candidate_teacher(model,processor,q,candidate,clips[candidate['canonical_id']],device=device)
     row['alignment_text_sha256']=digest(alignment_prompt(q,row['raw_caption']));caption_calls+=1
    finally:ta.caption_prompt,ta.alignment_prompt=previous_caption,previous_align
   row.update(job_id=jid,source_case_sha256=case['source_case_sha256'],P0_teacher_sha256=job['teacher_sha256'],P0_caption_sha256=job['caption_sha256'],wall_seconds=time.perf_counter()-before,teacher_weight_update=False)
   new.append(row);calls+=1
  files.append(save_pt(BASE/phase/'teacher'/(cid+'.pt'),dict(case_id=cid,dataset=rec['dataset'],phase=phase,slots=new,source_case_sha256=case['source_case_sha256'],source_file_sha256=sha(cp),runtime_sha256=sha(BASE/'RUNTIME.json'),GT_read=False)))
  write(BASE/phase/'STATUS.json',dict(status='running',stage='teacher',done=len(files),total=28,GT_read=False,time=time.time()),mutable=True);print(phase,cid,len(files),flush=True);del case,old,new,clips;gc.collect();torch.cuda.empty_cache()
 assert len(files)==28 and calls==88 and caption_calls==(88 if phase=='P2'else 0)
 write(BASE/phase/'TEACHER_GLOBAL_BARRIER.json',dict(status='complete',count=28,files=files,alignment_calls=calls,caption_calls=caption_calls,model_load_seconds=load_s,wall_seconds=time.perf_counter()-start,runtime_sha256=sha(BASE/'RUNTIME.json'),GT_read=False))

def fit(phase):
 import torch,numpy as np
 from scripts import stvg_table1_temporal_adapter_v1 as ta
 from scripts.stvg_evidence_temporal_fit_v1 import fit as temporal_fit
 check_runtime();sys.addaudithook(guard);torch.set_num_threads(2);bar=read(BASE/phase/'TEACHER_GLOBAL_BARRIER.json');assert bar['status']=='complete'and bar['count']==28 and not bar['GT_read'];files_by_id={Path(x['path']).stem:x for x in bar['files']};files=[];actual=0
 for rec in vid_roster():
  cid=rec['probe_alias'];case,cp=load_case(rec);p0=p0_interval(rec);f=None;rewards=None;iv=p0
  if is_question(rec):
   item=files_by_id[cid];p=ROOT/item['path'];assert sha(p)==item['sha256'];teacher=torch.load(p,map_location='cpu',weights_only=False);assert teacher['source_case_sha256']==case['source_case_sha256']and not teacher['GT_read']
   rewards=[ta._alignment_reward(c['canonical_id'],r)if r is not None else None for c,r in zip(case['temporal_candidates'],teacher['slots'])]
   if any(x is not None for x in rewards):f=temporal_fit(case,ta._hidden_from_source_case(case),case['temporal_candidates'],rewards,variant='offset_trust');iv=f['final_interval'];actual+=1;assert f['active_parameters']==512 and f['GT_read']is False
  boxes=np.asarray(case['native_boxes_xyxy_normalized']);arms={'F':dict(boxes=boxes,interval=case['Native_interval']),'T':dict(boxes=boxes,interval=p0),phase:dict(boxes=boxes,interval=iv)}
  files.append(save_pt(BASE/phase/'student'/(cid+'.pt'),dict(case_id=cid,dataset='vidstg',arms=arms,fit=f,rewards=rewards,question=is_question(rec),frame_ids=case['frame_ids'],source_case_sha256=case['source_case_sha256'],source_file_sha256=sha(cp),runtime_sha256=sha(BASE/'RUNTIME.json'),source_restored=True,new_full_model_forward=False,GT_read=False)))
 assert len(files)==64
 write(BASE/phase/'STUDENT_GLOBAL_BARRIER.json',dict(status='complete',count=64,files=files,actual_CPU_temporal_fits=actual,new_full_Student_GPU_forwards=0,HC_unchanged_source_count=64,runtime_sha256=sha(BASE/'RUNTIME.json'),GT_read=False))
 print(phase,'sealed64',actual,'CPU temporal fits')
if __name__=='__main__':globals()[sys.argv[1]](sys.argv[2])
