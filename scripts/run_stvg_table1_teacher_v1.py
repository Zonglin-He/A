"""Frozen hot R1 temporal+identity inference for the fixed engineering roster.

All outputs are private; annotations are blocked. One model load serves all
queries. Existing sealed queries are verified and never counted as new calls.
"""
from pathlib import Path
import os,sys,json,time,hashlib,argparse,gc
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
BASE=ROOT/'artifacts/stvg_final_table1_v1'
from scripts.stvg_native_instance_common_v1 import read,sha,write,guard
from scripts import stvg_table1_temporal_adapter_v1 as temporal
from scripts import run_stvg_table1_native_instance_adapter_v1 as native

def verify():
 rt=read(BASE/'TEACHER_RUNTIME.json');assert rt['status']=='locked' and rt['GT_read'] is False
 for p,h in {**rt['pins'],**rt['protected']}.items():assert sha(ROOT/p)==h,p
 assert read(BASE/'R1_EXECUTION_LOCK.json')['use_cache'] is True
 return rt

def rgb_for_candidates(case,request,candidates):
 import cv2,numpy as np
 frame_map=request['frame_map'];available=sorted(int(x['physical_frame_id'])for x in frame_map)
 wanted=set(f for c in candidates for f in temporal._sample_candidate_ids(c,available)[0])
 local_to_physical={int(x['local_frame_index']):int(x['physical_frame_id'])for x in frame_map}
 p=Path(request['video_path']);assert sha(p)==request['canonical_video_sha256']
 cap=cv2.VideoCapture(str(p));rgb={};local=0
 try:
  while wanted:
   ok,bgr=cap.read()
   if not ok:raise RuntimeError('canonical clip ended before fixed temporal samples')
   physical=local_to_physical.get(local)
   if physical in wanted:
    rgb[physical]=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);wanted.remove(physical)
   local+=1
 finally:cap.release()
 return rgb

def temporal_clip_eligibility(case,request):
 # The source edge record uses declared video_bounds. Recheck the same fixed
 # eight-frame samples against the actually available canonical REQUEST clock
 # before opening media; a truncated canonical tail must fall back, not abort.
 candidates=case.get('temporal_candidates',[])
 roles=case.get('candidate_validity',{}).get('candidate_roles',[])
 if len(candidates)!=4 or len(roles)!=4:
  return False,'candidate_pool_or_role_count_not_four'
 if not all(row.get('clip_request_eligible') is True for row in roles):
  return False,'source_declared_clock_marks_a_clip_unavailable'
 role_by_id={}
 for role in roles:
  cid=role.get('candidate_id')
  if not isinstance(cid,str) or not cid or cid in role_by_id:
   return False,'duplicate_or_missing_candidate_role_identity'
  role_by_id[cid]=role
 candidate_ids=[temporal._candidate_id(row) for row in candidates]
 if len(set(candidate_ids))!=4 or set(candidate_ids)!=set(role_by_id):
  return False,'candidate_and_role_identity_mismatch'
 try:
  normalized=native._native()._normalize_request(request)
  physical_to_local=temporal._frame_map_rows(normalized['frame_map'])
  available=sorted(physical_to_local)
 except (KeyError,TypeError,ValueError) as exc:
  return False,'canonical_request_clock_not_usable_for_original_temporal_adapter:'+type(exc).__name__
 for candidate in candidates:
  cid=temporal._candidate_id(candidate)
  role=role_by_id[cid]
  try:
   sampled,_=temporal._sample_candidate_ids(candidate,available)
  except (KeyError,TypeError,ValueError,IndexError):
   return False,'canonical_request_has_fewer_than_eight_candidate_frames:'+cid
  locked=role.get('clip',{}).get('sampled_physical_frame_ids')
  if not isinstance(locked,list) or list(map(int,locked))!=sampled:
   return False,'canonical_request_samples_differ_from_locked_source_clip:'+cid
 return True,None

def save_tensor(path,value,torch):
 path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():raise FileExistsError(path)
 tmp=path.with_suffix('.tmp');torch.save(value,tmp);os.replace(tmp,path)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--case-alias');args=ap.parse_args()
 rt=verify();sys.addaudithook(guard)
 import torch,numpy as np
 from transformers import AutoProcessor
 from scripts import run_stvg_native_instance_teacher_storage002 as teacher
 from scripts import run_stvg_table1_source_v1 as source
 from scripts import stvg_table1_spatial_adapter_v1 as spatial
 os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
 roster=read(BASE/'PREFLIGHT_ROSTER_LOCK.json')['records'];rows=[r for r in roster if args.case_alias is None or r['probe_alias']==args.case_alias]
 assert len(rows)==(64 if args.case_alias is None else 1)
 assert read(BASE/'SOURCE_CAPTURE_BARRIER.json')['status']=='sealed'
 teacher.verify_local_model_receipt();teacher.MAX_MEMORY={0:'22GiB','cpu':'4GiB'}
 who_processor,model,torch,device,mapping,load_s,peak=teacher._load_qwen(ROOT/teacher.CHECKPOINT_REL,BASE/'teacher_offload')
 temporal_processor=AutoProcessor.from_pretrained(str(ROOT/teacher.CHECKPOINT_REL),local_files_only=True,trust_remote_code=False)
 model.generation_config.use_cache=True
 out=BASE/'teacher_cases';out.mkdir(exist_ok=True);accepted=[];calls=0;started=time.perf_counter()
 for rec in rows:
  alias=rec['probe_alias'];dst=out/(alias+'.pt');receipt=out/(alias+'.receipt.json')
  src=BASE/'source_cases'/(alias+'.pt');src_receipt=read(src.with_suffix('.receipt.json'))
  assert sha(src)==src_receipt['sha256']
  binding=read(BASE/'tracking_cases'/alias/'TRACKING_BINDING.json')
  request_path=BASE/'tracking_cases'/alias/'REQUEST.json';assert sha(request_path)==binding['request_sha256']
  request=read(request_path);track={'accepted':read(ROOT/binding['accepted_path']),'mask_inventory':read(ROOT/binding['inventory_path'])}
  assert sha(ROOT/binding['accepted_path'])==binding['accepted_sha256'] and sha(ROOT/binding['inventory_path'])==binding['inventory_sha256']
  if dst.exists()or receipt.exists():
   assert dst.exists()and receipt.exists();rr=read(receipt)
   assert rr['sha256']==sha(dst)and rr['source_sha256']==sha(src)and rr['request_sha256']==sha(request_path)and rr['runtime_sha256']==sha(BASE/'TEACHER_RUNTIME.json') and rr['tracking_binding_sha256']==sha(BASE/'tracking_cases'/alias/'TRACKING_BINDING.json')
   accepted.append(dict(alias=alias,path=str(dst.relative_to(ROOT)),sha256=sha(dst),reused=True));continue
  torch.manual_seed(20261015);torch.cuda.manual_seed_all(20261015);np.random.seed(20261015)
  case=torch.load(src,map_location='cpu',weights_only=False);assert source.content_sha256(case)==case['source_case_sha256']
  assert binding['case_alias']==alias and binding['dataset']==rec['dataset'] and binding['query_key_sha256']==rec['query_key_sha256'] and binding['source_case_sha256']==case['source_case_sha256']
  _, official_row=source._validate_roster_record(rec)
  assert request['query_text']==official_row['input']['caption'] and request['source_video_sha256']==case['input']['video_sha256']
  query=request['query_text'];candidates=case['temporal_candidates'];t=time.perf_counter()
  # Fixed unavailable-input rule is resolved before teacher calls, never using labels.
  valid,skip_reason=temporal_clip_eligibility(case,request)
  if valid:
   rgb=rgb_for_candidates(case,request,candidates)
   result=temporal.adapt_hot_query(case,query,candidates,model=model,processor=temporal_processor,device=device,decoded_rgb_by_physical_id=rgb,frame_map=request['frame_map']);del rgb
   calls+=8
  else:result=dict(status='unavailable_temporal_input_native_fallback',reason=case['candidate_validity'],request_clock_fallback_reason=skip_reason,final_interval=case['Native_interval'],GT_read=False,optimizer_steps=0)
  who=native.infer_who(request,track,who_processor,model,torch,device);calls+=1
  selected=who['who'].get('candidate_id') if who['who']['status']=='valid' else None
  ref=spatial.build_reference(case,result['final_interval'],request,selected,track['mask_inventory'])
  value=dict(case_id=alias,dataset=case['dataset'],source_sha256=sha(src),request_sha256=sha(request_path),tracking_binding_sha256=sha(BASE/'tracking_cases'/alias/'TRACKING_BINDING.json'),temporal=result,who=who,reference=ref,GT_read=False,query_reset=True,wall_seconds=time.perf_counter()-t)
  save_tensor(dst,value,torch)
  rr=dict(status='sealed',sha256=sha(dst),bytes=dst.stat().st_size,source_sha256=sha(src),request_sha256=sha(request_path),runtime_sha256=sha(BASE/'TEACHER_RUNTIME.json'),tracking_binding_sha256=sha(BASE/'tracking_cases'/alias/'TRACKING_BINDING.json'),GT_read=False,time=time.time())
  write(receipt,rr);accepted.append(dict(alias=alias,path=str(dst.relative_to(ROOT)),sha256=rr['sha256'],reused=False))
  write(BASE/'TEACHER_STATUS.json',dict(status='running',done=len(accepted),total=len(rows),GT_read=False,time=time.time()),mutable=True)
  print('TABLE1_TEACHER',len(accepted),len(rows),alias,flush=True);del case,value,result,who,ref;gc.collect();torch.cuda.empty_cache()
 write(BASE/('TEACHER_GLOBAL_BARRIER.json'if args.case_alias is None else 'TEACHER_'+args.case_alias+'_BARRIER.json'),dict(status='sealed',count=len(accepted),files=accepted,actual_model_loads=1,actual_teacher_calls=calls,model_load_s=load_s,actual_device_map=mapping,wall_seconds=time.perf_counter()-started,GT_read=False))
if __name__=='__main__':main()
