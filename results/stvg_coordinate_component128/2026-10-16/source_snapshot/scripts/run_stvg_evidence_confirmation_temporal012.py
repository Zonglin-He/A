"""Fresh-cohort frozen caption/query feedback; source checkpoints unchanged, no GT."""
import sys,os,time,json,hashlib,gc,copy,collections,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_evidence_common_v1 import *
from scripts.stvg_evidence_fresh_custody012 import checked_load_case as load_case
def temporal():
 import torch,numpy as np
 from transformers import AutoProcessor
 from scripts import run_stvg_native_instance_teacher_storage002 as t
 from scripts import stvg_table1_temporal_adapter_v1 as ta
 from scripts import run_stvg_table1_source_v1 as src
 from scripts import run_stvg_table1_teacher_v1 as oldt
 from scripts import run_stvg_table1_native_instance_adapter_v1 as nat
 t.verify_local_model_receipt();t.MAX_MEMORY={0:'20GiB','cpu':'4GiB'}
 _,m,torch,device,mapping,load_s,peak=t._load_qwen(ROOT/t.CHECKPOINT_REL,BASE/'confirmation/offload_temporal');m.generation_config.use_cache=True
 assert read(BASE/'FROZEN_METHOD_LOCK.json')['status']=='locked'
 p=AutoProcessor.from_pretrained(ROOT/t.CHECKPOINT_REL,local_files_only=True,trust_remote_code=False);files=[];calls=0;start=time.perf_counter()
 for rec in roster('confirmation'):
  cid=rec['probe_alias'];c,sourcepath=load_case(rec);_,row=src._validate_roster_record(rec);query=row['input']['caption'];rp=BASE/'confirmation/requests'/(cid+'.json');ts=time.perf_counter()
  if rp.exists():
   request=read(rp);assert request['source_video_sha256']==c['input']['video_sha256'] and request['query_text']==query and sha(Path(request['video_path']))==request['canonical_video_sha256'];reused=True
  else:request=nat.canonicalize(c,query,BASE/'confirmation/canonical_cache');reused=False
  write(BASE/'confirmation/requests'/(cid+'.json'),request)
  eligible,reason=oldt.temporal_clip_eligibility(c,request)
  if eligible:
   rgb=oldt.rgb_for_candidates(c,request,c['temporal_candidates']);value=ta.adapt_hot_query(c,query,c['temporal_candidates'],model=m,processor=p,device=device,decoded_rgb_by_physical_id=rgb,frame_map=request['frame_map']);calls+=8;del rgb
  else:value=dict(status='unavailable_temporal_input_native_fallback',reason=reason,final_interval=c['Native_interval'],GT_read=False,optimizer_steps=0)
  d=dict(case_id=cid,dataset=c['dataset'],temporal=value,source_file_sha256=sha(sourcepath),request_sha256=sha(BASE/'confirmation/requests'/(cid+'.json')),canonical_reused=reused,wall_seconds=time.perf_counter()-ts,GT_read=False)
  files.append(save_pt(BASE/'confirmation/temporal_teacher'/(cid+'.pt'),d));write(BASE/'confirmation/TEMPORAL_STATUS.json',dict(status='running',done=len(files),total=32,GT_read=False),mutable=True);print('temporal',cid,len(files),flush=True);del c,value,d;gc.collect();torch.cuda.empty_cache()
 write(BASE/'confirmation/TEMPORAL_TEACHER_GLOBAL_BARRIER.json',dict(status='complete',count=32,files=files,teacher_calls=calls,model_load_seconds=load_s,wall_seconds=time.perf_counter()-start,GT_read=False))
if __name__=="__main__":
 verify_runtime();sys.addaudithook(guard);temporal()
