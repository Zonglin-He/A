"""Hot Source capture for only 64 added parents; exact 64 old captures reused."""
from pathlib import Path
import sys,collections,gc,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_soft_ep_coordinate_common_v1 import *
def main():
 import torch
 from scripts import run_stvg_table1_source_v1 as source
 from scripts.run_decota_paper_main_v1 import gpu,model_for
 from methods.decota_final_simplified_v1.tensors import state_hash
 verify_runtime();sys.addaudithook(guard);gpu();source.RUNTIME=BASE/'SOURCE_RUNTIME.json';torch.set_num_threads(2)
 files=[];model=None;name=None;new=reused=0;start=time.perf_counter()
 for rec in roster():
  p=ROOT/rec['source_case_path'];receipt=p.with_suffix('.receipt.json')
  if p.exists():
   c,actual=load_case(rec);assert actual==p;reused+=int('cached_parent'in rec)
  else:
   assert 'cached_parent'not in rec and not receipt.exists()
   target=source.SOURCE_MODEL_NAME[rec['dataset']]
   if target!=name:
    if model is not None:del model;gc.collect();torch.cuda.empty_cache()
    model=model_for(target);name=target
   c=source.capture_one(model,rec,collections.OrderedDict());assert c['case_id']==rec['probe_alias'] and source.content_sha256(c)==c['source_case_sha256'] and not c['GT_read'] and state_hash(model.state_dict())==c['source_model_state_sha256']
   item=save_pt(p,c);write(receipt,dict(status='sealed',case_id=rec['probe_alias'],sha256=item['sha256'],bytes=item['bytes'],source_case_sha256=c['source_case_sha256'],source_runtime_sha256=sha(BASE/'SOURCE_RUNTIME.json'),GT_read=False));new+=1
  files.append(dict(case_id=rec['probe_alias'],path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size,source_case_sha256=c['source_case_sha256']))
  write(BASE/'SOURCE_STATUS.json',dict(status='running',done=len(files),total=128,actual_new_capture_count=new,exact_reused_count=reused,GT_read=False),mutable=True)
  print('expanded source',rec['probe_alias'],len(files),new,reused,flush=True);del c;gc.collect();torch.cuda.empty_cache()
 write(BASE/'SOURCE_CAPTURE_BARRIER.json',dict(status='complete',count=128,files=files,actual_new_capture_count=new,exact_reused_count=reused,roster_sha256=sha(BASE/'ROSTER_LOCK.json'),wall_seconds=time.perf_counter()-start,GT_read=False))
if __name__=='__main__':main()
