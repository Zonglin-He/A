"""Hot frozen source capture for the newly locked confirmation parents; no GT/teacher."""
import sys,os,time,gc,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_evidence_common_v1 import *
def main():
 import torch,numpy as np
 from scripts import run_stvg_table1_source_v1 as source
 from scripts.run_decota_paper_main_v1 import gpu,model_for
 from methods.decota_final_simplified_v1.tensors import state_hash
 runtime=read(BASE/'SOURCE_RUNTIME.json')
 for p,h in {**runtime['pins'],**runtime.get('protected',{})}.items():assert sha(ROOT/p)==h,p
 source.RUNTIME=BASE/'SOURCE_RUNTIME.json';install_lease=gpu();torch.set_num_threads(2);b=BASE/'confirmation';files=[];model=None;name=None;start=time.perf_counter()
 for rec in roster('confirmation'):
  target=source.SOURCE_MODEL_NAME[rec['dataset']]
  if target!=name:
   if model is not None:del model;gc.collect();torch.cuda.empty_cache()
   model=model_for(target);name=target
  path=ROOT/rec['source_case_path'];assert not path.exists()and not path.with_suffix('.receipt.json').exists();c=source.capture_one(model,rec,collections.OrderedDict());assert c['case_id']==rec['probe_alias']and c['source_case_sha256']==source.content_sha256(c)and not c['GT_read']and state_hash(model.state_dict())==c['source_model_state_sha256'];item=save_pt(path,c)
  write(path.with_suffix('.receipt.json'),dict(status='sealed',case_id=c['case_id'],source_case_sha256=c['source_case_sha256'],query_key_sha256=rec['query_key_sha256'],source_runtime_sha256=sha(BASE/'SOURCE_RUNTIME.json'),sha256=item['sha256'],bytes=item['bytes'],GT_read=False))
  files.append(item);write(b/'SOURCE_STATUS.json',dict(status='running',done=len(files),total=32,GT_read=False),mutable=True);print('source',c['case_id'],len(files),flush=True);del c;gc.collect();torch.cuda.empty_cache()
 write(b/'SOURCE_CAPTURE_BARRIER.json',dict(status='complete',count=32,files=files,roster_sha256=sha(BASE/'CONFIRMATION_ROSTER_LOCK.json'),runtime_sha256=sha(BASE/'SOURCE_RUNTIME.json'),wall_seconds=time.perf_counter()-start,GT_read=False))
if __name__=='__main__':main()
