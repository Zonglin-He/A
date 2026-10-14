"""Fixed cached-feedback temporal update comparison; never reads evaluation labels."""
import sys, json, time, traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.stvg_temporal_caption_query_v1 import ROOT,BASE as OLD,read,write,sha,roster,catalog,inference_guard
BASE=ROOT/'artifacts/stvg_support_aware_temporal_spatial_update_v1'

def main():
 import torch
 from scripts.stvg_support_aware_temporal_v1 import fit
 torch.set_num_threads(2);sys.addaudithook(inference_guard)
 rt=read(BASE/'TEMPORAL_RUNTIME.json')
 for p,h in rt['pins'].items():assert sha(ROOT/p)==h,p
 for name in ['TEACHER_GLOBAL_BARRIER.json','STUDENT_GLOBAL_BARRIER.json']:
  seal=read(OLD/name);assert seal['status']=='complete' and seal['GT_read'] is False
  for f in seal['files']:assert sha(ROOT/f['path'])==f['sha256']
 jobs=read(OLD/'CAPTION_JOB_LOCK.json')['records'];files=[]
 for rec in roster():
  case,_=catalog(rec);prior=OLD/'student'/f"{rec['case_id']}.pt"
  z=torch.load(prior,map_location='cpu',weights_only=False)
  local=[j for j in jobs if j['case_id']==rec['case_id']]
  rewards=[read(OLD/'teacher'/f"{rec['case_id']}_{j['candidate']['canonical_id']}_alignment.json")['reward'] for j in local]
  results={}
  for mode in ['original_adam','ridge','unconstrained_ridge','trust','per_offset_trust']:
   result=fit(case,z['temporal_hidden'],[j['candidate'] for j in local],rewards,mode)
   if mode=='original_adam':
    for old,new in zip(z['temporal_fit']['path'],result['path']):
     assert torch.equal(old['theta'],new['theta']) and old['loss']==new['loss'] and old['physical_interval']==new['physical_interval']
   results[mode]=result
  dst=BASE/'temporal'/f"{rec['case_id']}.pt";dst.parent.mkdir(exist_ok=True);assert not dst.exists()
  torch.save(dict(case_id=rec['case_id'],dataset=rec['dataset'],source_student_sha256=sha(prior),results=results,GT_read=False,new_teacher_calls=0,new_model_forwards=0),dst)
  files.append(dict(case_id=rec['case_id'],dataset=rec['dataset'],path=str(dst.relative_to(ROOT)),sha256=sha(dst)))
  write(BASE/'TEMPORAL_STATUS.json',dict(status='running',done=len(files),total=8,GT_read=False,CPU_only=True,time=time.time()),mutable=True)
 assert len(files)==8
 write(BASE/'TEMPORAL_GLOBAL_BARRIER.json',dict(status='complete',files=files,count=8,CPU_fit_modes=5,actual_new_teacher_calls=0,actual_new_model_forwards=0,GT_read=False,CPU_only=True,time=time.time()))
 write(BASE/'TEMPORAL_STATUS.json',dict(status='complete_prediction_seal_pending_root_readback',done=8,total=8,GT_read=False,CPU_only=True,time=time.time()),mutable=True)
 print(json.dumps({'status':'complete_prediction_seal','cases':8,'mode_count':5,'teacher_calls':0,'model_forward_calls':0}))

if __name__=='__main__':
 try:main()
 except BaseException as e:
  write(BASE/'TEMPORAL_FAILURE_CAPTURE.json',dict(status='failure',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),GT_read=False),mutable=True);raise
