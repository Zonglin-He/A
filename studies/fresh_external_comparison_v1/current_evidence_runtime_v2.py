"""Original numerical preparation unchanged; keep owned expert temp/logs on close."""
from pathlib import Path
import sys,shutil
sys.path.insert(0,str(Path(__file__).parent))
import current_evidence_v1 as original
from current_evidence_v1 import *
import math_endpoint_contract_v2 as compatible_math

def validate_execution():
 original.validate();p=read(N/'execution_v2_plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for field in ['code_sha256','input_sha256']:
  for f,h in p[field].items():assert sha(R/f)==h,f
 assert read(N/'preflight_v1.json')['status']=='passed_four_actual_same_RGB_original_teacher_current_baseline_full_math_repeat_exact'
 return p

if __name__=='__main__':
 import argparse
 import stvg.teacher_bridge as module
 validate_execution();original.math=compatible_math;a=argparse.ArgumentParser();a.add_argument('command',choices=['run','bridge']);q=a.parse_args()
 if q.command=='bridge':original.bridge()
 else:
  old=module.ResidentTeacher
  class PreservingTeacher(old):
   def close(self):
    if not getattr(self,'_preservation_cleanup_installed',False):
     cleanup=self.directory.cleanup;temp=Path(self.directory.name);self._preservation_cleanup_installed=True
     def keep_before_cleanup():
      if temp.exists():
       dest=N/'owned_current_expert_runtime_preserved_v2';shutil.copytree(temp,dest,dirs_exist_ok=True);atomic(N/'owned_current_expert_runtime_preservation_v2.json',dict(status='owned_request_reply_and_expert_log_saved_after_process_close_before_temp_cleanup',files={str(f.relative_to(dest)):dict(sha256=sha(f),bytes=f.stat().st_size)for f in dest.rglob('*')if f.is_file()},new_OriginalSource=0))
      cleanup()
     self.directory.cleanup=keep_before_cleanup
    return super().close()
  module.ResidentTeacher=PreservingTeacher
  try:original.run()
  finally:module.ResidentTeacher=old
