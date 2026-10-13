"""Single finite sequence; no scheduler, polling daemon, signals, or retries."""
import os,subprocess,sys,time
from scripts.stvg_opd_admission_association_common_v1 import *

def run():
 verify();steps=[('qualification','scripts.run_stvg_opd_admission_association_v1',['qualification']),
  ('qualification_root','scripts.audit_stvg_opd_admission_association_v1',['qualification']),
  ('formal','scripts.run_stvg_opd_admission_association_v1',['formal']),
  ('Oracle_design_postseal','scripts.score_stvg_opd_admission_association_v1',['prepare_oracle']),
  ('oracle','scripts.run_stvg_opd_admission_association_v1',['oracle']),
  ('score','scripts.score_stvg_opd_admission_association_v1',[]),
  ('root','scripts.audit_stvg_opd_admission_association_v1',[])]
 write(BASE/'LAUNCH.json',dict(status='launched_finite_authorized_sequence',pid=os.getpid(),runtime_sha256=sha(BASE/'RUNTIME.json'),GT_read=False,monitor_recreated=False,time=time.time()))
 for stage,module,args in steps:
  verify();status(BASE/'STATUS.json',dict(status='running_actual_'+stage,stage=stage,pid=os.getpid(),GT_read=stage in ['Oracle_design_postseal','oracle','score','root'],time=time.time()))
  p=BASE/'logs'/(stage+'.log');p.parent.mkdir(exist_ok=True)
  with p.open('xb') as f:
   result=subprocess.run([sys.executable,'-B','-m',module,*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  write(BASE/'stage_exit'/(stage+'.json'),dict(status='pass' if result.returncode==0 else 'failed_preserved',stage=stage,returncode=result.returncode,log_sha256=sha(p),time=time.time()))
  if result.returncode:
   status(BASE/'STATUS.json',dict(status='failed_preserved_pending_root',stage=stage,time=time.time()));raise SystemExit(result.returncode)
 status(BASE/'STATUS.json',dict(status='pending_actual_PDF_view_publication_archive',done=448,total=448,GT_read=True,monitor_recreated=False,time=time.time()))
 write(BASE/'CPU_SEQUENCE_COMPLETION.json',dict(status='pending_actual_PDF_view_publication_archive',finite_sequence_completed=True,monitor_recreated=False,time=time.time()))
if __name__=='__main__':run()
