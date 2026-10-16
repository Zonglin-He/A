"""Resume completed Source capture into the pre-existing qualified R1 environment."""
from pathlib import Path
import sys,subprocess,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_soft_ep_coordinate_common_v1 import *
PY=ROOT/'.conda/tubedetr/bin/python'
def main():
 verify_runtime();r=read(BASE/'RECOVERY_RUNTIME002.json')
 for p,h in {**r['pins'],**r['protected']}.items():assert sha(ROOT/p)==h,p
 assert read(BASE/'SOURCE_CAPTURE_BARRIER.json')['count']==128
 for stage in ['temporal','fit','inputs','teacher','references','assembly']:
  write(BASE/'PREPARATION_STATUS002.json',dict(status='running',stage=stage,GT_read=False,time=time.time()),mutable=True)
  if stage in ['temporal','teacher']:
   subprocess.run(['systemd-run','--user','--wait','--collect','--unit=stvg-soft-ep-coordinate128-'+stage+'-002','--property=MemoryHigh=16G','--property=MemoryMax=20G','--property=MemorySwapMax=1G','--working-directory='+str(ROOT),str(PY),'-B','scripts/launch_stvg_soft_ep_coordinate002.py',stage],cwd=ROOT,check=True)
   assert read(BASE/('execution_'+stage+'002')/'WORKER_EXIT.json')['exit_code']==0
  else:
   script='scripts/assemble_stvg_soft_ep_coordinate_v1.py'if stage=='assembly'else'scripts/run_stvg_soft_ep_coordinate_inputs_v1.py'
   subprocess.run([str(PY),'-B',script,*([]if stage=='assembly'else[stage])],cwd=ROOT,check=True)
 write(BASE/'PREPARATION_STATUS002.json',dict(status='pending_actual_coordinate_search',stage='root_handoff',GT_read=False,time=time.time()),mutable=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  write(BASE/'PREPARATION_FAILURE002.json',dict(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),GT_read=False,time=time.time()),mutable=True);raise
