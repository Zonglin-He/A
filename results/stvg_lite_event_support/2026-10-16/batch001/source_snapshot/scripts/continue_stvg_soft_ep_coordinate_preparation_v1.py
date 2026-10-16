"""Finite serial preparation, then an explicit root handoff before parameter fits."""
from pathlib import Path
import sys,subprocess,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_soft_ep_coordinate_common_v1 import *
PY=ROOT/'.conda/tubedetr/bin/python'
def main():
 verify_runtime()
 for stage in ['source','temporal','fit','inputs','teacher','references']:
  write(BASE/'PREPARATION_STATUS.json',dict(status='running',stage=stage,GT_read=False,time=time.time()),mutable=True)
  if stage in ['source','temporal','teacher']:
   subprocess.run(['systemd-run','--user','--wait','--collect','--unit=stvg-soft-ep-coordinate128-'+stage+'-001','--property=MemoryHigh=16G','--property=MemoryMax=20G','--property=MemorySwapMax=1G','--working-directory='+str(ROOT),str(PY),'-B','scripts/launch_stvg_soft_ep_coordinate_v1.py',stage],cwd=ROOT,check=True)
  else:subprocess.run([str(PY),'-B','scripts/run_stvg_soft_ep_coordinate_inputs_v1.py',stage],cwd=ROOT,check=True)
 write(BASE/'PREPARATION_STATUS.json',dict(status='pending_actual_root_assembly_and_coordinate_search',stage='root_handoff',GT_read=False,time=time.time()),mutable=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  write(BASE/'PREPARATION_FAILURE.json',dict(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),GT_read=False,time=time.time()),mutable=True);raise
