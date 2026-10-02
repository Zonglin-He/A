"""Finite two-dataset serial queue, root must handle scoring and token P0."""
import os,sys,fcntl,subprocess,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_routed_common_v1 import *
def run():
 verify();fh=(BASE/'PROCESS.lock').open('w');fcntl.flock(fh,fcntl.LOCK_EX|fcntl.LOCK_NB);status(BASE/'LAUNCH.json',dict(pid=os.getpid(),time=time.time(),entrypoint=__file__))
 for ds in DATASETS:
  status(BASE/'STATUS.json',dict(status='running_online',dataset=ds,pid=os.getpid(),GT_read=False,time=time.time()))
  with (BASE/ds/'WORKER.log').open('a') as log:subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/run_tastvg_routed_online_v1.py',ds],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 verify();write(BASE/'GLOBAL_PREDICTION_BARRIER.json',dict(cells=1536,new_R_cells=768,reused_A_cells=768,GT_read=False,files={f'{ds}/{a}/PREDICTION_BARRIER.json':sha(BASE/ds/a/'PREDICTION_BARRIER.json') for ds in DATASETS for a in ['A','R']},time=time.time()))
 status(BASE/'STATUS.json',dict(status='online_sealed_pending_root_and_token_P0',done=768,total=768,GT_read=False,time=time.time()))
if __name__=='__main__':
 try:run()
 except BaseException as e:status(BASE/'STATUS.json',dict(status='failed',error=repr(e),traceback=traceback.format_exc(),time=time.time()));raise
