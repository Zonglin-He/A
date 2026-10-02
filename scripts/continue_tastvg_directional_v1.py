"""Finite serial six-stream queue; GT scoring is a later root operation."""
import os,time,subprocess,traceback,fcntl,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_directional_common_v1 import *
def run():
 verify();assert read(BASE/'SMOKE_ROOT_ACCEPTANCE.json')['status']=='pass'
 fh=(BASE/'PROCESS.lock').open('w');fcntl.flock(fh,fcntl.LOCK_EX|fcntl.LOCK_NB)
 status(BASE/'LAUNCH.json',dict(pid=os.getpid(),time=time.time(),entrypoint=__file__))
 try:
  for ds in DATASETS:
   for arm in ['A','E','F']:
    status(BASE/'STATUS.json',dict(status='running_predictions',dataset=ds,arm=arm,pid=os.getpid(),time=time.time()))
    with (BASE/ds/(arm+'.log')).open('a') as log:
     subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/run_tastvg_directional_trial_v1.py',ds,str(BASE/ds/arm/'REQUEST.json')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
  write(BASE/'GLOBAL_PREDICTION_BARRIER.json',dict(cells=2304,GT_read=False,files={f'{ds}/{a}/PREDICTION_BARRIER.json':sha(BASE/ds/a/'PREDICTION_BARRIER.json') for ds in DATASETS for a in ['A','E','F']},time=time.time()))
  status(BASE/'STATUS.json',dict(status='predictions_complete_pending_root_score_audit_publication',pid=os.getpid(),time=time.time()))
 except BaseException as e:
  status(BASE/'STATUS.json',dict(status='failed',error=repr(e),traceback=traceback.format_exc(),pid=os.getpid(),time=time.time()));raise
if __name__=='__main__':run()
