"""Finite serial matched batch; no old task controller is invoked."""
import os,time,subprocess,traceback,fcntl
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_proximal_common_v1 import *
def call(script,args,log):
 with Path(log).open('a') as f:
  subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/'+script,*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
def run():
 verify();fh=(BASE/'PROCESS.lock').open('w');fcntl.flock(fh,fcntl.LOCK_EX|fcntl.LOCK_NB)
 status(BASE/'LAUNCH.json',dict(pid=os.getpid(),time=time.time(),entrypoint=__file__))
 try:
  for arms in [['A'],['B','C','D']]:
   for ds in DATASETS:
    for a in arms:
     req=BASE/ds/a/'REQUEST.json';status(BASE/'STATUS.json',dict(status='running_spatial',dataset=ds,arm=a,pid=os.getpid(),time=time.time()))
     call('run_tastvg_proximal_trial_v1.py',[ds,str(req)],BASE/ds/(a+'.log'))
   if arms==['A']:
    call('calibrate_tastvg_proximal_v1.py',[],BASE/'CALIBRATION.log')
  write(BASE/'GLOBAL_PREDICTION_BARRIER.json',dict(cells=3072,GT_read=False,
   files={f'{ds}/{a}/PREDICTION_BARRIER.json':sha(BASE/ds/a/'PREDICTION_BARRIER.json') for ds in DATASETS for a in ['A','B','C','D']},time=time.time()))
  status(BASE/'STATUS.json',dict(status='spatial_predictions_complete_pending_cpu_score',pid=os.getpid(),time=time.time()))
 except BaseException as e:
  status(BASE/'STATUS.json',dict(status='failed',error=repr(e),traceback=traceback.format_exc(),pid=os.getpid(),time=time.time()));raise
if __name__=='__main__':run()
