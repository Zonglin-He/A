"""Finite authorized follow-through: CPU scores, conditional control, native head.

Waits for this batch's prediction controller only, never for historical queues.
No second spatial controller is created. Immutable additions are locked below.
"""
import sys,time,os,subprocess,traceback,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_proximal_common_v1 import *
def call(name,args,log):
 with Path(log).open('a') as f:subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B','scripts/'+name,*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
def state(stage,**kw):status(BASE/'FOLLOWTHROUGH_STATUS.json',dict(status=stage,pid=os.getpid(),time=time.time(),**kw))
def run():
 verify();lock=read(BASE/'FOLLOWTHROUGH_RUNTIME_LOCK.json')
 for rel,h in lock['pins'].items():assert sha(ROOT/rel)==h
 fh=(BASE/'FOLLOWTHROUGH_PROCESS.lock').open('w');fcntl.flock(fh,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state('waiting_for_spatial_prediction_barrier')
 try:
  while not (BASE/'GLOBAL_PREDICTION_BARRIER.json').exists():
   s=read(BASE/'STATUS.json');assert s['status']!='failed',s
   if 'pid' in s:os.kill(s['pid'],0)
   time.sleep(10)
  for ds in DATASETS:
   for a in ['A','B','C','D']:
    state('cpu_spatial_score',dataset=ds,arm=a)
    call('score_tastvg_proximal_v1.py',[ds,a],BASE/ds/(a+'_score.log'))
  call('readout_tastvg_proximal_v1.py',['controls'],BASE/'READOUT.log')
  controls=[ds for ds in DATASETS if read(BASE/ds/'CONTROL_TRIGGER.json')['run_fixed_mixture']]
  for ds in controls:
   state('running_fixed_control',dataset=ds,arm='E_fixed')
   call('run_tastvg_proximal_trial_v1.py',[ds,str(BASE/ds/'E_fixed/REQUEST.json')],BASE/ds/'E_fixed.log')
  write(BASE/'CONTROL_PREDICTION_BARRIER.json',dict(datasets=controls,files={f'{ds}/E_fixed/PREDICTION_BARRIER.json':sha(BASE/ds/'E_fixed/PREDICTION_BARRIER.json') for ds in controls},GT_read=False,time=time.time()))
  for ds in controls:call('score_tastvg_proximal_v1.py',[ds,'E_fixed'],BASE/ds/'E_fixed_score.log')
  call('readout_tastvg_proximal_v1.py',['selection'],BASE/'READOUT.log')
  for ds in DATASETS:
   state('native_temporal_smoke',dataset=ds)
   call('smoke_tastvg_native_temporal_v1.py',[ds],BASE/ds/'T_smoke.log')
   assert read(BASE/ds/'T/SMOKE.json')['status']=='pass'
  write(BASE/'TEMPORAL_SMOKE_ACCEPTANCE.json',dict(status='pass',checks='current-output parity, 514 parameters, spatial bitwise, legal pairs, cap',files={ds:sha(BASE/ds/'T/SMOKE.json') for ds in DATASETS},GT_read=False,time=time.time()))
  for ds in DATASETS:
   state('native_temporal_predictions',dataset=ds,arm='T')
   call('run_tastvg_proximal_temporal_v1.py',[ds,str(BASE/ds/'T/REQUEST.json')],BASE/ds/'T.log')
  write(BASE/'TEMPORAL_PREDICTION_BARRIER.json',dict(cells=768,GT_read=False,files={f'{ds}/T/PREDICTION_BARRIER.json':sha(BASE/ds/'T/PREDICTION_BARRIER.json') for ds in DATASETS},selection_sha256={ds:sha(BASE/ds/'SPATIAL_SELECTION.json') for ds in DATASETS},time=time.time()))
  for ds in DATASETS:
   state('cpu_native_temporal_score',dataset=ds,arm='T')
   call('score_tastvg_proximal_v1.py',[ds,'T'],BASE/ds/'T_score.log')
   call('audit_tastvg_native_temporal_v1.py',[ds],BASE/ds/'T_audit.log')
  state('completed_pending_root_audit_publication')
 except BaseException as e:
  state('failed',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':run()
