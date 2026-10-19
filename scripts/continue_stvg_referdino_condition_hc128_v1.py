"""One finite sequential worker, no recurring schedule or automatic research."""
import os,sys,time,subprocess,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put
D=ROOT/'artifacts/stvg_referdino_condition_hc128_v1'
def main():
 lock=(D/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 # A single already-running capture is a dependency; never restart it here.
 while not(D/'CAPTURE_BARRIER.json').exists():
  active=subprocess.run(['systemctl','--user','is-active','stvg-refer-condition-hc128-capture-003.service'],capture_output=True).stdout.decode().strip();assert active=='active',('capture did not complete',active);time.sleep(15)
 for stage,bar in [('teacher0','TEACHER0_BARRIER.json'),('student1','STUDENT1_BARRIER.json'),('teacher1','TEACHER1_BARRIER.json'),('student2','GLOBAL_PREDICTION_BARRIER.json')]:
  if(D/bar).exists():continue
  p=read(D/'PLAN_LOCK.json');assert time.time()<p['deadline_epoch'];cmd=[str(ROOT/'.conda/tubedetr/bin/python'),'-u','-B','scripts/run_stvg_referdino_condition_hc128_v1.py',stage]
  env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
  with(D/(stage+'.log')).open('ab')as out:
   child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT);put(D/'WORKER.json',dict(stage=stage,pid=child.pid,started_epoch=time.time()),True);rc=child.wait(timeout=900)
  assert rc==0,(stage,rc);assert(D/bar).exists()
 put(D/'CONTROLLER_COMPLETION.json',dict(status='all128_outputs_sealed',GT_read=False))
if __name__=='__main__':
 try:main()
 except BaseException as e:put(D/f'CONTROLLER_FAILURE_{time.time_ns()}.json',dict(error=str(e),type=type(e).__name__,partials_preserved=True));raise
