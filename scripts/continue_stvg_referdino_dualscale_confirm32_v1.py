"""One finite serial confirmation, after the development selection lock."""
from pathlib import Path
import os,sys,subprocess,fcntl,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put
from scripts.run_stvg_referdino_dualscale_confirm32_v1 import D
def main():
 lock=(D/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);py=str(ROOT/'.conda/tubedetr/bin/python');env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
 jobs=[('capture',[py,'-u','-B','scripts/run_stvg_referdino_dualscale_confirm32_v1.py','capture']),('teacher',[py,'-u','-B','scripts/run_stvg_referdino_dualscale_confirm32_v1.py','teacher'])]+[(f'measure{i}',[py,'-u','-B','scripts/run_stvg_referdino_dualscale_confirm32_v1.py','measure','--chunk',str(i)])for i in range(2)]+[(s,[py,'-u','-B',f'scripts/{s}_stvg_referdino_dualscale_confirm32_v1.py'])for s in ['score','audit','report']]
 jobs.insert(1,('source_lock',[py,'-u','-B','scripts/evaluate_stvg_referdino_dualscale_source_baseline_v1.py','lock','--confirmation']))
 jobs.append(('source_score',[py,'-u','-B','scripts/evaluate_stvg_referdino_dualscale_source_baseline_v1.py','score','--confirmation']))
 for stage,cmd in jobs:
  assert time.time()<read(D/'PLAN_LOCK.json')['deadline_epoch']
  with (D/(stage+'.log')).open('ab')as f:
   child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT);put(D/'WORKER.json',dict(stage=stage,pid=child.pid),True);rc=child.wait(timeout=900)
  assert rc==0,(stage,rc)
 put(D/'CONTROLLER_COMPLETION.json',dict(status='confirmation_model_score_audit_report_complete',count=32))
if __name__=='__main__':
 try:main()
 except BaseException as e:put(D/f'CONTROLLER_FAILURE_{time.time_ns()}.json',dict(type=type(e).__name__,error=str(e),partials_preserved=True));raise
