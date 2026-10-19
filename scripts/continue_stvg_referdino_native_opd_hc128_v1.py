"""One finite serial four-chunk run, then postseal score/audit/report; no schedule."""
from pathlib import Path
import os,sys,time,subprocess,fcntl
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put
D=ROOT/'artifacts/stvg_referdino_native_opd_hc128_v1'
def main():
 lock=(D/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);plan=read(D/'PLAN_LOCK.json');py=str(ROOT/'.conda/tubedetr/bin/python');env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
 jobs=[('measure',i,[py,'-u','-B','scripts/run_stvg_referdino_native_opd_hc128_v1.py','measure','--chunk',str(i)],D/f'CHUNK_{i}_BARRIER.json')for i in range(4)]
 jobs +=[(s,None,[py,'-u','-B',f'scripts/{s}_stvg_referdino_native_opd_hc128_v1.py'],None)for s in ['score','audit','report']]
 for stage,chunk,cmd,bar in jobs:
  if bar and bar.exists():continue
  assert time.time()<plan['deadline_epoch'];log=D/(stage+(''if chunk is None else str(chunk))+'.log')
  with log.open('ab')as out:
   child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT);put(D/'WORKER.json',dict(stage=stage,chunk=chunk,pid=child.pid,start_epoch=time.time()),True);rc=child.wait(timeout=900)
  assert rc==0,(stage,chunk,rc)
  if bar:assert bar.exists()
 put(D/'CONTROLLER_COMPLETION.json',dict(status='run_score_audit_report_complete',count=128,seeds=plan['seeds'],publication_archive_pending=True))
if __name__=='__main__':
 try:main()
 except BaseException as e:put(D/f'CONTROLLER_FAILURE_{time.time_ns()}.json',dict(type=type(e).__name__,error=str(e),partials_preserved=True));raise
