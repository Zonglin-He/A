"""Single bounded continuation of source capture, two heads, gated target, audit."""
from pathlib import Path
import sys,os,time,subprocess,fcntl
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put
D=ROOT/'artifacts/stvg_dfine_refinement32_v1';P=ROOT/'results/stvg_error_state/2026-10-18/dfine_refinement32'

def main():
 handle=(D/'controller.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);put(D/'CONTROLLER_RUNTIME001.json',dict(pid=os.getpid(),epoch=time.time(),scope='single finite source teacher prototype and conditional target five-arm'))
 deadline=read(D/'PLAN_LOCK.json')['deadline_epoch']
 while not(D/'SOURCE_FEATURE_BARRIER.json').exists()or not(D/'SOURCE_GPU_STOP_LATEST.json').exists():
  assert time.time()<deadline
  state=subprocess.check_output(['systemctl','--user','show','stvg-dfine-source-capture-002.service','-p','ActiveState','--value'],text=True).strip()
  assert state in ['active','activating'],('source capture stopped without complete barrier',state)
  time.sleep(5)
 def run(script,stage):
  with (D/('stage_'+stage+'.log')).open('ab')as log:
   subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B',script,stage],cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT,timeout=max(1,deadline-time.time()),check=True)
 run('scripts/run_stvg_dfine_refinement32_v1.py','lock');run('scripts/run_stvg_dfine_refinement32_v1.py','train')
 if read(P/'SOURCE_GATE.json')['passed']:run('scripts/run_stvg_dfine_refinement32_v1.py','target')
 for stage in ['score','audit','report']:run('scripts/audit_stvg_dfine_refinement32_v1.py',stage)
 put(D/'CONTROLLER_COMPLETION.json',dict(status='success',source_models=2,target_test_ran=read(P/'SOURCE_GATE.json')['passed'],publication_archive_pending=True))
if __name__=='__main__':
 try:main()
 except BaseException as e:put(D/('CONTROLLER_FAILURE_'+str(time.time_ns())+'.json'),dict(error=str(e),all_partials_preserved=True));raise
