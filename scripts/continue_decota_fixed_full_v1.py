"""Finite serial queue; preparation wait, smoke, four full jobs, global seal."""
import sys,os,time,subprocess,fcntl,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_fixed_full_common_v1 import *
def run():
 BASE.mkdir(parents=True,exist_ok=True)
 lease=(BASE/'CONTROLLER.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for ds in DATASETS:
  while not all((BASE/d/'subjects/00001.json').exists() for d in DATASETS):
   status(BASE/'STATUS.json',dict(status='waiting_first_query_parses',controller_pid=os.getpid(),predictions=0,GT_read=False,time=time.time()));time.sleep(10)
 def step(action):
  log=BASE/f'{action}.log'
  with log.open('ab') as f:
   p=subprocess.Popen([str(ROOT/'.conda/tubedetr/bin/python'),'-B',str(ROOT/'scripts/run_decota_fixed_full_v1.py'),action],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
   status(BASE/'STATUS.json',dict(status='running',stage=action,controller_pid=os.getpid(),worker_pid=p.pid,log=str(log),GT_read=False,time=time.time()))
   assert p.wait()==0,'Worker failed; no automatic retry: '+action
 step('lock')
 if not (BASE/'SMOKE_ROOT_ACCEPTANCE.json').exists():step('smoke')
 while not all((BASE/d/'SUBJECT_BARRIER.json').exists() for d in DATASETS):
  status(BASE/'STATUS.json',dict(status='waiting_complete_subject_preparation',controller_pid=os.getpid(),smoke_pass=True,GT_read=False,time=time.time()));time.sleep(20)
 for job,_,_ in JOBS:
  if not (BASE/job/'PREDICTION_BARRIER.json').exists():step(job)
 if not (BASE/'GLOBAL_PREDICTION_BARRIER.json').exists():step('seal')
 status(BASE/'STATUS.json',dict(status='completed_pending_root_CPU_score_diagnosis_publication',controller_pid=os.getpid(),cells=330840,GT_read=False,time=time.time()))
 archive('四个完整队列330840预测已实际封存；须根接续本轮CPU计分/GT pipeline诊断/独立审计/图与GitHub公开核验，尚非最终完成')
if __name__=='__main__':
 try:run()
 except BaseException:
  f=BASE/'controller_failure'/str(time.time_ns());f.mkdir(parents=True,exist_ok=True);(f/'traceback.txt').write_text(traceback.format_exc())
  status(BASE/'STATUS.json',dict(status='failed',controller_pid=os.getpid(),failure_path=str(f),time=time.time()));raise
