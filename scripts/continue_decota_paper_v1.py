"""Finite priority queue: actual clean cross-domain Ours first, then root handoff."""
import sys,os,time,subprocess,fcntl,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_paper_common_v1 import *

def run():
 BASE.mkdir(parents=True,exist_ok=True)
 f=(BASE/'CONTROLLER.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 def step(action):
  log=BASE/f'{action}.log'
  with log.open('ab') as out:
   p=subprocess.Popen([str(ROOT/'.conda/tubedetr/bin/python'),'-B',str(ROOT/'scripts/run_decota_paper_main_v1.py'),action],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
   status(BASE/'STATUS.json',dict(status='running',phase='table1_ours',stage=action,controller_pid=os.getpid(),worker_pid=p.pid,log=str(log),GT_read=False,time=time.time()))
   assert p.wait()==0,'Worker failed; preserve and root recovery required: '+action
 step('lock')
 if not (BASE/'SMOKE_ROOT_ACCEPTANCE.json').exists():step('smoke')
 for job,_,_ in JOBS:
  if not (BASE/job/'PREDICTION_BARRIER.json').exists():step(job)
 if not (BASE/'GLOBAL_PREDICTION_BARRIER.json').exists():step('seal')
 # This barrier covers Ours and its shared Source/DINO inputs, not all baselines.
 status(BASE/'STATUS.json',dict(status='table1_ours_sealed_pending_root_baselines_and_remaining_stages',controller_pid=os.getpid(),ours_cells=41355,GT_read=False,
  root_required=['qualify_TENT_EATA_SAR_ports_and_source_Fisher','target_trained_frozen_reference','complete_all_Table1_arms_and_seal_before_GT','Table2_then_remaining_MASTER_PLAN'],time=time.time()))
 archive('Table1两个clean跨域Ours完整41355预测已封存，尚非Table1全方法或完整paper完成；根必须接续已授权baselines/Fisher/reference与后续所有stage，不能只停留监测')

if __name__=='__main__':
 try:run()
 except BaseException:
  p=BASE/'controller_failure'/str(time.time_ns());p.mkdir(parents=True,exist_ok=True);(p/'traceback.txt').write_text(traceback.format_exc())
  status(BASE/'STATUS.json',dict(status='failed',phase='table1_ours',controller_pid=os.getpid(),failure_path=str(p),GT_read=False,time=time.time()));raise
