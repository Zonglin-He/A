"""Finite serial Table1 continuation after the existing Ours controller.

No GPU concurrency, scoring, silent row omission or historic-queue resumption.
Missing official HC train media is an explicit remaining-EATA handoff; the
other authorized Table1 arms run after independently checked live port smoke.
"""
import os,sys,time,subprocess,fcntl,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_paper_common_v1 import *
STAGE=BASE/'TABLE1_CONTINUATION_STAGE.json'

def active(pid):
 try:os.kill(pid,0);return True
 except ProcessLookupError:return False

def run():
 gate=(BASE/'TABLE1_CONTINUATION.lock').open('a');fcntl.flock(gate,fcntl.LOCK_EX|fcntl.LOCK_NB)
 launch=read(BASE/'LAUNCH.json');old_pid=launch['pid']
 status(STAGE,dict(status='waiting_for_existing_Ours_controller',pid=os.getpid(),predecessor_pid=old_pid,GT_read=False,time=time.time()))
 while active(old_pid):
  # PID reuse does not license treating an unrelated process as predecessor.
  command=(Path('/proc')/str(old_pid)/'cmdline')
  if command.exists():assert b'continue_decota_paper_v1.py' in command.read_bytes(),'Original predecessor PID was reused'
  time.sleep(10)
 assert read(BASE/'STATUS.json')['status']=='table1_ours_sealed_pending_root_baselines_and_remaining_stages','No valid Ours handoff; preserve all artifacts, root recovery required'
 verify();assert read(BASE/'GLOBAL_PREDICTION_BARRIER.json')['status']=='sealed'
 # Acquire the original controller guard only after that controller exits.
 guardfile=(BASE/'CONTROLLER.lock').open('a');fcntl.flock(guardfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
 files=['scripts/continue_decota_paper_table1_v1.py','scripts/run_decota_paper_baselines_v1.py','vg_tta/decota_paper_baselines_20261005_v1.py','vg_tta/decota_paper_baseline_math_v1.py',
  'scripts/run_decota_paper_reference_v1.py','scripts/derive_decota_paper_stateless_v1.py','vg_tta/decota_paper_controls_v1.py','protocols/decota_paper_baselines_20261005_v1.md']
 lockfile=BASE/'TABLE1_CONTINUATION_RUNTIME.json'
 if not lockfile.exists():write(lockfile,dict(pins={f:sha(ROOT/f) for f in files},main_runtime_sha256=sha(BASE/'RUNTIME_LOCK.json'),GT_read=False,time=time.time()))
 pinned=read(lockfile)
 for f,h in pinned['pins'].items():assert sha(ROOT/f)==h
 def step(script,action=''):
  budget();name=Path(script).stem+'_'+action;log=BASE/(name+'.log')
  with log.open('ab') as out:
   cmd=[str(ROOT/'.conda/tubedetr/bin/python'),'-B',str(ROOT/script)]+([action] if action else [])
   p=subprocess.Popen(cmd,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
   state=dict(status='running',pid=os.getpid(),controller_pid=os.getpid(),worker_pid=p.pid,stage=name,log=str(log),GT_read=False,time=time.time())
   status(STAGE,state);status(BASE/'STATUS.json',{**state,'phase':'table1_remaining_arms'})
   assert p.wait()==0,'Table1 continuation worker failed: '+name
 step('scripts/derive_decota_paper_stateless_v1.py')
 for ds in ['hc2','vidstg']:step('scripts/run_decota_paper_reference_v1.py',ds)
 step('scripts/run_decota_paper_baselines_v1.py','lock')
 for method in ['TENT','SAR']:
  for ds in ['hc2','vidstg']:
   if not (BASE/'baselines'/f'{method}_{ds}'/'PREDICTION_BARRIER.json').exists():
    step('scripts/run_decota_paper_baselines_v1.py','smoke_'+method+'_'+ds)
    step('scripts/run_decota_paper_baselines_v1.py',method+'_'+ds)
 step('scripts/run_decota_paper_baselines_v1.py','fisher_vidstg')
 if not (BASE/'baselines/EATA_hc2/PREDICTION_BARRIER.json').exists():
  step('scripts/run_decota_paper_baselines_v1.py','smoke_EATA_hc2');step('scripts/run_decota_paper_baselines_v1.py','EATA_hc2')
 if not (BASE/'source_Fisher/hc2/MEDIA_BARRIER.json').exists():
  final=dict(status='remaining_EATA_HC_source_media_pending_root',phase='table1',controller_pid=os.getpid(),completed_rows=['Ours','Source Only','DINO-Refine','Target-trained reference','TENT-STVG','SAR-STVG','EATA-STVG Vid-source to HC2'],
   pending=['EATA-STVG HC-source to Vid: exact sealed source2000 media/Fisher/live smoke','all-arm seal and independent Table1 scoring/publication','Table2 and remaining MASTER_PLAN'],GT_read=False,time=time.time())
  status(STAGE,final);status(BASE/'STATUS.json',final)
  archive('Table1已完成可执行所有行封存，HC源EATA缺锁定训练媒体明确handoff，未评分/未省略EATA/未称主表完成；根接续官方媒体和剩余stage')
  return
 step('scripts/run_decota_paper_baselines_v1.py','fisher_hc2')
 if not (BASE/'baselines/EATA_vidstg/PREDICTION_BARRIER.json').exists():
  step('scripts/run_decota_paper_baselines_v1.py','smoke_EATA_vidstg');step('scripts/run_decota_paper_baselines_v1.py','EATA_vidstg')
 barriers={}
 for job,ds,_ in JOBS:
  expected=read(BASE/ds/'PLAN.json')['queries']*3
  paths=[BASE/job/'PREDICTION_BARRIER.json',BASE/'table1_stateless'/job/'PREDICTION_BARRIER.json',BASE/'table1_target_reference'/ds/'PREDICTION_BARRIER.json']
  paths += [BASE/'baselines'/f'{m}_{ds}'/'PREDICTION_BARRIER.json' for m in ['TENT','EATA','SAR']]
  for path in paths:
   rc=read(path);assert rc['status']=='sealed'
   assert rc.get('cells',rc.get('logical_rows_per_arm',rc.get('logical_rows')))==expected
   for p,h in rc['files'].items():assert sha(BASE/p)==h
   barriers[str(path.relative_to(BASE))]=sha(path)
 bf=BASE/'TABLE1_ALL_ARM_PREDICTION_BARRIER.json'
 if not bf.exists():write(bf,dict(status='sealed',methods=7,orders=3,logical_rows_per_method=41355,barriers=barriers,GT_read=False,time=time.time()))
 status(STAGE,dict(status='table1_all_predictions_sealed_pending_root_independent_audit',controller_pid=os.getpid(),GT_read=False,time=time.time()))
 status(BASE/'STATUS.json',read(STAGE));archive('Table1七行两方向三序全部预测已封存，根须独立数学/dense/配对审计和公开，之后继续Table2及全部MASTER_PLAN，尚未完成paper')

if __name__=='__main__':
 try:run()
 except BaseException:
  fd=BASE/'table1_continuation_failure'/str(time.time_ns());fd.mkdir(parents=True,exist_ok=True);(fd/'traceback.txt').write_text(traceback.format_exc())
  status(STAGE,dict(status='failed',pid=os.getpid(),failure=str(fd),GT_read=False,time=time.time()));raise
