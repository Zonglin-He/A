"""Resume the saved original paper queue only after root closes the OPD study.

This is a queue/optimizer restoration wrapper, not a change to either method.
The original Ours, baseline implementations and scientific locks stay intact.
"""
import sys, os, time, subprocess, fcntl, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_paper_common_v1 import *
PAUSE=BASE/'user_opd_pause_20261006'
OPD=ROOT/'artifacts/decota_spatial_opd_v1'
STAGE=BASE/'TABLE1_CONTINUATION_STAGE.json'

def run():
 closing=read(OPD/'ROOT_CLOSING_RECEIPT.json')
 assert closing['status']=='complete' and closing['visual_inspected']
 assert closing['remote_verified'] and closing['adapted_arrivals']==2880
 assert read(OPD/'GPU_CPU_COMPLETION.json')['adapted_arrivals']==2880
 # No historical queue other than the explicitly saved paper continuation.
 assert read(PAUSE/'PAUSE_COMPLETION.json')['resume_authorized_after_OPD']
 assert read(PAUSE/'EXACT_RESUME_RECEIPT.json')['status']=='pass'
 gates=[]
 for name in ['TABLE1_CONTINUATION.lock','CONTROLLER.lock']:
  f=(BASE/name).open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);gates.append(f)
 verify();pinned=read(BASE/'TABLE1_CONTINUATION_RUNTIME.json')
 for f,h in pinned['pins'].items():assert sha(ROOT/f)==h,f
 own=PAUSE/'CONTINUATION_RUNTIME.json'
 fs=['scripts/continue_decota_paper_after_opd_v1.py','scripts/resume_decota_paper_baseline_saved_v1.py']
 if not own.exists():write(own,dict(pins={f:sha(ROOT/f) for f in fs},main_runtime_sha256=sha(BASE/'RUNTIME_LOCK.json'),original_continuation_runtime_sha256=sha(BASE/'TABLE1_CONTINUATION_RUNTIME.json'),OPD_closing_sha256=sha(OPD/'ROOT_CLOSING_RECEIPT.json'),GT_read=False,time=time.time()))
 for f,h in read(own)['pins'].items():assert sha(ROOT/f)==h,f
 def step(script,action=''):
  verify();budget();log=BASE/(Path(script).stem+'_'+action+'_after_opd.log')
  with log.open('ab') as out:
   command=[str(ROOT/'.conda/tubedetr/bin/python'),'-B',str(ROOT/script)]+([action] if action else [])
   p=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
   state=dict(status='running',pid=os.getpid(),controller_pid=os.getpid(),worker_pid=p.pid,stage=Path(script).stem+'_'+action,log=str(log),GT_read=False,resumed_saved_paper=True,time=time.time())
   status(STAGE,state);status(BASE/'STATUS.json',{**state,'phase':'table1_remaining_arms'})
   assert p.wait()==0,'Saved paper continuation worker failed: '+script+' '+action
 # Already sealed controls/references/ports are kept, not rerun.
 if not all((BASE/'table1_stateless'/j/'PREDICTION_BARRIER.json').exists() for j,_,_ in JOBS):step('scripts/derive_decota_paper_stateless_v1.py')
 for ds in ['hc2','vidstg']:
  if not (BASE/'table1_target_reference'/ds/'PREDICTION_BARRIER.json').exists():step('scripts/run_decota_paper_reference_v1.py',ds)
 step('scripts/run_decota_paper_baselines_v1.py','lock')
 if not (BASE/'baselines/TENT_vidstg/PREDICTION_BARRIER.json').exists():step('scripts/resume_decota_paper_baseline_saved_v1.py')
 for method in ['TENT','SAR']:
  for ds in ['hc2','vidstg']:
   if not (BASE/'baselines'/f'{method}_{ds}'/'PREDICTION_BARRIER.json').exists():
    step('scripts/run_decota_paper_baselines_v1.py','smoke_'+method+'_'+ds)
    step('scripts/run_decota_paper_baselines_v1.py',method+'_'+ds)
 step('scripts/run_decota_paper_baselines_v1.py','fisher_vidstg')
 if not (BASE/'baselines/EATA_hc2/PREDICTION_BARRIER.json').exists():
  step('scripts/run_decota_paper_baselines_v1.py','smoke_EATA_hc2');step('scripts/run_decota_paper_baselines_v1.py','EATA_hc2')
 if not (BASE/'source_Fisher/hc2/MEDIA_BARRIER.json').exists():
  final=dict(status='remaining_EATA_HC_source_media_pending_root',phase='table1',controller_pid=os.getpid(),completed_rows=['Ours','Source Only','DINO-Refine','Target-trained reference','TENT-STVG','SAR-STVG','EATA-STVG Vid-source to HC2'],pending=['Exact source2000 HC media/Fisher/qualification/EATA','all-arm Table1 seal/audit/publication','Table2 cohort unit and remaining stages'],resumed_saved_paper=True,GT_read=False,time=time.time())
  status(STAGE,final);status(BASE/'STATUS.json',final)
  archive('独立OPD完成后已从保存完整optimizer与bitwise尾复核精确接续旧paper；可执行行已封存，HC源EATA锁定媒体与Table2单位仍pending，未省略/未评分/未称全部paper完成')
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
   rc=read(path);assert rc['status']=='sealed' and rc.get('cells',rc.get('logical_rows_per_arm',rc.get('logical_rows')))==expected
   for p,h in rc['files'].items():assert sha(BASE/p)==h
   barriers[str(path.relative_to(BASE))]=sha(path)
 final=BASE/'TABLE1_ALL_ARM_PREDICTION_BARRIER.json'
 if not final.exists():write(final,dict(status='sealed',methods=7,orders=3,logical_rows_per_method=41355,barriers=barriers,GT_read=False,time=time.time()))
 status(STAGE,dict(status='table1_all_predictions_sealed_pending_root_independent_audit',controller_pid=os.getpid(),resumed_saved_paper=True,GT_read=False,time=time.time()));status(BASE/'STATUS.json',read(STAGE))
 archive('保存旧paper接续完成Table1七行双向三序预测seal；根接续独立审计/公开及其余已授权paper阶段，Table2单位hold未擅自解除')

if __name__=='__main__':
 try:run()
 except BaseException:
  f=BASE/'table1_continuation_failure'/str(time.time_ns());f.mkdir(parents=True,exist_ok=True);(f/'traceback.txt').write_text(traceback.format_exc())
  status(STAGE,dict(status='failed',pid=os.getpid(),failure=str(f),resumed_saved_paper=True,GT_read=False,time=time.time()));raise
