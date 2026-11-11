"""Finite current-task stage chain. No timer, scheduler, retries or new search."""
from common69 import *
import subprocess,sys,time,datetime,argparse
H=ROOT/'research/round69';ap=argparse.ArgumentParser();ap.add_argument('--evidence-pid',type=int,required=True);args=ap.parse_args()
def state(phase):dump(H/'PIPELINE_STATE.json',{'phase':phase,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'finite_current_authorized_task':True,'no_new_method_or_search':True})
def run(resource,name,*argv):
 state(name+(' '+argv[0] if argv else ''));cmd=[sys.executable,str(ROOT/'tools/manual_r69_guard.py'),'--resource',resource,'--timeout-seconds','3600','--',sys.executable,str(H/name),*argv];print('STAGE69',name,*argv,flush=True);subprocess.run(cmd,cwd=ROOT,check=True)
try:
 state('waiting_for_existing_evidence_process_receipt')
 deadline=time.monotonic()+3600
 while True:
  match=[r for f in (H/'job_receipts').glob('*.json') if (r:=read(f))['pid']==args.evidence_pid]
  if match:assert len(match)==1 and match[0]['returncode']==0;break
  assert time.monotonic()<deadline;time.sleep(2)
 p=read(H/'experiment/DESIGN_LOCK.json');assert read(ROOT/p['fit_root']/'SEALED.json')['n']==2304
 for stage,name,res,done in [
  ('C_random','C_random69.py','gpu',ROOT/p['C_random_root']/'SEALED.json'),
  ('predict','predict69.py','cpu',ROOT/p['prediction_root']/'SEALED.json'),
  ('audit_gpu','audit_gpu69.py','gpu',H/'independent_gpu_audit_attempt001/AUDIT_RESULT.json'),
  ('audit_cpu','audit_cpu69.py','cpu',H/'independent_audit_attempt001/AUDIT_RESULT.json')]:
  if done.exists():continue
  plan=H/'experiment'/(stage.upper()+'_PLAN_LOCK.json')
  if not plan.exists():run('cpu','stage69.py',stage)
  run(res,name)
 if not read(H/'STATE.json')['CPU_scoring_authorized']:run('cpu','release_scoring69.py')
 for stage,name,done in [('score','score69.py',H/'score_all2304_attempt001/SEALED.json'),('scalar','audit_scalar69.py',H/'independent_scalar_audit_attempt001/AUDIT_RESULT.json')]:
  if done.exists():continue
  plan=H/'experiment'/(stage.upper()+'_PLAN_LOCK.json')
  if not plan.exists():run('cpu','stage69.py',stage)
  run('cpu',name)
 if not (H/'ANALYSIS_RECEIPT.json').exists():run('cpu','finish_analysis69.py')
 state('fixed_experiments_audits_and_analysis_complete_publication_pending')
except BaseException as e:
 dump(H/'PIPELINE_FAILED.json',{'error_type':type(e).__name__,'error':str(e),'no_automatic_retry_or_method_change':True});raise
