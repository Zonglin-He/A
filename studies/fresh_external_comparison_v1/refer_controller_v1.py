"""Single non-retrying alternative-expert owner, after actual Edge root publication."""
from refer_common_v1 import *
import subprocess,traceback,fcntl
HERE=Path(__file__).parent
def run():
 validate_pilot();assert read(N/'refer_math_preflight_v1.json')['status']=='passed_four_prepared_current_expert_full_chain_repeat_exact';D.mkdir(parents=True,exist_ok=True);lock=(D/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not(D/'controller_v1.json').exists();who=identity();atomic(D/'controller_v1.json',who);start=time.perf_counter()
 def child(stage,entry,args,device):
  with(D/(stage+'.log')).open('xb')as log:
   c=subprocess.Popen(['taskset','-c','14-15','nice','-n','19','ionice','-c','3',sys.executable,'-B',str(HERE/entry),*args],env={**os.environ,'CUDA_VISIBLE_DEVICES':device,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},stdout=log,stderr=subprocess.STDOUT);atomic(D/(stage+'_child.json'),identity(c.pid));atomic(D/'state_v1.json',dict(status=stage,controller=who));code=c.wait()
  atomic(D/(stage+'_exit.json'),dict(code=code));assert code==0,stage+': preserve paid outputs/attempt; no automatic retry'
 try:
  atomic(D/'state_v1.json',dict(status='waiting_actual_Edge_whole_audits_root_analysis_ALLremote_publication_release',controller=who,new_model=0,new_expert=0,new_Source=0))
  while not(D/'root_predecessor_release_v1.json').exists():
   resources(start);s=N/'edgetam_v1/state_v1.json'
   if s.exists():assert not read(s)['status'].startswith('failed'),'Actual predecessor failure; preserve this zero-work owner'
   time.sleep(15)
  require_release();validate_pilot();child('GPU','refer_worker_v1.py',[],'0')
  for ds in ['hc2','vidstg']:child(ds+'_input_audit','refer_root_v1.py',['audit','--dataset',ds],'')
  for ds in ['hc2','vidstg']:
   for stage in ['score','audit']:child(ds+'_'+stage,'refer_evaluation_v1.py',[stage,'--dataset',ds],'')
  atomic(D/'state_v1.json',dict(status='all256_newexpert_small_comparison_scored_audited_pending_root_analysis_actualcost_and_publication',controller=who,elapsed_wall_seconds=time.perf_counter()-start,full12205_complete=False))
 except BaseException:atomic(D/'state_v1.json',dict(status='failed_preserved_no_automatic_paid_retry',controller=who,error=traceback.format_exc(),elapsed_wall_seconds=time.perf_counter()-start));raise
if __name__=='__main__':run()
