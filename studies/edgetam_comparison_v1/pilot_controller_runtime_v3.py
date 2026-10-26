"""One waiting owner, after active CG only; explicit root priority release."""
from pilot_execution_v3 import *
from pilot_gate_v3 import gate
import subprocess,traceback,fcntl
def run():
 validate_execution_v3();lock=(D/'pilot_controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not(D/'pilot_controller_v3.json').exists();who=identity();atomic(D/'pilot_controller_v3.json',who)
 try:
  atomic(D/'pilot_controller_state_v3.json',dict(status='waiting_CG_user_pause_owned_exit_and_cached_small_priority_release',controller=who,new_model=0,new_Source=0,new_original_expert=0))
  while True:
   dependency=gate()
   if dependency is not None:break
   time.sleep(15)
  atomic(D/'pilot_dependency_gate_v3.json',dependency);validate_execution_v3()
  with(D/'pilot_GPU_v3.log').open('ab')as log:
   c=subprocess.Popen(['taskset','-c','14-15','nice','-n','19','ionice','-c','3',sys.executable,'-B',str(S/'pilot_worker_runtime_v3.py')],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'});atomic(D/'pilot_GPU_child_v3.json',identity(c.pid));atomic(D/'pilot_controller_state_v3.json',dict(status='actual_fixed256_EdgeTAM_GPU_queries',controller=who));code=c.wait()
  atomic(D/'pilot_GPU_exit_v3.json',dict(code=code));assert code==0,'Preserve raw/attempts, no automatic paid retry'
  # Scoring and then auditing run in independent CPU processes. The small
  # fixed cohort and checkpoint/rules remain unchanged by these scores.
  for ds in ['hc2','vidstg']:
   for stage in ['score','audit']:
    with(D/f'pilot_{ds}_{stage}.log').open('ab')as log:
     c=subprocess.Popen(['taskset','-c','14-15','nice','-n','19','ionice','-c','3',sys.executable,'-B',str(S/'pilot_report_runtime_v3.py'),stage,'--dataset',ds],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''});atomic(D/f'pilot_{ds}_{stage}_child.json',identity(c.pid));atomic(D/'pilot_controller_state_v3.json',dict(status=stage,dataset=ds,controller=who));code=c.wait()
    atomic(D/f'pilot_{ds}_{stage}_exit.json',dict(code=code));assert code==0
  atomic(D/'pilot_controller_state_v3.json',dict(status='fixed256_EdgeTAM_scored_audited_pending_root_all_signed_cost_analysis_publication',controller=who,full12205_complete=False))
 except BaseException:atomic(D/'pilot_controller_state_v3.json',dict(status='failed_preserved',error=traceback.format_exc(),controller=who));raise
if __name__=='__main__':run()
