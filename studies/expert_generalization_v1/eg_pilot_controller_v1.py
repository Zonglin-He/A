"""Waiting is not model execution; one non-retrying pilot owner after root gate."""
from eg_common_v1 import *
from eg_pilot_v1 import validate_pilot
from eg_pilot_gate_v1 import gate
import subprocess,traceback,fcntl

def run():
 validate_pilot();lock=(D/'pilot_controller_v1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not(D/'pilot_controller_v1.json').exists();who=identity();atomic(D/'pilot_controller_v1.json',who)
 try:
  atomic(D/'pilot_controller_state_v1.json',dict(status='waiting_actual_CG_gamma_same_expert_TTA_exits_audits_root_publication_and_release',controller=who,new_model=0,new_Source=0,new_expert=0,full_experiment_complete=False))
  while True:
   value=gate()
   if value is not None:break
   time.sleep(15)
  validate_pilot();atomic(D/'pilot_actual_predecessor_gate_v1.json',value)
  with(D/'pilot_GPU_v1.log').open('ab')as log:
   p=subprocess.Popen(['taskset','-c','14-15','nice','-n','19','ionice','-c','3',sys.executable,'-B',str(S/'eg_pilot_v1.py')],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'})
   atomic(D/'pilot_GPU_child_v1.json',identity(p.pid));atomic(D/'pilot_controller_state_v1.json',dict(status='actual_alternative_expert_fixed256_GPU_pilot',controller=who));code=p.wait()
  atomic(D/'pilot_GPU_exit_v1.json',dict(code=code));assert code==0,'Preserve every paid attempt; no automatic retry'
  atomic(D/'pilot_controller_state_v1.json',dict(status='pilot_GPU_exited_success_pending_independent_audit_cost_feasibility_and_full_release',controller=who,full_experiment_complete=False))
 except BaseException:atomic(D/'pilot_controller_state_v1.json',dict(status='failed_preserved',controller=who,error=traceback.format_exc()));raise
if __name__=='__main__':run()
