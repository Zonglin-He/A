from common import *
from runtime_v2 import execution
import subprocess,fcntl,traceback

def run():
 p=execution();lock=(D/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not(D/'controller_v2.json').exists();who=identity();atomic(D/'controller_v2.json',who)
 try:
  assert read(D/'preserved_serial_runtime_v1/owned_exit.json')['actual_owned_exit_verified'];assert read(D/'hc2/reporting_receipt.json')['queries']==1902
  for command in ['predict','score','audit']:
   stage='vidstg_'+command;atomic(D/'state_v2.json',dict(status=stage,controller=who))
   with(D/(stage+'_v2.log')).open('xb')as log:
    c=subprocess.Popen(['taskset','-c','14-15','nice','-n','19','ionice','-c','3',sys.executable,'-B',str(S/'runtime_v2.py'),command,'--dataset','vidstg'],env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},stdout=log,stderr=subprocess.STDOUT);atomic(D/(stage+'_child_v2.json'),identity(c.pid));code=c.wait()
   atomic(D/(stage+'_exit_v2.json'),dict(code=code));assert code==0,stage
  atomic(D/'state_v2.json',dict(status='both_full_domains_geometric_controls_scored_audited_pending_root_analysis_publication',controller=who))
 except BaseException:atomic(D/'failure_v2.json',dict(error=traceback.format_exc(),worker=who));raise
if __name__=='__main__':run()
