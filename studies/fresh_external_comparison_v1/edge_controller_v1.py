"""Single owner: whole256 prepared audit, exact16 inheritance, GPU, CPU score/audit."""
from edge_common_v1 import *
import subprocess,traceback,fcntl
HERE=Path(__file__).parent
def run():
 validate();D.mkdir(parents=True,exist_ok=True);lock=(D/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not(D/'controller_v1.json').exists();who=identity();atomic(D/'controller_v1.json',who);start=time.perf_counter()
 def child(stage,entry,args,device):
  with(D/(stage+'.log')).open('xb')as log:
   c=subprocess.Popen(['taskset','-c','14-15','nice','-n','19','ionice','-c','3',sys.executable,'-B',str(HERE/entry),*args],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':device,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'});atomic(D/(stage+'_child.json'),identity(c.pid));atomic(D/'state_v1.json',dict(status=stage,controller=who));code=c.wait()
  atomic(D/(stage+'_exit.json'),dict(code=code));assert code==0,stage+': preserve paid attempt/raw and fail; no automatic retry'
 try:
  atomic(D/'state_v1.json',dict(status='waiting_independent_whole256_current_input_audit',controller=who,new_model=0,new_Source=0,new_expert=0))
  while not(N/'current_audit_completion_v1.json').exists():
   resources(start);assert not(N/'current_audit_failure_v1.json').exists();time.sleep(15)
  while True:
   x=read(N/'current_audit_worker_v1.json');st=Path('/proc')/str(x['pid'])/'stat'
   if not st.exists()or int(st.read_text().split(') ',1)[1].split()[19])!=x['startticks']:break
   resources(start);time.sleep(2)
  atomic(D/'actual_predecessor_release_v1.json',release());validate();child('inheritance','edge_inheritance_v1.py',[],'');child('GPU','edge_worker_v1.py',[],'0')
  for ds in ['hc2','vidstg']:
   for stage in ['score','audit']:child(ds+'_'+stage,'edge_report_v1.py',[stage,'--dataset',ds],'')
  atomic(D/'state_v1.json',dict(status='fixed256_EdgeTAM_whole_scored_audited_pending_root_signed_cost_analysis_publication',controller=who,elapsed_wall_seconds=time.perf_counter()-start,full12205_complete=False))
 except BaseException:
  atomic(D/'state_v1.json',dict(status='failed_preserved_no_automatic_paid_retry',controller=who,error=traceback.format_exc(),elapsed_wall_seconds=time.perf_counter()-start));raise
if __name__=='__main__':run()
