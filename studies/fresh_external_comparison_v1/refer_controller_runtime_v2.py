"""One explicitly frozen recovery execution, no automatic paid retry."""
from refer_runtime_v2 import *
import subprocess,traceback,fcntl,signal
HERE=Path(__file__).parent
def risk_pause(stage):
 pid=1318498;start=257114336;p=Path('/proc')/str(pid)/'stat'
 if not p.exists():return None
 s=p.read_text().split(') ',1)[1].split()
 if int(s[19])!=start:return None
 assert os.getpgid(pid)==pid;os.killpg(pid,signal.SIGSTOP);atomic(V2/(stage+'_own_risk_CPU_pause.json'),dict(pid=pid,startticks=start,pgid=pid,time=time.time(),only_own_temporal_risk_group=True));return pid,start
def risk_resume(record,stage):
 if record is None:return
 pid,start=record;p=Path('/proc')/str(pid)/'stat'
 if p.exists()and int(p.read_text().split(') ',1)[1].split()[19])==start:
  os.killpg(pid,signal.SIGCONT);atomic(V2/(stage+'_own_risk_CPU_resume.json'),dict(pid=pid,startticks=start,SIGCONT_sent=True,time=time.time()))
def run():
 validate_v2();require_release();assert read(V2/'preflight_v2.json')['worker_numeric_code_object_unchanged'];V2.mkdir(parents=True,exist_ok=True);lock=(V2/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not(V2/'controller_v2.json').exists();who=identity();atomic(V2/'controller_v2.json',who);start=time.perf_counter()
 def child(stage,command,args,device):
  reserve=risk_pause(stage)if device==''else None
  try:
   with(V2/(stage+'.log')).open('xb')as log:
    c=subprocess.Popen(['taskset','-c','14-15','nice','-n','19','ionice','-c','3',sys.executable,'-B',str(HERE/'refer_runtime_v2.py'),command,*args],env={**os.environ,'CUDA_VISIBLE_DEVICES':device,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},stdout=log,stderr=subprocess.STDOUT);atomic(V2/(stage+'_child.json'),identity(c.pid));atomic(V2/'state_v2.json',dict(status=stage,controller=who));code=c.wait()
   atomic(V2/(stage+'_exit.json'),dict(code=code));assert code==0,stage+': preserve paid outputs, no automatic retry'
  finally:risk_resume(reserve,stage)
 try:
  child('GPU','GPU',[],'0')
  for ds in ['hc2','vidstg']:child(ds+'_input_audit','input_audit',['--dataset',ds],'')
  for ds in ['hc2','vidstg']:
   for stage in ['score','audit']:child(ds+'_'+stage,stage,['--dataset',ds],'')
  atomic(V2/'state_v2.json',dict(status='all256_newexpert_small_comparison_scored_audited_pending_root_analysis_cost_publication',controller=who,elapsed_wall_seconds=time.perf_counter()-start,full12205_complete=False,earlier_failed_paid_forward=1))
 except BaseException:atomic(V2/'state_v2.json',dict(status='failed_preserved_no_automatic_paid_retry',controller=who,error=traceback.format_exc(),elapsed_wall_seconds=time.perf_counter()-start));raise
if __name__=='__main__':run()
