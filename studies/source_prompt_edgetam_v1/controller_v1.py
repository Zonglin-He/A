"""Source-only EdgeTAM single GPU, after the already active ReferFormer finishes."""
from common import *
import subprocess,traceback,fcntl,signal
def live(w):
 p=Path('/proc')/str(w['pid'])/'stat';return p.exists()and int(p.read_text().split(') ',1)[1].split()[19])==w['startticks']
def run():
 validate();lock=(D/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not(D/'controller_v1.json').exists();who=identity();atomic(D/'controller_v1.json',who);start=time.perf_counter()
 def child(stage,entry,args,device):
  own=dict(pid=1318498,startticks=257114336);pause=device==''and live(own)
  if pause:assert os.getpgid(own['pid'])==own['pid'];os.killpg(own['pid'],signal.SIGSTOP);atomic(D/(stage+'_CPU_pause.json'),dict(**own,time=time.time()))
  try:
   with(D/(stage+'.log')).open('xb')as log:
    c=subprocess.Popen(['taskset','-c','14-15','nice','-n','19','ionice','-c','3',sys.executable,'-B',str(HERE/entry),*args],env={**os.environ,'CUDA_VISIBLE_DEVICES':device,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},stdout=log,stderr=subprocess.STDOUT);atomic(D/(stage+'_child.json'),identity(c.pid));atomic(D/'state_v1.json',dict(status=stage,controller=who));code=c.wait()
   atomic(D/(stage+'_exit.json'),dict(code=code));assert code==0,stage+': preserve paid outputs; no automatic retry'
  finally:
   if pause and live(own):os.killpg(own['pid'],signal.SIGCONT);atomic(D/(stage+'_CPU_resume.json'),dict(**own,time=time.time(),SIGCONT_sent=True))
 try:
  atomic(D/'state_v1.json',dict(status='waiting_actual_active_ReferFormer_whole_audits_and_owned_exit',controller=who,new_model=0,new_tracking=0))
  ref=R/'outputs/fresh_external_comparison_v1/referformer_v4'
  while True:
   resources(start);q=read(ref/'state_v4.json');assert not q['status'].startswith('failed'),'Preserve zero-work waiter after actual predecessor failure'
   if q['status']=='all256_newexpert_small_comparison_scored_audited_pending_root_analysis_cost_publication'and not live(read(ref/'controller_v4.json')):break
   time.sleep(15)
  validate();atomic(D/'actual_predecessor_release_v1.json',release());child('GPU','worker_v1.py',[],'0')
  for ds in ['hc2','vidstg']:
   for stage in ['score','audit']:child(ds+'_'+stage,'report_v1.py',[stage,'--dataset',ds],'')
  atomic(D/'state_v1.json',dict(status='both_fixed128_single_expert_Source_prompt_EdgeTAM_scored_audited_pending_root_analysis_publication',controller=who,elapsed_wall_seconds=time.perf_counter()-start))
 except BaseException:atomic(D/'state_v1.json',dict(status='failed_preserved_no_automatic_paid_retry',error=traceback.format_exc(),controller=who));raise
if __name__=='__main__':run()
