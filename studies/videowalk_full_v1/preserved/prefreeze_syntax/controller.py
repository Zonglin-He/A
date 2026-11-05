"""Single new full-list controller, never launches any historical queue."""
from common import *
import traceback
def main():
 p=check();assert not(D/'controller.json').exists();atomic(D/'controller.json',dict(**identity(),plan_fingerprint=p['fingerprint']));success={}
 def stage(name,path,args,device):
  file=D/'logs'/(name+'.log');file.parent.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
  with file.open('xb')as f:
   child=subprocess.Popen([str(PYTHON),'-B',str(path),*args],cwd=R,env={**os.environ,'CUDA_VISIBLE_DEVICES':device},stdout=f,stderr=subprocess.STDOUT,start_new_session=True);who=identity(child.pid);atomic(D/'children'/(name+'.json'),dict(**who,argv=[str(path),*args]));atomic(D/'state.json',dict(status='running',stage=name,worker=who,controller=identity()));code=child.wait()
  e=dict(worker=who,actual_subprocess_wait=True,exit_code=code,elapsed_wall_seconds=time.perf_counter()-start,log_sha256=sha(file));atomic(D/'exits'/(name+'.json'),e);assert code==0,(name,code);success[name]=e
 try:
  for ds in COUNTS:
   stage(ds+'_GPU',S/'worker.py',['--dataset',ds],'0')
   for task in ['whole_input_audit','score','score_audit','export','report_audit']:stage(ds+'_'+task,S/'evaluate.py',[task,'--dataset',ds],'')
   atomic(D/(ds+'_complete.json'),dict(status='whole_domain_scored_allgrid_allarray_pixel_score_statistics_audited_PENDING_ROOT',report_sha256=sha(P/ds/'report.json'),worker=identity()))
  atomic(D/'state.json',dict(status='both_full_domains_complete_scored_audited_PENDING_ROOT_publication',actual_children=success,controller=identity(),no_other_scope_released=True));print('VideoWalk full lists complete pending ROOT/publication',flush=True)
 except BaseException:atomic(D/'controller_failure.json',dict(error=traceback.format_exc(),success=success,all_paid_state_preserved=True,controller=identity()));raise
if __name__=='__main__':main()
