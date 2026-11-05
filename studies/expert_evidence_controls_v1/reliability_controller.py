"""Source72 once, then independent no-target-label fitting and sealed gates."""
from common import *
from reliability_source_worker import C,check_source
import traceback
def main():
 plan=check_source();assert not(C/'controller.json').exists();atomic(C/'controller.json',dict(**identity(),plan_fingerprint=plan['fingerprint']));success={}
 stages=[('source72_GPU',S/'reliability_source_worker.py',[],'0')]+[(task,S/'reliability_study.py',[task],'')for task in ['source_input_audit','source_score','source_score_audit','fit','target_predict','target_input_audit','target_score','target_score_audit']]+[(task,S/'reliability_report.py',[task],'')for task in ['export','audit']]
 try:
  for name,path,argv,device in stages:
   log=C/'logs'/(name+'.log');log.parent.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
   with log.open('xb')as f:
    process=subprocess.Popen([str(PYTHON),'-B',str(path),*argv],cwd=R,env={**os.environ,'CUDA_VISIBLE_DEVICES':device},stdout=f,stderr=subprocess.STDOUT,start_new_session=True);who=identity(process.pid);atomic(C/'children'/(name+'.json'),dict(**who,argv=[str(path),*argv]));atomic(C/'state.json',dict(status='running',stage=name,child=who,controller=identity()));code=process.wait()
   e=dict(worker=who,actual_subprocess_wait=True,exit_code=code,elapsed_wall_seconds=time.perf_counter()-start,log_sha256=sha(log));atomic(C/'exits'/(name+'.json'),e);assert code==0,(name,code);success[name]=e
  atomic(C/'state.json',dict(status='source72_and_target256_reliability_scored_allarrays_selection_stats_audited_PENDING_ROOT_not_published',actual_children=success,controller=identity(),old_scopes_still_USER_PAUSED=True));print('reliability all stages complete pending ROOT',flush=True)
 except BaseException:atomic(C/'controller_failure.json',dict(error=traceback.format_exc(),success=success,all_paid_raw_and_logs_preserved=True,controller=identity()));raise
if __name__=='__main__':main()
