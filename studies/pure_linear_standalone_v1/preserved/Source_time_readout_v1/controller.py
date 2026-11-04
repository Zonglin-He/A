from common import *
import subprocess,concurrent.futures,traceback

def run(ds):
 for stage,entry,args in [('predict','run.py',['predict','--dataset',ds]),('score','score.py',['--dataset',ds]),('audit','audit.py',['--dataset',ds])]:
  start=time.perf_counter();cmd=[str(R/'.conda/tubedetr/bin/python'),'-B',str(S/entry),*args];log=D/'logs'/(ds+'_'+stage+'.log');log.parent.mkdir(parents=True,exist_ok=True)
  with log.open('wb')as f:
   c=subprocess.Popen(cmd,cwd=R,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},stdout=f,stderr=subprocess.STDOUT);w=dict(pid=c.pid,startticks=int(Path(f'/proc/{c.pid}/stat').read_text().split()[21]));atomic(D/'children'/(ds+'_'+stage+'.json'),w);atomic(D/(ds+'_state.json'),dict(status='running_'+stage,child=w));code=c.wait()
  atomic(D/'exits'/(ds+'_'+stage+'.json'),dict(worker=w,stage=stage,actual_subprocess_wait=True,exit_code=code,elapsed_wall_seconds=time.perf_counter()-start));assert code==0,(ds,stage,code)
 atomic(D/(ds+'_state.json'),dict(status='complete_scored_audited_PENDING_ROOT'))
if __name__=='__main__':
 validate();assert not(D/'controller.json').exists();atomic(D/'controller.json',identity())
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=2)as ex:
   fs=[ex.submit(run,ds)for ds in COUNTS]
   for f in fs:f.result()
  atomic(D/'state.json',dict(status='both_domains_complete_PENDING_ROOT_publication',new_Source=0,new_expert=0,new_GPU=0,new_model=0))
 except BaseException:atomic(D/'failure.json',dict(error=traceback.format_exc(),all_paid_source_and_previous_results_unchanged=True));raise
