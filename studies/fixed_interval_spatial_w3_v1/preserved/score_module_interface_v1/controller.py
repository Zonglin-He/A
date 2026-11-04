from common import *
import subprocess,concurrent.futures,traceback
def run(ds):
 for stage,entry,args in [('predict','run.py',['predict','--dataset',ds]),('score','score.py',['--dataset',ds]),('audit','audit.py',['--dataset',ds])]:
  start=time.perf_counter();cmd=[str(R/'.conda/tubedetr/bin/python'),'-B',str(S/entry),*args];log=D/'logs'/(ds+'_'+stage+'.log');log.parent.mkdir(parents=True,exist_ok=True)
  with log.open('wb')as f:
   p=subprocess.Popen(cmd,cwd=R,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},stdout=f,stderr=subprocess.STDOUT);w=dict(pid=p.pid,startticks=int(Path(f'/proc/{p.pid}/stat').read_text().split()[21]));atomic(D/'children'/(ds+'_'+stage+'.json'),w);atomic(D/(ds+'_state.json'),dict(status='running_'+stage,child=w,controller=identity()));code=p.wait()
  atomic(D/'exits'/(ds+'_'+stage+'.json'),dict(dataset=ds,stage=stage,worker=w,actual_subprocess_wait=True,exit_code=code,elapsed_wall_seconds=time.perf_counter()-start));assert code==0,(ds,stage,code)
 atomic(D/(ds+'_state.json'),dict(status='full_fixed_interval_spatial_controls_scored_audited_PENDING_ROOT',controller=identity()))
if __name__=='__main__':
 validate();assert not(D/'controller.json').exists();atomic(D/'controller.json',identity())
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=2)as ex:
   fs=[ex.submit(run,ds)for ds in COUNTS]
   for f in fs:f.result()
  atomic(D/'state.json',dict(status='both_full_domains_PENDING_ROOT_analysis_publication',controller=identity(),new_Source=0,new_expert=0,new_GPU=0,new_model=0))
 except BaseException:atomic(D/'failure.json',dict(error=traceback.format_exc(),all_source_rawpredictions_logs_preserved=True));raise
