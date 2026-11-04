from common import *
import subprocess,concurrent.futures,traceback
E=D/'runtime_v2'
def check_runtime():
 validate();p=read(D/'execution_v2.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for f,h in p['code_sha256'].items():assert sha(R/f)==h
 for ds in COUNTS:assert read(D/'exits'/(ds+'_predict.json'))['exit_code']==0 and read(D/ds/'sealed.json')['queries']==COUNTS[ds]
 outer=read(D/'controller_actual_exit.json');assert outer['actual_tool_wait']and outer['exit_code']==1
 return p

def run(ds):
 for stage,entry in [('score','score_runtime_v2.py'),('audit','audit.py')]:
  start=time.perf_counter();cmd=[str(R/'.conda/tubedetr/bin/python'),'-B',str(S/entry),'--dataset',ds];log=E/'logs'/(ds+'_'+stage+'.log');log.parent.mkdir(parents=True,exist_ok=True)
  with log.open('wb')as f:
   child=subprocess.Popen(cmd,cwd=R,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},stdout=f,stderr=subprocess.STDOUT);w=dict(pid=child.pid,startticks=int(Path(f'/proc/{child.pid}/stat').read_text().split()[21]));atomic(E/'children'/(ds+'_'+stage+'.json'),w);atomic(E/(ds+'_state.json'),dict(status='running_'+stage,child=w,controller=identity()));code=child.wait()
  atomic(E/'exits'/(ds+'_'+stage+'.json'),dict(dataset=ds,stage=stage,worker=w,actual_subprocess_wait=True,exit_code=code,elapsed_wall_seconds=time.perf_counter()-start));assert code==0,(ds,stage,code)
 atomic(E/(ds+'_state.json'),dict(status='whole_domain_scored_audited_PENDING_ROOT',controller=identity()))
if __name__=='__main__':
 check_runtime();assert not(E/'controller.json').exists();atomic(E/'controller.json',identity())
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=2)as ex:
   fs=[ex.submit(run,ds)for ds in COUNTS]
   for f in fs:f.result()
  atomic(E/'state.json',dict(status='both_full_domains_complete_PENDING_ROOT_analysis_publication',controller=identity(),original_prediction_seals_reused_without_reprediction=True))
 except BaseException:atomic(E/'failure.json',dict(error=traceback.format_exc()));raise
