from common import *
from native_time_v2 import check,N
import subprocess,concurrent.futures,traceback

def run(ds):
 for stage in ['score','audit']:
  start=time.perf_counter();log=N/'logs'/(ds+'_'+stage+'.log');log.parent.mkdir(parents=True,exist_ok=True)
  with log.open('wb')as f:
   p=subprocess.Popen([str(R/'.conda/tubedetr/bin/python'),'-B',str(S/'native_time_v2.py'),stage,'--dataset',ds],cwd=R,env=os.environ.copy(),stdout=f,stderr=subprocess.STDOUT);w=dict(pid=p.pid,startticks=int(Path(f'/proc/{p.pid}/stat').read_text().split()[21]));atomic(N/'children'/(ds+'_'+stage+'.json'),w);atomic(N/(ds+'_state.json'),dict(status='running_'+stage,child=w));code=p.wait()
  atomic(N/'exits'/(ds+'_'+stage+'.json'),dict(worker=w,stage=stage,actual_subprocess_wait=True,exit_code=code,elapsed_wall_seconds=time.perf_counter()-start));assert code==0,(ds,stage,code)
 atomic(N/(ds+'_state.json'),dict(status='complete_correct_native_Source_and_reused_FullTime_PENDING_ROOT'))
if __name__=='__main__':
 check();assert read(D/'controller_actual_exit.json')['actual_tool_wait']and read(D/'controller_actual_exit.json')['exit_code']==0;assert not(N/'controller.json').exists();atomic(N/'controller.json',identity())
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=2)as ex:
   fs=[ex.submit(run,ds)for ds in COUNTS]
   for f in fs:f.result()
  atomic(N/'state.json',dict(status='both_domains_native_Source_fixed_FullTime_reused_PENDING_ROOT_publication'))
 except BaseException:atomic(N/'failure.json',dict(error=traceback.format_exc()));raise
