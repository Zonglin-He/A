"""One GPU worker, whole domain independent CPU stages, no full auto-release."""
from common import *
import traceback
def main():
 p=check();assert not(D/'controller.json').exists();atomic(D/'controller.json',dict(**identity(),plan_fingerprint=p['fingerprint']));success={}
 def spawn(name,argv,device):
  path=D/'logs'/(name+'.log');path.parent.mkdir(parents=True,exist_ok=True);f=path.open('xb');child=subprocess.Popen([str(PYTHON),'-B',*map(str,argv)],cwd=R,env={**os.environ,'CUDA_VISIBLE_DEVICES':device},stdout=f,stderr=subprocess.STDOUT,start_new_session=True);who=identity(child.pid);atomic(D/'children'/(name+'.json'),dict(**who,argv=list(map(str,argv))));return child,who,f,time.perf_counter(),path
 def collect(name,child):
  process,who,f,start,path=child;code=process.wait();f.close();e=dict(worker=who,actual_subprocess_wait=True,exit_code=code,elapsed_wall_seconds=time.perf_counter()-start,log_sha256=sha(path));atomic(D/'exits'/(name+'.json'),e);assert code==0,(name,code);success[name]=e
 try:
  gpu=spawn('matched_mask_GPU',[S/'worker.py'],'0');atomic(D/'state.json',dict(status='running_matched_mask128',GPU_worker=gpu[1],controller=identity()))
  for ds in ['hc2','vidstg']:
   while not(D/'propagation'/ds/'sealed.json').exists():
    code=gpu[0].poll();assert code is None or code==0,('GPU failed, preserve all paid state',code)
    if code==0:assert(D/'propagation'/ds/'sealed.json').exists()
    time.sleep(5)
   for task in ['input_audit','score','audit','export']:
    key=ds+'_'+task;atomic(D/'state.json',dict(status='running',CPU_task=task,dataset=ds,GPU_worker=gpu[1],controller=identity()));collect(key,spawn(key,[S/'report.py',task,'--dataset',ds],''))
  collect('matched_mask_GPU',gpu);atomic(D/'state.json',dict(status='both128_scored_audited_exported_PENDING_ROOT_not_published',actual_children=success,controller=identity(),no_automatic_full=True,other_scopes_still_USER_PAUSED=True));print('matched mask controls complete pending root',flush=True)
 except BaseException:atomic(D/'controller_failure.json',dict(error=traceback.format_exc(),success=success,all_paid_state_preserved=True,controller=identity()));raise
if __name__=='__main__':main()
