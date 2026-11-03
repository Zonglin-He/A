from study_common import *
import subprocess,concurrent.futures,traceback
def unit(task,ds):
 for stage in ['predict','score','audit']:
  key=task+'_'+ds+'_'+stage;cmd=[str(R/'.conda/tubedetr/bin/python'),'-B',str(S/'execute.py'),stage,'--task',task,'--dataset',ds];assert not(D/'exits'/(key+'.json')).exists();tick=time.perf_counter()
  with(D/'logs'/(key+'.log')).open('wb')as f:
   child=subprocess.Popen(cmd,cwd=R,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},stdout=f,stderr=subprocess.STDOUT);who=dict(pid=child.pid,startticks=int(Path(f'/proc/{child.pid}/stat').read_text().split()[21]));atomic(D/'children'/(key+'.json'),who);atomic(D/(task+'_'+ds+'_state.json'),dict(status='running_'+stage,child=who,controller=identity()));code=child.wait()
  rec=dict(task=task,dataset=ds,stage=stage,worker=who,actual_subprocess_wait=True,exit_code=code,elapsed_wall_seconds=time.perf_counter()-tick);atomic(D/'exits'/(key+'.json'),rec);assert code==0,rec
 atomic(D/(task+'_'+ds+'_state.json'),dict(status='whole_domain_scored_audited_PENDING_ROOT',controller=identity()))
if __name__=='__main__':
 validate();assert not(D/'controller.json').exists();(D/'logs').mkdir(parents=True,exist_ok=True);atomic(D/'controller.json',identity())
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=2)as executor:
   futures=[executor.submit(unit,t,ds)for t in ['combination','regions']for ds in COUNTS]
   for future in futures:future.result()
  atomic(D/'state.json',dict(status='all_two_controls_four_domains_scored_audited_PENDING_ROOT_analysis_publication',controller=identity(),new_Source=0,new_expert=0,new_model=0,new_GPU=0,new_optimizer=0))
 except BaseException:atomic(D/'failure.json',dict(error=traceback.format_exc(),controller=identity(),all_paid_and_predicted_state_preserved=True));raise
