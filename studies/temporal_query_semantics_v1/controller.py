from common import *
import subprocess,traceback
if __name__=='__main__':
 validate();assert not(D/'controller.json').exists();atomic(D/'controller.json',identity());(D/'logs').mkdir(exist_ok=True)
 try:
  for stage,script,args in [('predict','run.py',['predict']),('evaluate','run.py',['evaluate']),('audit','audit.py',[])]:
   cmd=[str(R/'.conda/tubedetr/bin/python'),'-B',str(S/script),*args];start=time.perf_counter()
   with(D/'logs'/(stage+'.log')).open('wb')as f:
    p=subprocess.Popen(cmd,cwd=R,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},stdout=f,stderr=subprocess.STDOUT);who=dict(pid=p.pid,startticks=int(Path(f'/proc/{p.pid}/stat').read_text().split()[21]));atomic(D/'state.json',dict(status='running_'+stage,child=who,controller=identity()));atomic(D/'children'/(stage+'.json'),who);code=p.wait()
   atomic(D/'exits'/(stage+'.json'),dict(stage=stage,worker=who,actual_subprocess_wait=True,exit_code=code,elapsed_wall_seconds=time.perf_counter()-start));assert code==0,(stage,code)
  atomic(D/'state.json',dict(status='all32pairs64queries_scored_independently_audited_PENDING_ROOT_analysis_publication',controller=identity(),new_Source=0,new_expert=0,new_GPU=0,new_model=0,new_optimizer=0))
 except BaseException:atomic(D/'failure.json',dict(error=traceback.format_exc(),worker=identity(),all_sources_inputs_predictions_logs_preserved=True));raise
