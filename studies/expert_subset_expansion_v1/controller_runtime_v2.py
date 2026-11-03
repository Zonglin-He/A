"""Continue only unstarted MTTR after zero-model process-identity startup failure."""
from common_v1 import *
from worker_runtime_v2 import validate_execution
import subprocess,traceback
B=D/'runtime_v2'
def step(script,stage,ds=None):
 cmd=[str(R/'.conda/tubedetr/bin/python'),'-B',str(S/script),stage,'--expert','mttr'];key='mttr_'+(ds+'_'if ds else'')+stage
 if ds:cmd+=['--dataset',ds]
 start=time.perf_counter()
 with(B/'logs'/f'{key}.log').open('wb')as f:
  p=subprocess.Popen(cmd,cwd=R,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'if stage=='GPU'else''},stdout=f,stderr=subprocess.STDOUT);w=dict(pid=p.pid,startticks=int(Path(f'/proc/{p.pid}/stat').read_text().split()[21]));atomic(B/'state.json',dict(status='running_'+stage,expert='mttr',dataset=ds,child=w,controller=identity()));atomic(B/'children'/f'{key}.json',w);code=p.wait()
 e=dict(stage=stage,expert='mttr',dataset=ds,worker=w,actual_subprocess_wait=True,exit_code=code,elapsed_wall_seconds=time.perf_counter()-start);atomic(B/'exits'/f'{key}.json',e);assert code==0,e
if __name__=='__main__':
 validate_execution();assert not(B/'controller.json').exists();assert read(D/'controller_actual_exit.json')['exit_code']==1;assert not(D/'mttr').exists();previous=[]
 for p in (D/'exits').glob('*.json'):
  e=read(p)
  if e['expert']=='mttr':assert e['stage']=='GPU'and e['exit_code']==1;continue
  assert e['exit_code']==0 and e['actual_subprocess_wait'];previous.append(dict(path=str(p.relative_to(R)),sha256=sha(p)))
 assert len(previous)==27;atomic(B/'reused_completed27_actual_exits.json',previous);atomic(B/'controller.json',identity());(B/'logs').mkdir(parents=True,exist_ok=True)
 try:
  step('worker_runtime_v2.py','GPU')
  for ds in ['hc2','vidstg']:
   for stage in ['input_audit','score','audit','export']:step('evaluate_v1.py',stage,ds)
  atomic(B/'state.json',dict(status='all_four_experts_two128domains_scored_audited_exported_PENDING_ROOT_interpretation_publication',controller=identity(),reused_completed_stages=27,new_successful_stages=9,prior_zero_model_failed_startup_preserved=True,new_Source=0,new_original_expert=0,old_paused_scopes_not_resumed=True))
 except BaseException:atomic(B/'failure.json',dict(error=traceback.format_exc(),controller=identity(),all_paid_state_preserved=True));raise
