"""One new finite owner; no restoration of any previous controller."""
from common_v1 import *
import subprocess,traceback
def step(script,stage,expert,ds=None):
 cmd=[str(R/'.conda/tubedetr/bin/python'),'-B',str(S/script),stage,'--expert',expert];key=expert+'_'+(ds+'_'if ds else'')+stage
 if ds:cmd+=['--dataset',ds]
 assert not(D/'exits'/f'{key}.json').exists();tick=time.perf_counter()
 with(D/'logs'/f'{key}.log').open('wb')as f:
  child=subprocess.Popen(cmd,cwd=R,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'if stage=='GPU'else''},stdout=f,stderr=subprocess.STDOUT);who=dict(pid=child.pid,startticks=int(Path(f'/proc/{child.pid}/stat').read_text().split()[21]));atomic(D/'children'/f'{key}.json',who);atomic(D/'state.json',dict(status='running_'+stage,expert=expert,dataset=ds,child=who,controller=identity()));code=child.wait()
 receipt=dict(stage=stage,expert=expert,dataset=ds,worker=who,actual_subprocess_wait=True,exit_code=code,elapsed_wall_seconds=time.perf_counter()-tick);atomic(D/'exits'/f'{key}.json',receipt);assert code==0,receipt
def main():
 validate();assert not(D/'controller.json').exists();(D/'logs').mkdir(parents=True,exist_ok=True);atomic(D/'controller.json',identity())
 try:
  for expert in EXPERTS:
   step('worker_v1.py','GPU',expert)
   for ds in ['hc2','vidstg']:
    for stage in ['input_audit','score','audit','export']:step('evaluate_v1.py',stage,expert,ds)
  atomic(D/'state.json',dict(status='two_new_experts_two128domains_scored_audited_exported_PENDING_ROOT_analysis_publication',controller=identity(),new_Source=0,new_original_ReferDINO=0,new_optimizer=0,Current_hard_anchor_contract_changed=False,paper_changed=False))
 except BaseException:atomic(D/'controller_failure.json',dict(error=traceback.format_exc(),controller=identity(),all_paid_raw_and_steps_preserved=True));raise
if __name__=='__main__':main()
