"""Only this new expert-subset study; never revive an old dispatcher."""
from common_v1 import *
import argparse,subprocess,traceback

def step(script,stage,expert,ds=None):
 cmd=[str(R/'.conda/tubedetr/bin/python'),'-B',str(S/script),stage,'--expert',expert]
 if ds:cmd+=['--dataset',ds]
 key='_'.join(x for x in [expert,ds,stage]if x);env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'if stage=='GPU'else'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'};start=time.perf_counter()
 with(D/'logs'/f'{key}.log').open('wb')as f:
  p=subprocess.Popen(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT);who=dict(pid=p.pid,startticks=int(Path(f'/proc/{p.pid}/stat').read_text().split()[21]));atomic(D/'state.json',dict(status='running_'+stage,expert=expert,dataset=ds,child=who,controller=identity(),all_old_paused_tasks_not_resumed=True));atomic(D/'children'/f'{key}.json',who);code=p.wait()
 rec=dict(stage=stage,expert=expert,dataset=ds,worker=who,actual_subprocess_wait=True,exit_code=code,elapsed_wall_seconds=time.perf_counter()-start);atomic(D/'exits'/f'{key}.json',rec);assert code==0,rec

def run():
 validate();assert not(D/'controller.json').exists();atomic(D/'controller.json',identity());start=time.perf_counter();(D/'logs').mkdir(exist_ok=True)
 try:
  # Existing two experts: only derive the newly defined pure baselines; every
  # previous Current prediction and official Current score is reused exactly.
  for expert in REUSE:
   step('worker_v1.py','reuse',expert)
   for ds in ['hc2','vidstg']:
    for stage in ['input_audit','score','audit','export']:step('evaluate_v1.py',stage,expert,ds)
  # Each new expert has a separate process/model lifetime and one GPU lease.
  for expert in EXPERTS:
   step('worker_v1.py','GPU',expert)
   for ds in ['hc2','vidstg']:
    for stage in ['input_audit','score','audit','export']:step('evaluate_v1.py',stage,expert,ds)
  atomic(D/'state.json',dict(status='all_four_experts_two128domains_scored_audited_exported_PENDING_ROOT_interpretation_publication',controller=identity(),elapsed_wall_seconds=time.perf_counter()-start,new_Source=0,new_original_expert=0,new_expert_cap=512,all_other_scopes_paused=True,paper_changed=False))
 except BaseException:atomic(D/'controller_failure.json',dict(error=traceback.format_exc(),controller=identity(),all_old_and_new_paid_state_preserved=True));raise
if __name__=='__main__':run()
