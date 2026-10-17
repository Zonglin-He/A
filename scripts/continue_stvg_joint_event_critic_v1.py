"""One bounded serial pilot; no retry loop or historical queue revival."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import BASE,put,read,sha,status,verify

def main():
 verify();put(BASE/'EXECUTION_RUNTIME001.json',dict(status='active',controller_sha256=sha(__file__),scientific_runtime_sha256=sha(BASE/'RUNTIME_LOCK.json'),deadline_epoch=read(BASE/'PLAN_LOCK.json')['deadline_epoch'],stages=['target','headroom','source','source_labels','train','infer','score'],serial=True,no_auto_full_stream=True))
 for stage in ['target','headroom','source','source_labels','train','infer','score']:
  verify();status(stage,0,1)
  with (BASE/('worker_'+stage+'.log')).open('xb')as log:
   subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B',str(ROOT/'scripts/run_stvg_joint_event_critic_v1.py'),stage],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
  if stage=='headroom' and not read(BASE/'CANDIDATE_HEADROOM.json')['continue_source_pilot']:
   put(BASE/'DECISION.json',dict(status='stop_no_candidate_headroom',scope='finite six-hypothesis pilot',source_training_not_started=True,CURRENT_unchanged=True));break
 put(BASE/'ROOT_HANDOFF.json',dict(status='ready',count=32,scientific_decision=read(BASE/'DECISION.json')['status'],independent_root_pixel_publication_required=True,matched_OPD_required=read(BASE/'DECISION.json').get('matched_OPD')=='required_next'))
 status('root_audit_publication_pending',32,32)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  put(BASE/('CONTROLLER_FAILURE_'+str(time.time_ns())+'.json'),dict(status='failed',error_type=type(e).__name__,error=str(e),partial_preserved=True));raise
