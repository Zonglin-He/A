"""Finite after-search tail: freeze -> deletions -> temporal -> native-clock handoff.

This process blocks on the original search pidfd, never signals its controller,
and never starts inference without its completed 42-slot prediction/readback seal.
"""
from pathlib import Path
import sys,copy,time,subprocess,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_component_ablation_common_v1 import *
PY=ROOT/'.conda/tubedetr/bin/python'
def pinfile(p):return {str(Path(p).relative_to(ROOT)):sha(p)}
def freeze():
 verify_preparation();p=SEARCH/'SEARCH_COMPLETION006.json';s=read(p);assert s['status']=='complete_pending_independent_root_view_public_archive'and len(s['history'])==len(set(s['history']))==42
 pins=copy.deepcopy(read(BASE/'PREPARATION_RUNTIME.json')['pins']);pins.update(pinfile(BASE/'PREPARATION_RUNTIME.json'));protected=copy.deepcopy(read(SEARCH/'PREPARATION_RUNTIME.json')['protected']);pins.update(pinfile(p));actual={}
 for trial in s['history']:
  b=SEARCH/'trials'/trial
  for n in ['CPU_COMPLETION.json','ROOT_READBACK.json','GLOBAL_PREDICTION_BARRIER.json','STUDENT_GLOBAL_BARRIER.json','DESIGN_LOCK.json','RUNTIME.json']:
   pins.update(pinfile(b/n))
  assert read(b/'CPU_COMPLETION.json')['status']=='complete'and read(b/'ROOT_READBACK.json')['status']=='pass'
  sb=read(b/'STUDENT_GLOBAL_BARRIER.json');rr=read(b/'ROOT_READBACK.json');gb=read(b/'GLOBAL_PREDICTION_BARRIER.json');cc=read(b/'CPU_COMPLETION.json')
  assert sb['status']=='complete'and sb['GT_read']is False and sb['count']==128 and len(sb['files'])==len({x['path']for x in sb['files']})==128
  assert rr['count']==128 and rr['GT_read']is False and rr['barrier_sha256']==sha(b/'STUDENT_GLOBAL_BARRIER.json')
  assert gb['status']=='complete'and gb['GT_read']is False and gb['count']==128
  assert cc['bindings']['prediction_seal_sha256']==sha(b/'GLOBAL_PREDICTION_BARRIER.json')
 for ds in ('hc2','vidstg'):
  trial=s['best_trial'][ds]['M2'];seen=set()
  while True:
   assert trial not in seen;seen.add(trial);b=SEARCH/'trials'/trial;d=read(b/'DESIGN_LOCK.json')
   assert d['configs'][ds]['M2']==s['configs'][ds]['M2']and d['response_settings'][ds]['M2']==s['response_settings'][ds]['M2']
   prior=d['family_reuse_from'][ds]['M2']
   if prior is None:break
   trial=prior
  actual[ds]=trial
  pins.update(pinfile(b/'POSTSEAL_SCORE_ROWS.json'));pins.update(pinfile(b/'POSTSEAL_SCORE_SUMMARY.json'))
  for entry in read(b/'STUDENT_GLOBAL_BARRIER.json')['files']:
   assert sha(ROOT/entry['path'])==entry['sha256'];pins[entry['path']]=entry['sha256'];pins[entry['receipt_path']]=entry['receipt_sha256']
 for n in ['ROSTER_LOCK.json','REFERENCE_LOCK.json','TEMPORAL_LOCK.json','SOURCE_CAPTURE_BARRIER.json']:pins.update(pinfile(SEARCH/n))
 f=dict(schema='stvg-component-ablation-final-parameter-lock-v1',status='locked',count=128,search_finished=True,roster_sha256=sha(SEARCH/'ROSTER_LOCK.json'),search_completion_sha256=sha(p),configs={ds:s['configs'][ds]['M2']for ds in actual},response_settings={ds:s['response_settings'][ds]['M2']for ds in actual},actual_baseline_trial=actual,pins=pins,protected=protected,selection_used_development_GT=True,GT_read=False,ablation_retuning_allowed=False,bootstrap_executed=False,CURRENT_promoted=False)
 write(BASE/'FINAL_PARAMETER_LOCK.json',f);return f

def prepare_trial(tid,variant,flags,final):
 b=BASE/'trials'/tid;b.mkdir(parents=True,exist_ok=False)
 d=dict(status='locked',trial_id=tid,variant=variant,ablation_flags=flags,scope='one matched component deletion; all128 exposed development; fixed original QA-clock reference',configs={ds:{'M2':final['configs'][ds]}for ds in final['configs']},response_settings={ds:{'M1':final['response_settings'][ds],'M2':final['response_settings'][ds]}for ds in final['configs']},family_reuse_from={ds:{'M2':final['actual_baseline_trial'][ds]if variant=='baseline'else None}for ds in final['configs']},GT_read=False)
 write(b/'DESIGN_LOCK.json',d);pins=copy.deepcopy(final['pins']);pins.update(pinfile(BASE/'FINAL_PARAMETER_LOCK.json'));pins.update(pinfile(b/'DESIGN_LOCK.json'))
 write(b/'RUNTIME.json',dict(status='locked',trial_id=tid,pins=pins,protected=final['protected'],GT_read=False))

def run(script,*args):subprocess.run([str(PY),'-B','scripts/'+script,*args],cwd=ROOT,check=True)
def main():
 verify_preparation();authority=read(BASE/'QUEUE_AUTHORITY.json')
 from scripts.continue_stvg_soft_ep_coordinate_search006 import pidfd_wait
 write(BASE/'STATUS.json',dict(status='waiting',stage='original_search_exit_and_42_slot_barrier',done=0,total=len(SPATIAL),GT_read=False,time=time.time()),mutable=True)
 pidfd_wait(authority['search_pid'],authority['search_start_ticks'])
 # No guessed progress/completion; a failed or manually stopped search fails this barrier.
 final=freeze()
 for i,(tid,variant,flags)in enumerate(SPATIAL):
  prepare_trial(tid,variant,flags,final)
  write(BASE/'STATUS.json',dict(status='running',stage='spatial_component',trial_id=tid,done=i,total=len(SPATIAL),GT_read=False,time=time.time()),mutable=True)
  subprocess.run(['systemd-run','--user','--wait','--collect','--unit=stvg-component128-'+tid.lower()+'-001','--property=MemoryHigh=16G','--property=MemoryMax=20G','--property=MemorySwapMax=1G','--working-directory='+str(ROOT),str(PY),'-B','scripts/launch_stvg_component_trial_v1.py',tid],cwd=ROOT,check=True)
  run('review_stvg_component_trial_v1.py',tid);run('measure_stvg_component_trial_v1.py','--trial',tid,'--score')
 write(BASE/'STATUS.json',dict(status='running',stage='temporal_component',done=len(SPATIAL),total=len(SPATIAL),GT_read=False,time=time.time()),mutable=True)
 pins=copy.deepcopy(final['pins']);pins.update(pinfile(BASE/'FINAL_PARAMETER_LOCK.json'))
 write(BASE/'TEMPORAL_COMPONENT_RUNTIME.json',dict(status='locked',pins=pins,protected=final['protected'],GT_read=False))
 run('run_stvg_component_temporal_v1.py','--run')
 run('measure_stvg_component_trial_v1.py','--temporal','--score')
 # Native-clock candidate construction is a new GT-free teacher scope. Its exact
 # prepared lock is handed to root before launching that GPU worker, never to an old queue.
 run('prepare_stvg_component_native_clock_v1.py','--prepare')
 write(BASE/'ACTUAL_COMPONENT_HANDOFF.json',dict(status='pending_actual_native_clock_teacher_student_root_then_complete_suite_public_archive',spatial_count=len(SPATIAL),temporal_variants=TEMPORAL,GT_read=False,time=time.time()))
 write(BASE/'STATUS.json',dict(status='pending_actual_native_clock_teacher_student_root',stage='root_handoff',done=len(SPATIAL),total=len(SPATIAL),GT_read=False,time=time.time()),mutable=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  write(BASE/'FAILURE.json',dict(status='failed',stage='required_root_review',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),GT_read=False,time=time.time()),mutable=True)
  write(BASE/'STATUS.json',dict(status='failed',stage='required_root_review',GT_read=False,time=time.time()),mutable=True);raise
