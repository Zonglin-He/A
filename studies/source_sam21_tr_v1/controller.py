"""Pilot then full under one controller; CPU scoring may overlap next GPU domain."""
from common import *
import subprocess,traceback
def main():
 plan=check();assert not(D/'controller.json').exists();atomic(D/'controller.json',dict(**identity(),plan_fingerprint=plan['fingerprint']));success={}
 def spawn(name,argv,device):
  path=D/'logs'/(name+'.log');path.parent.mkdir(parents=True,exist_ok=True);f=path.open('xb');p=subprocess.Popen([str(PYTHON),'-B',*map(str,argv)],cwd=R,env={**os.environ,'CUDA_VISIBLE_DEVICES':device},stdout=f,stderr=subprocess.STDOUT,start_new_session=True);w=identity(p.pid);atomic(D/'children'/(name+'.json'),dict(**w,argv=list(map(str,argv))));return p,w,f,time.perf_counter(),path
 def collect(name,process):
  p,w,f,start,path=process;code=p.wait();f.close();e=dict(worker=w,actual_subprocess_wait=True,exit_code=code,elapsed_wall_seconds=time.perf_counter()-start,log_sha256=sha(path));atomic(D/'exits'/(name+'.json'),e);assert code==0,(name,code);success[name]=e
 try:
  for stage in ['pilot','full']:
   if stage=='full':
    for ds in COUNTS:
     assert read(D/'pilot'/ds/'reporting_receipt.json')['queries']==128;assert read(D/'pilot'/ds/'export_receipt.json')['audit_sha256']==sha(D/'pilot'/ds/'reporting_receipt.json')
    atomic(D/'full_release.json',dict(status='user_authorized_full_released_after_both128_whole_functional_and_score_audits_not_target_accuracy_selection',pilot_GPU_actual_exit_sha256=sha(D/'exits/pilot_GPU.json'),pilot_audits={ds:sha(D/'pilot'/ds/'reporting_receipt.json')for ds in COUNTS},unchanged_plan_fingerprint=plan['fingerprint'],new_full_calls_cap=11949,pilot_calls_reused_exactly=256,worker=identity()))
   name=stage+'_GPU';gpu=spawn(name,[S/'worker.py','--stage',stage],'0');atomic(D/'state.json',dict(status='running',stage=stage,GPU_worker=gpu[1],controller=identity()))
   for ds in COUNTS:
    while not(D/stage/ds/'sealed.json').exists():
     code=gpu[0].poll();assert code is None or code==0,('GPU failed; preserve paid state',code)
     if code==0:assert(D/stage/ds/'sealed.json').exists()
     time.sleep(5)
    for task in ['input_audit','score','audit','export']:
     key=f'{stage}_{ds}_{task}';atomic(D/'state.json',dict(status='running',stage=stage,CPU_task=task,dataset=ds,GPU_worker=gpu[1],controller=identity()));collect(key,spawn(key,[S/'report.py',task,'--stage',stage,'--dataset',ds],''))
   collect(name,gpu);atomic(D/stage/'stage_complete.json',dict(status='both_domains_predictions_input_official_manual_audits_signed_exports_complete',worker=identity(),audits={ds:sha(D/stage/ds/'reporting_receipt.json')for ds in COUNTS},new_Source=0,new_ReferDINO=0,new_optimizer=0))
  atomic(D/'controller_complete.json',dict(status='pilot256_and_full12205_done_scored_audited_exported_PENDING_ROOT_analysis_publication',all_actual_successful_children=success,worker=identity()));atomic(D/'state.json',dict(status='complete_pending_ROOT_analysis_publication',pilot_complete=True,full_complete=True,all_new_SAM2_queries_cap=12205,no_new_Source_ReferDINO_optimizer=True));print('SAM21 pilot and full complete pending ROOT/publication',flush=True)
 except BaseException:atomic(D/'controller_failure.json',dict(error=traceback.format_exc(),successful_children=success,worker=identity(),all_paid_state_preserved=True));raise
if __name__=='__main__':main()
