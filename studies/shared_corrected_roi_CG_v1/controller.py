"""Single waiting CG GPU owner after actual timing and CPU synchronization audits."""
from cg_common import *
import subprocess,traceback
def require_audit(path):
 a=read(path);assert a['status']=='verified_whole_domain'and a['all_means_recomputed'];return dict(path=str(path.relative_to(R)),sha256=sha(path))
validate();assert read(D/'preflight_receipt.json')['status']=='passed';assert not(D/'controller.json').exists();who=identity();atomic(D/'controller.json',who)
try:
 sync=R/'outputs/shared_corrected_roi_sync_v1';timing=R/'outputs/shared_corrected_roi_timing_v1';sw=read(sync/'controller.json');tw=read(timing/'controller_v2.json');atomic(D/'state.json',dict(status='waiting_actual_timing_and_CPU_sync_successful_exit_audits',controller=who,corruption_release_allowed=False))
 while not exited(sw)or not exited(tw):time.sleep(15)
 gate={}
 for ds in COUNTS:
  assert read(selected.D/f'components_{ds}_domain_exit.json')['code']==0;gate['clean_'+ds]=require_audit(selected.D/f'components_{ds}_reporting_receipt.json')
  for task in ['budget','spatial']:
   assert read(sync/f'{task}_{ds}_domain_exit.json')['code']==0;gate[task+'_'+ds]=require_audit(sync/f'{task}_{ds}_reporting_receipt.json')
 assert read(timing/'timing_exit.json')['code']==0;ts=read(timing/'deployment/sealed.json');assert ts['all_current_MAP_outputs_bitwise']and ts['calls']==192 and ts['expert_calls']==96;gate['timing']=dict(path=str((timing/'deployment/sealed.json').relative_to(R)),sha256=sha(timing/'deployment/sealed.json'));atomic(D/'dependency_gate.json',gate)
 jobs=[]
 for ds in COUNTS:
  log=(D/f'{ds}_GPU.log').open('ab');c=subprocess.Popen([sys.executable,'-B',str(S/'worker.py'),'--dataset',ds],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'});atomic(D/f'{ds}_GPU_child.json',identity(c.pid));atomic(D/'state.json',dict(status='CG_GPU_'+ds,controller=who,corruption_release_allowed=False));rc=c.wait();log.close();atomic(D/f'{ds}_GPU_exit.json',dict(code=rc));assert rc==0,'Preserve paid acquisition and failure; no retry'
  log=(D/f'{ds}_CPU.log').open('ab');a=subprocess.Popen(['taskset','-c','14-15','nice','-n','19','ionice','-c','3',sys.executable,'-B',str(S/'report.py'),'domain','--dataset',ds],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'});atomic(D/f'{ds}_CPU_child.json',identity(a.pid));jobs.append((ds,a,log))
 for ds,a,log in jobs:
  rc=a.wait();log.close();atomic(D/f'{ds}_CPU_exit.json',dict(code=rc));assert rc==0,'Independent scoring/audit failed; preserve'
 atomic(D/'state.json',dict(status='both_CG_domains_scored_audited_pending_root_analysis_publication',controller=who,corruption_release_allowed=False))
except BaseException:atomic(D/'state.json',dict(status='failed_preserved',controller=who,error=traceback.format_exc(),corruption_release_allowed=False));raise
