"""One GPU panel-prefix worker only after the CG transfer owner succeeds."""
from panel_common import *
import subprocess,traceback
validate();assert read(D/'preflight_receipt.json')['status']=='passed_CPU_contract_and_cached_math_only';assert not(D/'controller.json').exists();who=identity();atomic(D/'controller.json',who)
try:
 cg=read(CGNEW/'controller.json');atomic(D/'state.json',dict(status='waiting_CG_transfer_actual_successful_exit_and_audits',controller=who,corruption_release_allowed=False))
 while not exited(cg):time.sleep(15)
 gate={}
 for ds in COUNTS:
  assert read(CGNEW/f'{ds}_GPU_exit.json')['code']==0 and read(CGNEW/f'{ds}_CPU_exit.json')['code']==0;ap=CGNEW/f'{ds}_reporting_receipt.json';a=read(ap);assert a['status']=='verified_whole_domain'and a['all_means_recomputed'];gate[ds]=dict(path=str(ap.relative_to(R)),sha256=sha(ap));assert read(selected.D/f'components_{ds}_reporting_receipt.json')['status']=='verified_whole_domain'
 atomic(D/'dependency_gate.json',gate);jobs=[]
 for ds in COUNTS:
  log=(D/f'{ds}_GPU.log').open('ab');c=subprocess.Popen([sys.executable,'-B',str(S/'worker.py'),'--dataset',ds],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'});atomic(D/f'{ds}_GPU_child.json',identity(c.pid));atomic(D/'state.json',dict(status='panel_prefix_GPU_'+ds,controller=who,corruption_release_allowed=False));rc=c.wait();log.close();atomic(D/f'{ds}_GPU_exit.json',dict(code=rc));assert rc==0,'Preserve paid prefix; no fallback or retry'
  log=(D/f'{ds}_CPU.log').open('ab');a=subprocess.Popen(['taskset','-c','14-15','nice','-n','19','ionice','-c','3',sys.executable,'-B',str(S/'report.py'),'domain','--dataset',ds],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'});atomic(D/f'{ds}_CPU_child.json',identity(a.pid));jobs.append((ds,a,log))
 for ds,a,log in jobs:
  rc=a.wait();log.close();atomic(D/f'{ds}_CPU_exit.json',dict(code=rc));assert rc==0,'Score/audit failure preserved'
 atomic(D/'state.json',dict(status='all768_selected_method_panels_scored_audited_pending_root_analysis',controller=who,corruption_release_allowed=False))
except BaseException:atomic(D/'state.json',dict(status='failed_preserved',controller=who,error=traceback.format_exc(),corruption_release_allowed=False));raise
