"""Only two whole-domain CPU clean-reference jobs; no GPU/corruption launcher."""
from shared_common import *
import subprocess,traceback
validate();assert read(D/'preflight_receipt.json')['status']=='passed';assert not(D/'controller.json').exists();who=identity();atomic(D/'controller.json',who);jobs={}
try:
 for ds in COUNTS:
  log=(D/f'components_{ds}_domain.log').open('ab');c=subprocess.Popen([sys.executable,'-B',str(S/'run.py'),'domain','--dataset',ds],stdout=log,stderr=subprocess.STDOUT);jobs[ds]=(c,log);atomic(D/f'components_{ds}_domain_child.json',identity(c.pid))
 atomic(D/'state.json',dict(status='clean_components_final_DP_two_domain_CPU_recalculation',controller=who,corruption_release_allowed=False))
 for ds,(c,log)in jobs.items():
  code=c.wait();log.close();atomic(D/f'components_{ds}_domain_exit.json',dict(code=code));assert code==0,'Preserve failure; no retry'
 atomic(D/'state.json',dict(status='clean_components_final_DP_scored_audited_pending_root_export_and_remaining_synchronization',controller=who,corruption_release_allowed=False))
except BaseException:
 atomic(D/'state.json',dict(status='failed_preserved',error=traceback.format_exc(),corruption_release_allowed=False));raise
