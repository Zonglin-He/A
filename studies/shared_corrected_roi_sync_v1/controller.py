"""Successor CPU-only scheduler; wait for initial clean owner to exit, then max two jobs."""
from sync_common import *
import subprocess,traceback
validate();assert read(D/'preflight_receipt.json')['status']=='passed';assert not(D/'controller.json').exists();who=identity();atomic(D/'controller.json',who)
try:
 upstream=read(selected.D/'controller.json');atomic(D/'state.json',dict(status='waiting_initial_clean_controller_success',upstream=upstream,controller=who,corruption_release_allowed=False))
 while not exited(upstream):time.sleep(10)
 gate={}
 for ds in COUNTS:
  assert read(selected.D/f'components_{ds}_domain_exit.json')['code']==0
  ap=selected.D/f'components_{ds}_reporting_receipt.json';assert read(ap)['status']=='verified_whole_domain';gate[ds]=dict(path=str(ap.relative_to(R)),sha256=sha(ap))
 atomic(D/'clean_dependency_gate.json',gate)
 todo=[(task,ds)for task in ['budget','spatial']for ds in COUNTS];running=[];done=[]
 while todo or running:
  for c,log,task,ds in list(running):
   rc=c.poll()
   if rc is not None:
    log.close();atomic(D/f'{task}_{ds}_domain_exit.json',dict(code=rc));assert rc==0,'Failure preserved; no blind retry';running.remove((c,log,task,ds));done.append([task,ds])
  while todo and len(running)<2:
   assert next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:'))>=8*2**30
   task,ds=todo.pop(0);log=(D/f'{task}_{ds}_domain.log').open('ab');c=subprocess.Popen([sys.executable,'-B',str(S/'run.py'),'domain','--task',task,'--dataset',ds],stdout=log,stderr=subprocess.STDOUT);running.append((c,log,task,ds));atomic(D/f'{task}_{ds}_domain_child.json',identity(c.pid))
  atomic(D/'state.json',dict(status='CPU_sync_running'if running else'all_CPU_sync_scored_audited_pending_root_analysis',controller=who,running=[[t,ds]for c,l,t,ds in running],completed=done,corruption_release_allowed=False));time.sleep(10)
except BaseException:
 atomic(D/'state.json',dict(status='failed_preserved',error=traceback.format_exc(),corruption_release_allowed=False));raise
