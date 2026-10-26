"""Resume only frozen HC predictions; independent report-interface version."""
from review_common import *
from controller import resources,other_cpu,live
from matched_report_runtime_v2 import validate_runtime
import subprocess,signal,traceback,fcntl

def main():
 validate_runtime();assert not(D/'controller_runtime_v2.json').exists();lock=(D/'controller_runtime_v2.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);who=identity();atomic(D/'controller_runtime_v2.json',who)
 try:
  se=read(D/'matched_cpu/hc2/sealed.json');assert se['queries']==1902 and not se['GT_read']
  for e in se['files'].values():
   for k,h in [('path','sha256'),('receipt_path','receipt_sha256')]:assert sha(D/e[k])==e[h]
  for ds in COUNTS:
   for stage in (['score','audit']if ds=='hc2'else['predict','score','audit']):
    while other_cpu()>=2 or not resources():time.sleep(10)
    entry='matched_cpu.py'if stage=='predict'else'matched_report_runtime_v2.py';log=(D/f'runtime_v2_{ds}_{stage}.log').open('ab');c=subprocess.Popen(['taskset','-c','14-15','nice','-n','19','ionice','-c','3',sys.executable,'-B',str(S/entry),stage,'--dataset',ds],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''});w=identity(c.pid);atomic(D/f'runtime_v2_{ds}_{stage}_child.json',w);atomic(D/'runtime_v2_state.json',dict(status=stage,dataset=ds,controller=who,worker=w,whole_scope_complete=False));paused=False
    try:
     while c.poll()is None:
      pause=other_cpu()>=2 or not resources()
      if pause!=paused and live(w):os.killpg(c.pid,signal.SIGSTOP if pause else signal.SIGCONT);paused=pause
      time.sleep(5)
    finally:
     if paused and live(w):os.killpg(c.pid,signal.SIGCONT)
    rc=c.wait();log.close();atomic(D/f'runtime_v2_{ds}_{stage}_exit.json',dict(code=rc,worker=w));assert rc==0,'Preserve failed attempt; no retry'
  atomic(D/'runtime_v2_state.json',dict(status='matched_CPU_both_full_domains_scored_audited_pending_root_analysis_publication',controller=who,whole_scope_complete=False))
 except BaseException:atomic(D/'runtime_v2_state.json',dict(status='failed_preserved',controller=who,error=traceback.format_exc()));raise
if __name__=='__main__':main()
