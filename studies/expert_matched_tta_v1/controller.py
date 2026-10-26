"""One CPU consumer only; no GPU/model/teacher/optimizer scheduling."""
from common import *
import subprocess,signal,traceback,fcntl
def live(w):
 try:
  s=Path('/proc',str(w['pid']),'stat').read_text().split();return s[2]!='Z'and s[21]==str(w['startticks'])
 except FileNotFoundError:return False
def main():
 validate();assert not(D/'controller.json').exists();lock=(D/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);who=identity();atomic(D/'controller.json',who)
 try:
  for ds in COUNTS:
   for stage in ['predict','score','audit']:
    while cg_cpu_jobs()>=2 or not resources():time.sleep(10)
    log=(D/f'{ds}_{stage}.log').open('ab');c=subprocess.Popen(['taskset','-c','14-15','nice','-n','19','ionice','-c','3',sys.executable,'-B',str(S/'run.py'),stage,'--dataset',ds],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''});w=identity(c.pid);atomic(D/f'{ds}_{stage}_child.json',w);atomic(D/'state.json',dict(status=stage,dataset=ds,controller=who,worker=w));paused=False
    try:
     while c.poll()is None:
      pause=cg_cpu_jobs()>=2 or not resources()
      if pause!=paused and live(w):os.killpg(c.pid,signal.SIGSTOP if pause else signal.SIGCONT);paused=pause
      time.sleep(5)
    finally:
     if paused and live(w):os.killpg(c.pid,signal.SIGCONT)
    code=c.wait();log.close();atomic(D/f'{ds}_{stage}_exit.json',dict(code=code,worker=w));assert code==0,'Preserve failure; no automatic retry'
  atomic(D/'state.json',dict(status='both_full_domains_scored_audited_pending_root_analysis_publication',controller=who,whole_scope_complete=False))
 except BaseException:atomic(D/'state.json',dict(status='failed_preserved',controller=who,error=traceback.format_exc()));raise
if __name__=='__main__':main()
