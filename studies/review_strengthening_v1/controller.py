"""One own CPU consumer; reserve capacity for the existing CG scorer/auditor."""
from review_common import *
import subprocess,signal,traceback,shutil,fcntl

def live(w):
 try:a=Path(f'/proc/{w["pid"]}/stat').read_text().rsplit(')',1)[1].split();return int(a[19])==w['startticks']and a[0]!='Z'
 except FileNotFoundError:return False

def other_cpu():
 d=R/'outputs/shared_corrected_roi_CG_v1';return sum(live(read(p))for p in [d/'hc2_CPU_child.json',d/'vidstg_CPU_child.json']if p.exists())

def resources():
 return shutil.disk_usage(R).free>=80*2**30 and next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:'))>=8*2**30

def main():
 validate();assert read(D/'matched_cpu_preflight_receipt.json')['status']=='passed';assert not(D/'controller.json').exists();lk=(D/'controller.lock').open('a');fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB);who=identity();atomic(D/'controller.json',who)
 try:
  for ds in COUNTS:
   for stage in ['predict','score','audit']:
    while other_cpu()>=2 or not resources():time.sleep(10)
    log=(D/f'{ds}_{stage}.log').open('ab');c=subprocess.Popen(['taskset','-c','14-15','nice','-n','19','ionice','-c','3',sys.executable,'-B',str(S/'matched_cpu.py'),stage,'--dataset',ds],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''});w=identity(c.pid);atomic(D/f'{ds}_{stage}_child.json',w);atomic(D/'state.json',dict(status=stage,dataset=ds,controller=who,worker=w,whole_scope_complete=False));paused=False
    try:
     while c.poll()is None:
      pause=other_cpu()>=2 or not resources()
      if pause!=paused and live(w):os.killpg(c.pid,signal.SIGSTOP if pause else signal.SIGCONT);paused=pause;atomic(D/'resource_pause.json',dict(worker=w,paused=paused,time=time.time(),reason='max2 CPU jobs and RAM/disk guard; only own child signaled'))
      time.sleep(5)
    finally:
     if paused and live(w):os.killpg(c.pid,signal.SIGCONT)
    rc=c.wait();log.close();atomic(D/f'{ds}_{stage}_exit.json',dict(code=rc,worker=w));assert rc==0,'Failure preserved; no retry'
  atomic(D/'state.json',dict(status='matched_CPU_both_full_domains_scored_audited_pending_root_analysis_publication',controller=who,whole_scope_complete=False,SAM2_complete=False,gamma_complete=False))
 except BaseException:atomic(D/'state.json',dict(status='failed_preserved',controller=who,error=traceback.format_exc()));raise
if __name__=='__main__':main()
