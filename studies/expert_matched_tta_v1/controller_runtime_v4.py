"""Resume once from preserved48CPU commits; no old dispatcher resurrection."""
import sys,importlib.util,subprocess,signal,traceback,time,os,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('_expert_matched_runtime4',ROOT/'runtime_v4.py');v=importlib.util.module_from_spec(s);s.loader.exec_module(v);c=v.c
def live(w):
 try:
  s=Path('/proc',str(w['pid']),'stat').read_text().split();return s[2]!='Z'and s[21]==str(w['startticks'])
 except FileNotFoundError:return False
def jobs():
 count=0
 for ds in c.COUNTS:
  p=c.R/'outputs/shared_corrected_roi_CG_v1'/f'{ds}_CPU_child.json'
  if p.exists():count+=int(live(c.read(p)))
 return count
def main():
 v.validate_execution();assert not(c.D/'controller_runtime_v4.json').exists();lock=(c.D/'controller_runtime_v4.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);who=c.identity();c.atomic(c.D/'controller_runtime_v4.json',who)
 try:
  for ds in c.COUNTS:
   for stage in ['predict','score','audit']:
    while jobs()>=2 or not c.resources():time.sleep(10)
    log=(c.D/f'v4_{ds}_{stage}.log').open('ab');p=subprocess.Popen(['taskset','-c','14-15','nice','-n','19','ionice','-c','3',sys.executable,'-B',str(ROOT/'runtime_v4.py'),stage,'--dataset',ds],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''});w=c.identity(p.pid);c.atomic(c.D/f'v4_{ds}_{stage}_child.json',w);c.atomic(c.D/'runtime_v4_state.json',dict(status=stage,dataset=ds,controller=who,worker=w));paused=False
    try:
     while p.poll()is None:
      pause=jobs()>=2 or not c.resources()
      if pause!=paused and live(w):os.killpg(p.pid,signal.SIGSTOP if pause else signal.SIGCONT);paused=pause
      time.sleep(5)
    finally:
     if paused and live(w):os.killpg(p.pid,signal.SIGCONT)
    code=p.wait();log.close();c.atomic(c.D/f'v4_{ds}_{stage}_exit.json',dict(code=code,worker=w));assert code==0,'Preserve failure and no blind retry'
  c.atomic(c.D/'runtime_v4_state.json',dict(status='both_full_domains_scored_audited_pending_root_analysis_publication',controller=who,whole_scope_complete=False))
 except BaseException:c.atomic(c.D/'runtime_v4_state.json',dict(status='failed_preserved',controller=who,error=traceback.format_exc()));raise
if __name__=='__main__':main()
