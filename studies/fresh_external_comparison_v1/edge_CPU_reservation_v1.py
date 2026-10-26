"""Reserve CPU priority by pausing only this root's exact temporal-risk group."""
from pathlib import Path
import os,signal,time,json,hashlib
R=Path(__file__).resolve().parents[2];D=R/'outputs/fresh_external_comparison_v1/edgetam_v1';T=R/'outputs/temporal_feedback_risk_v1';PID=1318498;START=257114336
def atomic(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2)+'\n');os.replace(q,p)
def same(pid,start):
 try:
  a=(Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1].split();return int(a[19])==start and a[0]!='Z'
 except FileNotFoundError:return False
def own_group():
 assert same(PID,START)and os.getpgid(PID)==PID
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   if os.getpgid(int(p.name))==PID:assert b'temporal_feedback_risk_v1/'in(p/'cmdline').read_bytes()
  except(FileNotFoundError,ProcessLookupError):pass
def main():
 start=time.time();owner=json.loads((D/'controller_v1.json').read_text());assert owner==dict(pid=1230206,startticks=257018909);assert(T/'Edge_CPU_reservation_initial_pause_v1.json').exists();own_group();atomic(D/'CPU_reservation_worker_v1.json',dict(pid=os.getpid(),risk_controller=dict(pid=PID,startticks=START),edge_controller=owner,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()));held=True
 try:
  while same(owner['pid'],owner['startticks']):
   state=json.loads((D/'state_v1.json').read_text())['status']
   if state in ['hc2_score','hc2_audit','vidstg_score','vidstg_audit']:
    if same(PID,START):own_group();os.killpg(PID,signal.SIGSTOP);held=True
   elif state.startswith(('fixed256_','failed_')):break
   time.sleep(2)
 finally:
  if held and same(PID,START)and not(T/'user_pause.json').exists():own_group();os.killpg(PID,signal.SIGCONT);atomic(T/'Edge_CPU_reservation_resume_v1.json',dict(controller=dict(pid=PID,startticks=START),SIGCONT_sent=True,time=time.time(),reservation_wall_seconds=time.time()-start))
if __name__=='__main__':main()
