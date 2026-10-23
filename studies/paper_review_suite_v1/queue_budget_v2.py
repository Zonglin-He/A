"""Serial GPU queue: actual consistency completion, then two-level budgets."""
import sys,time,subprocess,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R/'tools')]
from full import read,atomic,identity,sha
D=R/'outputs/paper_review_suite_v1';p=read(D/'budget_plan.json');assert sha(Path(__file__))==read(D/'budget_queue_v2_plan.json')['code_sha256'];atomic(D/'budget_queue_worker.json',identity());atomic(D/'budget_state.json',dict(status='queued_after_consistency',controller=identity(),fingerprint=p['fingerprint']))
while not(D/'consistency_v2_score_receipt.json').exists():
 who=read(D/'consistency_v2_launch.json')
 try:st=Path(f"/proc/{who['pid']}/stat").read_text().rsplit(')',1)[1].split();assert int(st[19])==who['startticks']and st[0]!='Z'
 except (OSError,AssertionError):atomic(D/'budget_state.json',dict(status='dependency_failed_no_start',dependency='consistency',controller=identity()));raise RuntimeError('Consistency dependency failed, no budget run starts')
 time.sleep(10)
# Verify previous GPU controller has actually exited and no live lock owner.
while True:
 import fcntl
 f=(R/'runtime/gpu.lock').open('a')
 try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);f.close();break
 except BlockingIOError:f.close();time.sleep(5)
for stage in ['screen','confirm']:
 for command in ['predict','score']:
  result=D/('budget_'+stage+('_sealed.json'if command=='predict'else'_score_receipt.json'))
  assert not result.exists(),'Do not silently repeat budget phase'
  atomic(D/'budget_state.json',dict(status=stage+'_'+command,controller=identity(),fingerprint=p['fingerprint']));c=subprocess.Popen([sys.executable,'-B',str(R/'studies/paper_review_suite_v1/budget.py'),command,'--stage',stage]);atomic(D/('budget_'+stage+'_'+command+'_child.json'),identity(c.pid));exit=c.wait();atomic(D/('budget_'+stage+'_'+command+'_exit.json'),dict(code=exit))
  if exit:atomic(D/'budget_state.json',dict(status='failed_preserved',stage=stage,command=command,exit=exit));raise RuntimeError('Budget failed; preserve logs, do not blindly restart')
atomic(D/'budget_state.json',dict(status='complete_scored_pending_publication',fingerprint=p['fingerprint']))
