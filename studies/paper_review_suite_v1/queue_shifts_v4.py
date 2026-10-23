"""Same frozen shift inference/scoring; dependency follows recovered budget."""
import json
from pathlib import Path
import subprocess
import sys
import time
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R/'tools')]
from full import read,atomic,identity,sha
D=R/'outputs/paper_review_suite_v1';p=read(D/'shifts_plan.json');q=read(D/'shifts_queue_v4_plan.json')
assert sha(Path(__file__))==q['code_sha256']and q['shift_plan_sha256']==sha(D/'shifts_plan.json')
assert sha(R/'studies/paper_review_suite_v1/shifts.py')==p['code_sha256']
atomic(D/'shifts_v4_queue_worker.json',identity());atomic(D/'shifts_v4_state.json',dict(status='queued_after_budget_v2',controller=identity()))
while not(D/'budget_v2_confirm_score_receipt.json').exists():
    who=read(D/'budget_v4_launch.json')
    try:
        st=Path(f"/proc/{who['pid']}/stat").read_text().rsplit(')',1)[1].split();assert int(st[19])==who['startticks']and st[0]!='Z'
    except(OSError,AssertionError):
        atomic(D/'shifts_v4_state.json',dict(status='dependency_failed_no_start'));raise RuntimeError('Recovered budget failed before shift start')
    time.sleep(10)
for stage in ['screen','confirm']:
    for command in ['predict','score']:
        assert not(D/('shifts_'+stage+('_sealed.json'if command=='predict'else'_score_receipt.json'))).exists()
        atomic(D/'shifts_v4_state.json',dict(status=stage+'_'+command,controller=identity()))
        child=subprocess.Popen([sys.executable,'-B',str(R/'studies/paper_review_suite_v1/shifts.py'),command,'--stage',stage])
        atomic(D/('shifts_v4_'+stage+'_'+command+'_child.json'),identity(child.pid));code=child.wait()
        if code:
            atomic(D/'shifts_v4_state.json',dict(status='failed_preserved',stage=stage,command=command,exit=code));raise RuntimeError('Shift failed; preserve outputs and plan')
atomic(D/'shifts_v4_state.json',dict(status='complete_scored_pending_publication',fingerprint=p['fingerprint']))
