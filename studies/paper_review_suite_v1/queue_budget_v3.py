"""Independent control-flow recovery; original numerical budget contract fixed."""
import json
from pathlib import Path
import subprocess
import sys
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R/'tools')]
from full import read,atomic,identity,sha
D=R/'outputs/paper_review_suite_v1'
p=read(D/'budget_v2_plan.json');q=read(D/'budget_queue_v3_plan.json')
assert sha(Path(__file__))==q['code_sha256']and q['budget_plan_sha256']==sha(D/'budget_v2_plan.json')
assert(D/'consistency_v2_score_receipt.json').exists()
atomic(D/'budget_v3_queue_worker.json',identity())
for stage in ['screen','confirm']:
    for command in ['predict','score']:
        result=D/('budget_v2_'+stage+('_sealed.json'if command=='predict'else'_score_receipt.json'))
        assert not result.exists(),'Do not blindly duplicate completed phases'
        atomic(D/'budget_v2_state.json',dict(status=stage+'_'+command,controller=identity(),fingerprint=p['fingerprint']))
        child=subprocess.Popen([sys.executable,'-B',str(R/'studies/paper_review_suite_v1/budget_v2.py'),command,'--stage',stage])
        atomic(D/('budget_v2_'+stage+'_'+command+'_child.json'),identity(child.pid));code=child.wait()
        atomic(D/('budget_v2_'+stage+'_'+command+'_exit.json'),dict(code=code))
        if code:
            atomic(D/'budget_v2_state.json',dict(status='failed_preserved',stage=stage,command=command,exit=code));raise RuntimeError('Preserve new failure; no automatic retry')
atomic(D/'budget_v2_state.json',dict(status='complete_scored_pending_publication',fingerprint=p['fingerprint']))
