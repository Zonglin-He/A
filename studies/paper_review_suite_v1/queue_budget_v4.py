"""Resume sealed budget screening via a CPU path adapter, then fixed confirm."""
from pathlib import Path
import subprocess
import sys
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R/'tools')]
from full import read,sha,atomic,identity
D=R/'outputs/paper_review_suite_v1'
q=read(D/'budget_queue_v4_plan.json');p=read(D/'budget_v2_plan.json')
assert sha(Path(__file__))==q['code_sha256'] and sha(D/'budget_v2_plan.json')==q['budget_plan_sha256']
assert sha(D/'budget_v2_screen_sealed.json')==q['screen_seal_sha256']
atomic(D/'budget_v4_queue_worker.json',identity())
for stage,command in [('screen','score'),('confirm','predict'),('confirm','score')]:
    result=D/('budget_v2_'+stage+('_sealed.json'if command=='predict'else'_score_receipt.json'))
    assert not result.exists(),'No duplicate completed phases'
    atomic(D/'budget_v2_state.json',dict(status=stage+'_'+command,controller=identity(),fingerprint=p['fingerprint'],recovery_controller_plan=q['fingerprint']))
    args=[sys.executable,'-B',str(R/'studies/paper_review_suite_v1'/('budget_score_adapter_v3.py'if command=='score'else'budget_v2.py'))]
    if command=='predict':args.append('predict')
    child=subprocess.Popen(args+['--stage',stage])
    atomic(D/('budget_v4_'+stage+'_'+command+'_child.json'),identity(child.pid));code=child.wait()
    atomic(D/('budget_v4_'+stage+'_'+command+'_exit.json'),dict(code=code))
    if code:
        atomic(D/'budget_v2_state.json',dict(status='failed_preserved',stage=stage,command=command,exit=code))
        raise RuntimeError('Preserve failure; no automatic retry')
atomic(D/'budget_v2_state.json',dict(status='complete_scored_pending_publication',fingerprint=p['fingerprint']))
