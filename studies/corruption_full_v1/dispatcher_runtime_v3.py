"""Rebind the zero-work corruption waiter to the verified TTA interface recovery."""
from common import *
import inspect,textwrap

def validate_execution():
    validate();p=read(D/'execution_v3_plan.json')
    assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
    assert p['scientific_plan_sha256']==sha(D/'plan.json')
    assert p['parent_execution_sha256']==sha(R/'outputs/tta_full_v1/execution_v3_plan.json')
    for f,h in p['code_sha256'].items():assert sha(R/f)==h,f
    return p

def queue():
    p=validate_execution()
    old=ns.load_old('_full_corruption_preserved_dispatcher_v1',S/'dispatcher.py',sys.modules['common'])
    source=textwrap.dedent(inspect.getsource(old.queue))
    before="assert parent==dict(pid=1924421,startticks=245882503)"
    after="assert read(R/'outputs/tta_full_v1/active_execution.json')['execution_fingerprint']==read(R/'outputs/tta_full_v1/execution_v3_plan.json')['fingerprint']"
    assert source.count(before)==1;source=source.replace(before,after)
    before="assert state['status']=='all_full_TTA_and_current_recalculations_scored_audited_pending_root_export_publication'"
    after="assert state['status'] in {'all_full_TTA_and_current_recalculations_scored_audited_pending_root_export_publication','complete_published'}"
    assert source.count(before)==1;source=source.replace(before,after)
    scope={**vars(old)}
    exec(compile(source,'<parent_interface_recovery_corruption_gate_v3>','exec'),scope)
    atomic(D/'active_execution.json',dict(entry=str(Path(__file__).relative_to(R)),fingerprint=p['fingerprint'],scientific_fingerprint=read(D/'plan.json')['fingerprint']))
    scope['queue']()

if __name__=='__main__':queue()
