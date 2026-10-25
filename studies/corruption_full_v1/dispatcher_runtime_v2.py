"""Accept both successful parent audit and published status; no science changes."""
from common import *
import inspect, textwrap

def prepare():
    validate();assert not(D/'execution_v2_plan.json').exists()
    files=[Path(__file__),S/'dispatcher.py',S/'common.py']
    p=dict(schema='corruption_execution_v2',scientific_plan_sha256=sha(D/'plan.json'),
        change='The previous full-TTA owner can be root-marked complete_published after successful exit. Accept that successful presentation status as well as its raw final audit status; still require actual owner exit and all original per-domain audit SHA gates. No transform/method/query/prediction/metric/budget change; zero prior corruption model calls.',
        code_sha256={str(f.relative_to(R)):sha(f)for f in files})
    p['fingerprint']=digest(p);atomic(D/'execution_v2_plan.json',p);atomic(P/'execution_v2_protocol.json',p)
    print(p['fingerprint'],flush=True)

def queue():
    validate();p=read(D/'execution_v2_plan.json')
    assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
    assert p['scientific_plan_sha256']==sha(D/'plan.json')
    for f,h in p['code_sha256'].items():assert sha(R/f)==h
    old=ns.load_old('_full_corruption_original_dispatcher',S/'dispatcher.py',sys.modules['common'])
    code=textwrap.dedent(inspect.getsource(old.queue))
    before="assert state['status']=='all_full_TTA_and_current_recalculations_scored_audited_pending_root_export_publication'"
    after="assert state['status'] in {'all_full_TTA_and_current_recalculations_scored_audited_pending_root_export_publication','complete_published'}"
    assert code.count(before)==1
    scope={**vars(old)};exec(compile(code.replace(before,after),'<successful_parent_status_gate_v2>','exec'),scope)
    atomic(D/'active_execution.json',dict(entry=str(Path(__file__).relative_to(R)),fingerprint=p['fingerprint'],scientific_fingerprint=read(D/'plan.json')['fingerprint']))
    scope['queue']()

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','queue']);z=a.parse_args()
    globals()[z.command]()
