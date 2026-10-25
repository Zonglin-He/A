"""New owned continuation; zero old TTA outputs/attempts were overwritten."""
from runtime_v4 import *
import inspect,textwrap

def controller():
    old=ns.load_old('_full_TTA_controller_preserved_v1',S/'controller.py')
    source=textwrap.dedent(inspect.getsource(old.queue))
    replacements=[
      ("while c.poll()is None:maintain();monitor(name,w);time.sleep(5)","while c.poll()is None:\n   if name!='current_kappa0_timing':maintain()\n   monitor(name,w);time.sleep(5)"),
      ("S/'run.py',['predict','--dataset','hc2']","S/'runtime_v4.py',['predict','--dataset','hc2']"),
      ("S/'spatial_tta.py',[]","S/'spatial_runtime_v4.py',[]"),
    ]
    for a,b in replacements:assert source.count(a)==1;source=source.replace(a,b)
    scope={**vars(old),'validate':validate_execution,'__file__':str(Path(__file__))}
    exec(compile(source,'<full_TTA_preserved_continuation_v3>','exec'),scope)
    return old,scope['queue']

def cpu_job(ds):
    validate_execution();old,_=controller();old.validate=validate_execution
    # Frozen score/audit have no engine dependency; scientific binding is checked there.
    old.cpu_job(ds)

def queue():
    validate_execution();proof=read(D/'execution_v4_preflight.json');assert proof['status']=='passed'
    assert proof['execution_plan_sha256']==sha(D/'execution_v4_plan.json')
    atomic(D/'active_execution.json',dict(entry=str(Path(__file__).relative_to(R)),scientific_fingerprint=read(D/'plan.json')['fingerprint'],execution_fingerprint=read(D/'execution_v4_plan.json')['fingerprint']))
    _,fn=controller();fn()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['queue','cpu-job']);p.add_argument('--dataset',choices=COUNTS);a=p.parse_args();queue()if a.command=='queue'else cpu_job(a.dataset)
