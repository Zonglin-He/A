"""Timing-only scheduling correction before any full-TTA execution."""
from full_common import *
import inspect,textwrap,argparse

def prepare():
 validate();assert not(D/'execution_v2_plan.json').exists();source=S/'controller.py';p=dict(schema='full_TTA_execution_v2',scientific_plan_sha256=sha(D/'plan.json'),change='While current-kappa0 deployment timing has paused owned CPU groups, do not call maintain() to launch other ready CPU jobs. Existing paused groups stay accounted for and are resumed in finally. No optimizer/model/data/evaluation change; no prior fullTTA dispatcher or prediction was launched.',code_sha256={str(f.relative_to(R)):sha(f)for f in [Path(__file__),source,S/'full_common.py']});p['fingerprint']=digest(p);atomic(D/'execution_v2_plan.json',p);atomic(P/'execution_v2_protocol.json',p)
 for f in [Path(__file__),source]:q=D/'execution_v2_source_snapshot'/f.name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(f.read_bytes())
 print(p['fingerprint'])
def validate_execution():
 parent=validate();p=read(D/'execution_v2_plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})and p['scientific_plan_sha256']==sha(D/'plan.json')
 for f,h in p['code_sha256'].items():assert sha(R/f)==h
 return parent
def queue():
 validate_execution();old=ns.load_old('_full_TTA_controller_v1',S/'controller.py');src=textwrap.dedent(inspect.getsource(old.queue));before='while c.poll()is None:maintain();monitor(name,w);time.sleep(5)';after="while c.poll()is None:\n   if name!='current_kappa0_timing':maintain()\n   monitor(name,w);time.sleep(5)";assert src.count(before)==1;src=src.replace(before,after);scope={**old.queue.__globals__,'validate':validate_execution};exec(compile(src,'<owned_CPU_pause_timing_scheduler_v2>','exec'),scope);atomic(D/'active_execution.json',dict(scientific_fingerprint=read(D/'plan.json')['fingerprint'],execution_fingerprint=read(D/'execution_v2_plan.json')['fingerprint'],entry=str(Path(__file__).relative_to(R))));scope['queue']()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','queue']);z=p.parse_args();globals()[z.command]()
