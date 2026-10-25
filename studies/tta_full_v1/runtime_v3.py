"""Verified continuation after zero-paid-work metadata failure, same scientific plan."""
from full_common import *
import argparse

def validate_execution():
    original=validate();p=read(D/'execution_v3_plan.json')
    assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
    assert p['scientific_plan_sha256']==sha(D/'plan.json')
    for f,h in p['code_sha256'].items():assert sha(R/f)==h,f
    assert sha(D/'failure_interface_v1_20261025/preservation.json')==p['preservation_sha256']
    return original

def install_engine():
    import engine_runtime_v2
    sys.modules['engine']=engine_runtime_v2

def run(command,ds):
    validate_execution();install_engine()
    frozen=ns.load_old('_full_TTA_run_unchanged_v1',S/'run.py')
    frozen.validate=validate_execution
    getattr(frozen,command)(ds)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['predict','score','audit']);p.add_argument('--dataset',choices=COUNTS,required=True);a=p.parse_args();run(a.command,a.dataset)
