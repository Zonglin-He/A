"""Isolated identity/update experiment custody and no-label execution guard."""
import os
from pathlib import Path
from scripts.stvg_native_instance_common_v1 import read,write,sha,guard as old_guard
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/stvg_instance_identity_update_v1'
OLD=ROOT/'artifacts/stvg_native_instance_spatial_opd_v1'
def guard(event,args):
    old_guard(event,args)
    if event=='open' and args and isinstance(args[0],(str,bytes,Path)):
        s=os.fsdecode(args[0]).lower()
        if any(x in s for x in ['postseal_score','pool_candidate_coverage_postseal','/metrics.json','/statistics.json']):
            raise PermissionError('method worker refuses prior or current evaluation scores')
def verify(stage='student'):
    d=read(BASE/(stage.upper()+'_RUNTIME.json'))
    assert d['status']=='locked' and d['GT_read'] is False
    for p,h in d['pins'].items():assert sha(ROOT/p)==h,p
    return d
def row(case):
    return next(r for r in read(OLD/'ROSTER_LOCK.json')['records'] if r['case_id']==case)
