"""Preserve strictly roster-bound aliases already anonymized by the speed profiler."""
from pathlib import Path
import ast,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import report_stvg_controls256_speed004 as previous

def source(use_A):
    s=previous.source(use_A)
    needle=";assert len(aliases)==count"
    assert s.count(needle)==1
    s=s.replace(needle,needle+"\n    public_ids=set(aliases.values());assert len(public_ids)==count\n    assert not(set(aliases)&public_ids)\n    aliases.update({key:key for key in public_ids})")
    needle="names=['report_stvg_controls256_speed004.py',"
    assert s.count(needle)==1
    return s.replace(needle,"names=['report_stvg_controls256_speed005.py','report_stvg_controls256_speed004.py',")

def contracts():
    for x in (False,True):ast.parse(source(x))
    aliases={'private1':'R001','private2':'R002'}
    aliases.update({x:x for x in set(aliases.values())})
    assert aliases['private1']==aliases['R001']=='R001'
    assert aliases['private2']==aliases['R002']=='R002'
    for bad in ('R003','unbound_private'):
        try:aliases[bad]
        except KeyError:pass
        else:raise AssertionError('unknown identity accepted')
    print('EXPORT_ALIAS005_CPU_CONTRACT_PASS_NO_MODEL_NO_GT')

def main(kind,group):
    use_A=False
    if kind=='core256':
        from scripts.add_stvg_controls256_A_comparisons_v1 import prepare
        use_A=prepare(group)
    ns=dict(previous.previous.previous.original.__dict__)
    ns.update(use_A=use_A,engineering_cost=previous.engineering_cost)
    exec(compile(source(use_A),str(Path(__file__)),'exec'),ns)
    ns['main'](kind,group)

if __name__=='__main__':contracts() if sys.argv[1]=='contracts' else main(*sys.argv[1:3])
