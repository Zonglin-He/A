"""Anonymize roster identities consistently in scalar lists as well as row fields."""
from pathlib import Path
import ast,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import report_stvg_controls256_speed005 as previous
def source(use_A):
    s=previous.source(use_A);needle='        return v\n'
    assert s.count(needle)==1
    s=s.replace(needle,'        return aliases.get(v,v) if isinstance(v,str) else v\n')
    needle="names=['report_stvg_controls256_speed005.py',";assert s.count(needle)==1
    return s.replace(needle,"names=['report_stvg_controls256_speed006.py','report_stvg_controls256_speed005.py',")
def main(kind,group):
    use_A=False
    if kind=='core256':
        from scripts.add_stvg_controls256_A_comparisons_v1 import prepare
        use_A=prepare(group)
    ns=dict(previous.previous.previous.previous.original.__dict__)
    ns.update(use_A=use_A,engineering_cost=previous.previous.engineering_cost)
    exec(compile(source(use_A),str(Path(__file__)),'exec'),ns);ns['main'](kind,group)
if __name__=='__main__':
    if sys.argv[1]=='contracts':
        for a in (False,True):ast.parse(source(a))
        print('EXPORT006_LIST_AND_ROW_ALIAS_CONTRACT_PASS_NO_GT_NO_MODEL')
    else:main(*sys.argv[1:3])
