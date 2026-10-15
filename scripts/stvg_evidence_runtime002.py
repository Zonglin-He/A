"""Explicit compatibility for the source runtime's optional protected mapping."""
import sys,runpy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import stvg_evidence_common_v1 as c
def verify_runtime():
 r=c.read(c.BASE/'RUNTIME002.json');assert r['status']=='locked'and not r['GT_read']
 for p,h in {**r['pins'],**r.get('protected',{})}.items():assert c.sha(ROOT/p)==h,p
 return r
if __name__=='__main__':
 verify_runtime();c.verify_runtime=verify_runtime;target=sys.argv[1];sys.argv=sys.argv[1:];runpy.run_path(str(ROOT/target),run_name='__main__')
