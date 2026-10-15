"""Run a new phase under an explicit immutable supplementary runtime."""
import sys,runpy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import stvg_evidence_common_v1 as common
RUNTIME=ROOT/sys.argv[1]
def verify_runtime():
 from scripts.stvg_evidence_runtime002 import verify_runtime as original
 original();r=common.read(RUNTIME);assert r['status']=='locked'and not r['GT_read']
 for p,h in {**r['pins'],**r.get('protected',{})}.items():assert common.sha(ROOT/p)==h,p
 return r
if __name__=='__main__':
 verify_runtime();common.verify_runtime=verify_runtime;target=sys.argv[2];sys.argv=sys.argv[2:];runpy.run_path(str(ROOT/target),run_name='__main__')
