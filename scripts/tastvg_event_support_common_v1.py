"""Separate finite development batch; predecessors remain immutable."""
import sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,save,load,sha,status
BASE=ROOT/'artifacts/tastvg_event_support_v1'
POOL=ROOT/'artifacts/tastvg_extended_sensitivity_v3'
DATASETS=['vidstg','hc2']
def budget():assert shutil.disk_usage(ROOT).free>8*2**30
def verify(dataset=None):
 lock=read(BASE/'RUNTIME_LOCK.json')
 for rel,h in lock['pins'].items():assert sha(ROOT/rel)==h,rel
 for rel,h in lock['metadata'].items():assert sha(BASE/rel)==h,rel
 for rel,h in lock['predecessors'].items():assert sha(ROOT/rel)==h,rel
 return read(BASE/dataset/'PLAN.json') if dataset else lock
from scripts.tastvg_extended_common_v3 import guard,bind_decode
