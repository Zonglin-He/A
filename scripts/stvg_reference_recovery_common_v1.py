"""Inert bindings for matched episodic versus online LN-carry recovery."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha,guard
from scripts.stvg_evidence_common_v1 import load_case,native_arm
from scripts.stvg_ecld_common_v1 import save_pt
BASE=ROOT/'artifacts/stvg_reference_recovery_v1/LN_carry'
PARENT=ROOT/'artifacts/stvg_evidence_constrained_v1/round2_prompt009'
OLD=ROOT/'artifacts/stvg_evidence_constrained_v1'
def roster():return read(BASE/'ROSTER_LOCK.json')['records']
def verify_runtime():
 r=read(BASE/'RUNTIME.json');assert r['status']=='locked' and not r['GT_read']
 for p,h in {**r['pins'],**r['protected']}.items():assert sha(ROOT/p)==h,p
 return r
