"""Locked expanded development cohort; annotation-free preparation and recovery."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha,guard
BASE=ROOT/'artifacts/stvg_soft_ep_coordinate128_v1'
OLD=ROOT/'artifacts/stvg_evidence_constrained_v1'
def roster():return read(BASE/'ROSTER_LOCK.json')['records']
def verify_runtime():
 r=read(BASE/'PREPARATION_RUNTIME.json');assert r['status']=='locked' and not r['GT_read']
 for p,h in {**r['pins'],**r['protected']}.items():assert sha(ROOT/p)==h,p
 return r
def load_case(rec):
 from scripts.stvg_evidence_common_v1 import load_case as original
 return original(rec)
def save_pt(p,d):
 from scripts.stvg_evidence_common_v1 import save_pt as original
 return original(p,d)
