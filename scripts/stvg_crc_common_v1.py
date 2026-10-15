"""Inert bindings for fixed-pool contrastive referent correction."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_native_instance_common_v1 import read, write, sha, guard
from scripts.stvg_evidence_common_v1 import load_case, native_arm
from scripts.stvg_ecld_common_v1 import save_pt
BASE = ROOT / 'artifacts/stvg_contrastive_correction_v1'
OLD = ROOT / 'artifacts/stvg_evidence_constrained_v1'
PARENT = OLD / 'round2_prompt009'
C = ROOT / 'artifacts/stvg_weightfree_coordinate_v1'

def roster():
    return read(BASE / 'ROSTER_LOCK.json')['records']

def verify_runtime():
    d = read(BASE / 'RUNTIME.json')
    assert d['status'] == 'locked' and d['GT_read'] is False
    for p, h in {**d['pins'], **d.get('protected', {})}.items():
        assert sha(ROOT / p) == h, p
    assert read(BASE/'DESIGN_LOCK.json')['status'] == 'locked'
    return d
