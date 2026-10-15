"""Inert bindings for the bounded evidence-conditioned development study."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_native_instance_common_v1 import read, write, sha, guard
from scripts.stvg_evidence_common_v1 import load_case, native_arm

BASE = ROOT / 'artifacts/stvg_evidence_conditioned_v1'
OLD = ROOT / 'artifacts/stvg_evidence_constrained_v1'
PARENT = OLD / 'round2_prompt009'
C = ROOT / 'artifacts/stvg_weightfree_coordinate_v1'


def roster():
    return read(BASE / 'ROSTER_LOCK.json')['records']


def verify_runtime():
    runtime = read(BASE / 'RUNTIME.json')
    assert runtime['status'] == 'locked' and runtime['GT_read'] is False
    for p, h in {**runtime['pins'], **runtime.get('protected', {})}.items():
        assert sha(ROOT / p) == h, p
    design = read(BASE / 'DESIGN_LOCK.json')
    assert design['status'] == 'locked' and not design['bootstrap_executed']
    return runtime


def save_pt(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists()
    torch.save(value, path)
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()
