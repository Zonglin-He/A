"""Verify the immutable initial lock plus explicitly recorded bug-fix receipts."""
from pathlib import Path
from scripts.decota_matrix_common_v1 import read,sha
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/decota_refine_v1'


def verify_code(p):
    pins=dict(p['pins'])
    for path in sorted((OUT/'amendments').glob('*_applied.json')):
        amendment=read(path)
        for f,c in amendment['changes'].items():
            assert pins.get(f)==c['before'],f
            pins[f]=c['after']
    for f,h in pins.items():assert sha(ROOT/f)==h,f
    return pins
