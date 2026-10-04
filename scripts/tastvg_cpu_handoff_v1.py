"""Verify an independently pinned CPU scoring implementation before label access."""
from pathlib import Path
from scripts.decota_matrix_common_v1 import read,sha

ROOT=Path(__file__).resolve().parents[1]

def verify_cpu(base):
    base=Path(base);lock=read(base/'CPU_IMPLEMENTATION_LOCK.json')
    assert lock['GT_read'] is False
    pins=dict(lock['pins'])
    for f in sorted((base/'cpu_revisions').glob('*.json')):
        pins.update(read(f)['pin_overrides'])
    for f,h in pins.items():
        assert sha(ROOT/f)==h,('CPU implementation changed without a retained revision',f)
    return lock
