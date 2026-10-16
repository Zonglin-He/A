"""Independent CPU-reader recovery custody; original runtime remains pinned."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.stvg_lite_event_common_v1 import BASE,read,sha,verify


def verify_recovery():
    verify()
    old=read(BASE/'FINITE_TAIL_RUNTIME.json')
    for rel,h in {**old['pins'],**old['protected']}.items():assert sha(ROOT/rel)==h,rel
    d=read(BASE/'RECOVERY_RUNTIME002.json')
    assert d['status']=='locked' and not d['GT_read'] and d['scientific_algorithm_changed']is False
    for rel,h in {**d['pins'],**d['protected']}.items():assert sha(ROOT/rel)==h,rel
    return d
