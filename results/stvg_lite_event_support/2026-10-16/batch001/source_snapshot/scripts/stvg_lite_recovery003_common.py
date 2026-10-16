"""Bind original source-cache CPU environment without changing scientific code."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.stvg_lite_event_common_v1 import BASE,read,sha
from scripts.stvg_lite_recovery002_common import verify_recovery


def verify_recovery003():
    verify_recovery()
    d=read(BASE/'RECOVERY_RUNTIME003.json')
    assert d['status']=='locked' and d['GT_read']is False and not d['scientific_algorithm_changed']
    for rel,h in {**d['pins'],**d['protected']}.items():assert sha(ROOT/rel)==h,rel
    return d
