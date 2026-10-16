"""Reference tuple interface recovery; old science and runtime stay immutable."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.stvg_lite_event_common_v1 import BASE,read,sha
from scripts.stvg_lite_recovery004_common import verify_recovery004


def verify_recovery005():
    verify_recovery004()
    d=read(BASE/'RECOVERY_RUNTIME005.json')
    assert d['status']=='locked' and d['GT_read'] is False and d['scientific_algorithm_changed'] is False
    for rel,h in {**d['pins'],**d['protected']}.items():assert sha(ROOT/rel)==h,rel
    return d
