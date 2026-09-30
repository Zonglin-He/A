"""Required HC2 same-domain panel; parent science remains frozen."""
from pathlib import Path
from scripts.tastvg_paper48_common_v1 import BASE as PARENT, verify as parent_verify
from scripts.decota_matrix_common_v1 import read,write,sha
ROOT=Path(__file__).resolve().parents[1]
BASE=PARENT/'P5'
CHECKPOINT='checkpoints/TASTVG_HCSTVG2.pth'
CHECKPOINT_SHA='47d8f15841cd57e7bbf5a10e8bf23b1054d23b753e0becbd38a07f3dd60d5036'

def frame_ids(count):
    # Same official test sampler with this project's frozen 64-frame runtime.
    rate=64./count;ids=[0]
    for i in range(count-1):
        if int(ids[-1]*rate)<int(i*rate):ids.append(i)
    if ids[-1]!=count-2:ids.append(count-2)
    return ids

def verify(expert=False):
    parent_verify();lock=read(BASE/'LOCK.json')
    for f,h in lock['files'].items():assert sha(ROOT/f)==h,f
    assert sha(ROOT/CHECKPOINT)==CHECKPOINT_SHA
    return read(BASE/('EXPERT_PLAN.json' if expert else 'PLAN.json'))
