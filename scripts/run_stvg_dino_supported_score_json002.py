"""JSON-boundary recovery only; pinned scientific scorer and predictions unchanged."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import score_stvg_dino_supported_correction32_v1 as score
from scripts.run_stvg_joint_event_critic_v1 import put


def native(x):
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,np.ndarray):return native(x.tolist())
    if isinstance(x,dict):return {k:native(v)for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [native(v)for v in x]
    return x


def write(path,value,mutable=False):return put(path,native(value),mutable)


if __name__=='__main__':
    score.put=write
    score.score()
