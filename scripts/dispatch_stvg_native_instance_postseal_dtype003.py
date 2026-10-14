"""Restricted CPU NumPy dtype dispatch; original sealed scoring math unchanged."""
import argparse,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
def main():
    p=argparse.ArgumentParser();p.add_argument('target',choices=['score','coverage']);a=p.parse_args()
    import numpy as np
    import torch
    pins={'scripts/score_stvg_native_instance_postseal_v1.py':'8ba5403696075ef3397e7ff85079a6e5ee94ca40a83033ae12b1d4f5ddd5eb24',
          'scripts/score_stvg_native_instance_pool_coverage_v1.py':'4abd858176713fdd6de82bec8abcf8345c59bd1e5ce710fe7b7a0944c4e6e9d3'}
    for name,digest in pins.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    assert not torch.cuda.is_initialized()
    # NumPy scalar types and dtype classes are distinct. Keep weights_only=True;
    # only the nine already-declared numerical dtype classes are added.
    classes=[type(np.dtype(t)) for t in (np.float16,np.float32,np.float64,np.int8,np.int16,np.int32,np.int64,np.uint8,np.bool_)]
    with torch.serialization.safe_globals(classes):
        if a.target=='score':
            from scripts import score_stvg_native_instance_postseal_v1 as task
        else:
            from scripts import score_stvg_native_instance_pool_coverage_v1 as task
        task.main(['--score-after-seal'])
if __name__=='__main__':main()
