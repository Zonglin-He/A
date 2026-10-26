"""Engineering-only canonical metadata hashing; numerical run.py remains frozen."""
import run as original
import numpy as np
import argparse

def array_sha(value):
    # Geometry.metadata is a Python dict. np.asarray(dict).tobytes() serializes
    # an object pointer, not its content, and changed after deepcopy in v1.
    # All numeric array byte hashing and numerical computations stay unchanged.
    if isinstance(value, dict):
        return original.digest(value)
    if value is None:
        return original.digest(None)
    a = np.asarray(value)
    assert a.dtype.kind != 'O', 'Never hash an object pointer'
    return original.hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()

def validate_runtime():
    p=original.read(original.D/'execution_v2_plan.json')
    assert p['fingerprint']==original.digest({k:v for k,v in p.items() if k!='fingerprint'})
    for field in ['code_sha256','input_sha256']:
        for f,h in p[field].items():
            assert original.sha(original.R/f)==h,f
    original.validate()
    return p

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['preflight','audit','export']);ap.add_argument('--dataset',choices=original.COUNTS)
    args=ap.parse_args();validate_runtime();original.array_sha=array_sha
    if args.command=='preflight':original.preflight()
    else:
        assert args.dataset is not None
        getattr(original,args.command)(args.dataset)
