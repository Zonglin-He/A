"""CPU serialization adapter only; original sealed inference/scorer hashes retained."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import json,runpy
from pathlib import Path
ns=runpy.run_path(str(Path(__file__).with_name('score.py')),run_name='sealed_cpu_score')
original=ns['put']
def write_plain(path,value):
    plain=json.loads(json.dumps(value,default=lambda x:x.item(),allow_nan=False))
    original(path,plain)
ns['run'].__globals__['put']=write_plain
ns['run']()
