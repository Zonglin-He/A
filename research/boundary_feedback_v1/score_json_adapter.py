"""Serialization-only recovery; preserve sealed inference and original scorer."""
import os, json, runpy, sys
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
sys.path.insert(0,str(Path(__file__).parent))
ns=runpy.run_path(str(Path(__file__).with_name('score.py')),run_name='sealed_cpu_score')
original=ns['put']
def write_plain(path,value):
    original(path,json.loads(json.dumps(value,default=lambda x:x.item(),allow_nan=False)))
ns['main'].__globals__['put']=write_plain
ns['main']()
