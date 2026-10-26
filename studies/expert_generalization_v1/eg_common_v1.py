"""Unique independent namespace for referring-video expert evidence."""
import os,sys,json,hashlib,time
from pathlib import Path
from functools import lru_cache
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
R=Path(__file__).resolve().parents[2];S=Path(__file__).parent
sys.path[:0]=[str(R/'studies/review_strengthening_v1'),str(R),str(R/'tools')]
import review_common as rc
sys.path.insert(0,str(S))
D=R/'outputs/expert_generalization_v1';P=R/'results/expert_generalization_v1'
read=rc.read;sha=rc.sha;atomic=rc.atomic;identity=rc.identity;fixed=lru_cache(None)(read)
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def validate_math():
 p=read(D/'math_plan_v1.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for field in ['code_sha256','input_sha256']:
  for f,h in p[field].items():assert sha(R/f)==h,f
 rc.validate();return p
def pilot_rows(ds):return fixed(R/'outputs/paper_review_suite_v1/selection.json')[ds]['confirm']
def resources(start):
 import shutil
 assert time.perf_counter()-start<259200
 assert shutil.disk_usage(R).free>=80*2**30
 assert next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:'))>=8*2**30
 assert sum(p.stat().st_size for p in D.rglob('*')if p.is_file())<160*2**30
