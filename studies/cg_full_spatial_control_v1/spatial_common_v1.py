"""Independent HC complete-spatial control; all parent namespaces read-only."""
import os,sys,time,json,hashlib,shutil
from pathlib import Path
from functools import lru_cache
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
R=Path(__file__).resolve().parents[2];S=Path(__file__).parent
sys.path[:0]=[str(R/'studies/shared_corrected_roi_CG_v1'),str(R),str(R/'tools')]
import cg_common as base
import cg_core as conservative
sys.path.insert(0,str(S))
read=base.read;sha=base.sha;atomic=base.atomic;identity=base.identity;digest=base.digest;fixed=lru_cache(None)(read)
D=R/'outputs/cg_full_spatial_control_v1';P=R/'results/cg_full_spatial_control_v1'
PROFILE=dict(calibration='Transferred',gamma=.5,unobserved=.5,link=.25,beta=.5,kappa=0,context=1.5)
ARMS=['Source','SourceJointMAP','Conservative','FullSpatial'];PAIRS=[('FullSpatial',x)for x in ARMS if x!='FullSpatial']

def validate():
 p=read(D/'science_plan_v1.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for field in ['code_sha256','input_sha256']:
  for f,h in p[field].items():assert sha(R/f)==h,f
 base.validate();return p

def resources(start):
 assert time.perf_counter()-start<259200 and shutil.disk_usage(R).free>=80*2**30
 assert next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:'))>=8*2**30
 assert sum(p.stat().st_size for p in D.rglob('*')if p.is_file())<160*2**30

def save(path,obj):
 import torch
 path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();tmp=path.with_suffix('.tmp');torch.save(obj,tmp);os.replace(tmp,path)

def committed(key):
 rp=D/'hc2/receipts'/(key+'.json')
 if not rp.exists():return None
 e=read(rp)
 for f,h in [('path','sha256'),('compact_path','compact_sha256')]:assert sha(D/e[f])==e[h]
 return dict(e,receipt_path=str(rp.relative_to(D)),receipt_sha256=sha(rp))
