"""R57 same-source EMA heads; no task-label imports or external predictors."""
from pathlib import Path
import json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[2]
GAMMAS=[0.,.5,1.,1.5,2.]
ETAS=[.25,.5,.75,1.]
def tag(x):return str(x).replace('.','p')
ARMS=['source','R59','R66','C_source']+['G_'+tag(x) for x in GAMMAS]+['E_'+tag(x) for x in ETAS]+['C_E_'+tag(x) for x in ETAS]+['random_s0','random_s1','random_s2']
REFS=['source','R59','R66','G_0p0','random_s0']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def load(p,keys=None):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in (z.files if keys is None else keys)}
def pin(p):
 for group in ['programs_sha256','inputs_sha256']:
  for f,h in p.get(group,{}).items():assert sha(ROOT/f)==h,('pin',f)
def seal(out,state,**extra):dump(out/'SEALED.json',{'state':state,**extra,'files':{f.relative_to(out).as_posix():sha(f) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='SEALED.json'}})
def seed(row):return int(hashlib.sha256(('R57_student_C_v1|'+row['id']+'|'+str(row['seed'])).encode()).hexdigest()[:16],16)
