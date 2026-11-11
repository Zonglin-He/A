"""R57 same-source EMA heads; no task-label imports or external predictors."""
from pathlib import Path
import json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[2]
ARMS=['source','frozen_ABC','R59','R66_R59_tta1','native_R59','random_R59','feature_R59']
REFS=['source','R59','R66_R59_tta1','native_R59','random_R59']
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
