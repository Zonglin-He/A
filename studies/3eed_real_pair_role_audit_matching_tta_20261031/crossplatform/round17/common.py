"""R17 shared serialization. Input/model stages never import a task scorer."""
from pathlib import Path
import hashlib,json,os
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'research/round17/experiment'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def serial(x):
 if isinstance(x,dict):return {str(k):serial(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [serial(v) for v in x]
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 return x
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(serial(x),indent=2,allow_nan=False)+'\n');os.replace(t,p)
def arrsha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def seal(folder,state,**extra):
 folder=Path(folder);files={str(p.relative_to(folder)):sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='SEALED.json'}
 dump(folder/'SEALED.json',dict(state=state,files=files,**extra));return sha(folder/'SEALED.json')
def verify(folder):
 folder=Path(folder);s=read(folder/'SEALED.json')
 for rel,h in s['files'].items():
  p=Path(rel);assert not p.is_absolute() and '..' not in p.parts
  assert sha(folder/p)==h,(folder,rel)
 return s
def check_plan():
 p=read(EXP/'PLAN_LOCK.json')
 for rel,h in p['programs'].items():assert sha(ROOT/rel)==h,rel
 for rel,h in p['inputs'].items():assert sha(ROOT/rel)==h,rel
 assert p['authorized_execution'] and p['no_teacher_EMA'] and p['boxes_frozen']
 return p
