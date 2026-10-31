"""R19 bindings and immutable prediction artifacts."""
from pathlib import Path
import sys,hashlib,json,os
import numpy as np
ROOT=Path(__file__).resolve().parents[2];EXP=ROOT/'research/round19/experiment'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def serial(x):
 if isinstance(x,dict):return {str(k):serial(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [serial(v) for v in x]
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 return x
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(serial(x),indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def arrsha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def seal(p,state,**kw):
 p=Path(p);dump(p/'SEALED.json',dict(state=state,files={f.relative_to(p).as_posix():sha(f) for f in sorted(p.rglob('*')) if f.is_file() and f.name!='SEALED.json'},**kw))
def verify(p):
 p=Path(p);s=read(p/'SEALED.json')
 for f,h in s['files'].items():assert not Path(f).is_absolute() and '..' not in Path(f).parts and sha(p/f)==h
 return s
def locked(name):
 p=read(EXP/name)
 for rel,h in p.get('programs',{}).items():assert sha(ROOT/rel)==h,rel
 for rel,h in p.get('inputs',{}).items():assert sha(ROOT/rel)==h,rel
 return p
def feature(r):
 folder='round17/features96_attempt001' if r['stage']=='A' else 'round19/sourceH32_attempt001'
 with np.load(ROOT/'crossplatform'/folder/'samples_PRIVATE'/(r['id']+'.npz'),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
