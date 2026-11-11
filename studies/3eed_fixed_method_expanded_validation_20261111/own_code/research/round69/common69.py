from pathlib import Path
import json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[2]
ARMS=['source','frozen_ABC','R59','R66_R59','R66_E_direct','R66_set_direct','random_s0','random_s1','random_s2','C_source','C_R66']
REFS=['source','frozen_ABC','R59','R66_R59']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def load(p,keys=None):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in (z.files if keys is None else keys)}
def seal(out,state,**extra):dump(Path(out)/'SEALED.json',{'state':state,**extra,'files':{p.relative_to(out).as_posix():sha(p) for p in sorted(Path(out).rglob('*')) if p.is_file() and p.name!='SEALED.json'}})
def pin(p):
 for group in ['programs_sha256','inputs_sha256']:
  for f,h in p.get(group,{}).items():assert sha(ROOT/f)==h,('pin',f)
def bseed(r):return int(hashlib.sha256(('R50episodic_B_v1|'+r['id']+'|'+str(r['seed'])).encode()).hexdigest()[:16],16)
def cseed(r):return int(hashlib.sha256(('R57_student_C_v1|'+r['id']+'|'+str(r['seed'])).encode()).hexdigest()[:16],16)
