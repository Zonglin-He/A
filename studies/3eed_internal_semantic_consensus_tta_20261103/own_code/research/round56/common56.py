"""R56 frozen internal semantic consensus; no GT or instructions from samples."""
from pathlib import Path
import json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[2]
ARMS=['source','zero_step','proposal']+['decoder'+str(i) for i in range(5)]+['early_consensus','final_role','early_role_consensus','phrase_subject','phrase_role','semantic_consensus','shuffled_consensus','tta1','tta3','shuffle_tta1','shuffle_tta3']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def load(p,keys=None):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in (z.files if keys is None else keys)}
def pin(p):
 for g in ['programs_sha256','inputs_sha256']:
  for f,h in p.get(g,{}).items():assert sha(ROOT/f)==h,('pin',f)
def seal(out,state,**extra):dump(out/'SEALED.json',{'state':state,**extra,'files':{f.relative_to(out).as_posix():sha(f) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='SEALED.json'}})
