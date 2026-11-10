"""R62 sequential fixed-design combination/layer-location/semantic tests."""
from pathlib import Path
import json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[2]
COMBO_ARMS=['source','frozen_ABC','r59_direct_all','r61_last15','zero_ABC','zero_R59','native_l5','R59_l5']
LAYER_ARMS=COMBO_ARMS+['native_l'+str(i) for i in range(5)]
SEMANTIC_ARMS=['frozen_ABC','selected_native15','selected_value1','selected_value15','selected_increment_value','selected_increment_uniform','selected_increment_joint']
REFS=['source','frozen_ABC','r59_direct_all']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def load(p,keys=None):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in (z.files if keys is None else keys)}
def pin(p):
 for group in ['programs_sha256','inputs_sha256']:
  for f,h in p.get(group,{}).items():assert sha(ROOT/f)==h,('pin',f)
def seal(out,state,**extra):dump(out/'SEALED.json',{'state':state,**extra,'files':{f.relative_to(out).as_posix():sha(f) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='SEALED.json'}})
def engineering_rows(rows):return [min([r for r in rows if r['platform']==p],key=lambda r:hashlib.sha256(('R62_engineering|'+r['id']).encode()).hexdigest()) for p in ['drone','quad']]
