"""R61 sealed causal sensitivity experiment; runtime imports no task labels."""
from pathlib import Path
import json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[2]
REFERENCES=['source','frozen_ABC','r58_online_ema_teacher','r58_online_ema_student','r59_direct_all','r60_noise','r60_common']
HEAD_ARMS=['mlp_noise8_off','mlp_noise32_off','mlp_common8_off','mlp_random8_off','mlp_h1_energy32_off','mlp_h2_energy32_off']
DECODER_ARMS=[f'dec_{block}_{gain}' for block in ['self','language','vision','ffn'] for gain in ['off','half','boost']]
HEAD_OFF_ARMS=['dec_vision_head'+str(i)+'_off' for i in range(8)]
ARMS=REFERENCES+['zero_replay']+HEAD_ARMS+DECODER_ARMS+HEAD_OFF_ARMS
REFS=['source','frozen_ABC','r59_direct_all']
assert len(ARMS)==34
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def load(p,keys=None):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in (z.files if keys is None else keys)}
def pin(p):
 for group in ['programs_sha256','inputs_sha256']:
  for f,h in p.get(group,{}).items():assert sha(ROOT/f)==h,('pin',f)
def seal(out,state,**extra):dump(out/'SEALED.json',{'state':state,**extra,'files':{f.relative_to(out).as_posix():sha(f) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='SEALED.json'}})
def engineer(rows):return [min([r for r in rows if r['platform']==p],key=lambda r:hashlib.sha256(('R61_engineering|'+r['id']).encode()).hexdigest()) for p in ['drone','quad']]
