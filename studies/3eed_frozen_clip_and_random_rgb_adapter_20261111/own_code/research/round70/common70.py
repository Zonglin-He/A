from pathlib import Path
import json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[2];H=ROOT/'research/round70'
ARMS=['source','A_subject_tight','A_subject_joint','A_full_joint','A_projectonly_joint','A_mismatch_joint','A_adapter3','A_adapter16','B_random0','B_step3','B_step32','B_wrongtest32','B_wrongtrain32','B_NW','B_NW_wrong','B_mean','frozen_ABC','R59']
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def load(p,keys=None):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in (z.files if keys is None else keys)}
def seal(p,state,**kw):p=Path(p);dump(p/'SEALED.json',{'state':state,**kw,'files':{f.relative_to(p).as_posix():sha(f) for f in sorted(p.rglob('*')) if f.is_file() and f.name!='SEALED.json'}})
def pin(p):
 for group in ['programs_sha256','inputs_sha256']:
  for f,h in p.get(group,{}).items():assert sha(ROOT/f)==h,('pin',f)
def rows():return read(ROOT/'research/round24/input/ALL336_PRIVATE.json')
def prefix(row):
 with np.load(ROOT/'crossplatform/round66/prefusion336_attempt001/samples_PRIVATE'/(row['id']+'.npz'),allow_pickle=False) as z:return {k[2:]:z[k] for k in z.files if k.startswith('A_')}
def perm(n,salt):return np.random.default_rng(int(hashlib.sha256(salt.encode()).hexdigest()[:16],16)).permutation(n)
