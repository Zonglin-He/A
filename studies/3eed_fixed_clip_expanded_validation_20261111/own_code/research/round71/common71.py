from pathlib import Path
import json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[2];H=ROOT/'research/round71'
ARMS=['source','frozen_ABC','R59','CLIP_direct','CLIP_mismatch'];REFS=['source','frozen_ABC','R59','CLIP_direct']
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
def rows():return read(ROOT/'research/round69/input/ALL2304_PRIVATE.json')
def devrows():return read(ROOT/'research/round24/input/ALL336_PRIVATE.json')
def perm(n,salt):return np.random.default_rng(int(hashlib.sha256(salt.encode()).hexdigest()[:16],16)).permutation(n)
def prefix(r):
 with np.load(ROOT/'crossplatform/round69/AB2304_attempt001/samples_PRIVATE'/(r['id']+'.npz'),allow_pickle=False) as z:return {'boxes':z['A_boxes'].copy(),'scores':z['A_scores'].copy()}
def control_evidence(a,c,shuffle=False):
 sim=c['visual'].mean(1)@c['subject_text'];r=a['project_reliability'];valid=np.flatnonzero(r>0)
 if shuffle:sim=sim.copy();sim[valid]=sim[valid][perm(len(valid),'R70Avisualbinding')]
 center=float(np.sum(r*sim)/r.sum()) if r.sum()>0 else 0.;return (r*(sim-center)*float(c['logit_scale']))[a['query_group']]
