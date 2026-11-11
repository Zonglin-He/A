from pathlib import Path
import json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[2];H=ROOT/'research/round72'
ARMS=['source','frozen_ABC','R59','CLIP_reference','graph_direct_mu1','graph_direct_opt','graph_direct_hard','graph_tta1','graph_tta3','RGBshuffle_direct_opt','RGBshuffle_tta3','Nshuffle_direct_opt','Nshuffle_tta3','box_direct_opt','color_only_direct_opt','relation_only_direct_opt']
REFS=['source','R59','CLIP_reference','graph_direct_opt']
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
 for g in ['programs_sha256','inputs_sha256']:
  for name,digest in p.get(g,{}).items():assert sha(ROOT/name)==digest,('pin',name)
def rows():return read(ROOT/'research/round69/input/ALL2304_PRIVATE.json')+read(ROOT/'research/round24/input/ALL336_PRIVATE.json')
def cohort_map():return {**read(ROOT/'research/round69/input/COHORT_BY_ID_PRIVATE.json'),**{r['id']:'DEV336_DIAGNOSTIC' for r in read(ROOT/'research/round24/input/ALL336_PRIVATE.json')}}
def prefix(r,keys=('h','tokens','token_mask','query','boxes','scores')):
 root='crossplatform/round69/AB2304_attempt001' if r['id'] in read(ROOT/'research/round69/input/COHORT_BY_ID_PRIVATE.json') else 'crossplatform/round66/prefusion336_attempt001'
 with np.load(ROOT/root/'samples_PRIVATE'/(r['id']+'.npz'),allow_pickle=False) as z:return {k:z['A_'+k].copy() for k in keys}
def prep(r):
 root='crossplatform/round71/prepared2304_attempt001' if r['id'] in read(ROOT/'research/round69/input/COHORT_BY_ID_PRIVATE.json') else 'crossplatform/round70/prepared336_attempt001'
 return load(ROOT/root/'samples_PRIVATE'/(r['id']+'.npz'),['query_group','group_representatives','rects','rect_valid','project_reliability'])
def perm(n,salt):return np.random.default_rng(int(hashlib.sha256(salt.encode()).hexdigest()[:16],16)).permutation(n)
