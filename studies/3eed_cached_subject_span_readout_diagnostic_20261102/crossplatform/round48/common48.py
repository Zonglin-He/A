"""R48 old-development-only gates. No model, annotation, or final loader."""
from pathlib import Path
import json,hashlib,datetime,os,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OLD_ALL='research/round24/input/ALL336_PRIVATE.json'
SPLITS={k:'research/round24/input/'+k+'_PRIVATE.json' for k in ('D208','V64','M64')}
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def npz(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def seal(folder,state,**kw):
 p=Path(folder);dump(p/'SEALED.json',dict(state=state,files={x.relative_to(p).as_posix():sha(x) for x in sorted(p.rglob('*')) if x.is_file() and x!=p/'SEALED.json'},created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw))
def verify(folder):
 folder=Path(folder);s=read(folder/'SEALED.json')
 for rel,h in s['files'].items():assert not Path(rel).is_absolute() and '..' not in Path(rel).parts and sha(folder/rel)==h
 return s
def args():
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--authorization',required=True);return p.parse_args()
def context(a):
 auth=read(a.authorization);assert auth['state']=='ROOT_AUTHORIZED_R48_CPU_PARSE_AND_PREDICT_NO_GTBOX_NO_FINAL' and auth['plan_sha256']==sha(a.plan)
 plan=read(a.plan);assert plan['semantic_lock_sha256']==sha(ROOT/'research/round48/experiment/SEMANTIC_PLAN_LOCK.json')
 for rel,h in plan['program_sha256'].items():assert sha(ROOT/rel)==h,rel
 for rel,h in plan.get('input_sha256',{}).items():assert sha(ROOT/rel)==h,rel
 assert set(plan['manifest_sha256'])=={OLD_ALL,*SPLITS.values()}
 for rel,h in plan['manifest_sha256'].items():assert sha(ROOT/rel)==h,rel
 rows=read(ROOT/OLD_ALL);assert len(rows)==336 and len({r['id'] for r in rows})==336;membership={}
 for split,path in SPLITS.items():
  rr=read(ROOT/path);assert len(rr)==(208 if split=='D208' else 64)
  for r in rr:assert r['id'] not in membership;membership[r['id']]=split
 assert set(membership)=={r['id'] for r in rows};return plan,rows,membership,auth
