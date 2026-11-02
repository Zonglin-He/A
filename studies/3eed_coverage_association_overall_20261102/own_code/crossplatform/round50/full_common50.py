"""Run-local R50 full frontend I/O. Import opens no cohort or media.

Each stage has its own root grant/hash lock. No closed FINAL registration is
an input: the existing R8 full cohort is decoded only after this new grant.
"""
from pathlib import Path
import json,os,hashlib,datetime,argparse
from decimal import Decimal
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'research/round50/experiment'
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
 if isinstance(x,Decimal):return float(x)
 return x
def dump(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
 t.write_text(json.dumps(serial(v),indent=2,allow_nan=False)+'\n');os.replace(t,p)
def npz(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy(order='K') for k in z.files}
def seal(folder,state,**kw):
 folder=Path(folder);dump(folder/'SEALED.json',dict(state=state,files={p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='SEALED.json'},created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw))
def verify(folder):
 folder=Path(folder);s=read(folder/'SEALED.json')
 for rel,h in s['files'].items():
  assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
  assert sha(folder/rel)==h,(folder,rel)
 return s
def verified_file(folder,rel):
 folder=Path(folder);s=read(folder/'SEALED.json');assert s['files'][rel]==sha(folder/rel)
 return folder/rel
def args(description):
 p=argparse.ArgumentParser(description=description)
 for k in ('plan','authorization'):p.add_argument('--'+k,required=True)
 p.add_argument('--chunk',type=int);p.add_argument('--finalize',action='store_true');return p.parse_args()
def context(a,stage):
 """Hash gates precede full manifest decoding; stage grants never imply GT."""
 pp=Path(a.plan).resolve();ap=Path(a.authorization).resolve();auth=read(ap)
 assert auth['state']=='ROOT_AUTHORIZED_R50_FULL5912_'+stage.upper()
 assert auth['plan_sha256']==sha(pp)
 plan=read(pp);assert plan['stage']==stage and plan['n']==5912
 assert plan['selected_configuration']=='wideentry_oldW'
 assert plan['manifest']=='research/round8/experiment/full5912_gpu_inputs.json'
 assert plan['source_cache']=='crossplatform/round8/full5912_attempt001'
 assert plan['no_GT_or_protected_FINAL_specific_inputs'] is True
 fp=EXP/'FULL_METHOD_FREEZE_BEFORE_FULL_MEDIA.json';assert sha(fp)==plan['scientific_freeze_sha256']
 freeze=read(fp);assert freeze['state']=='ROOT_FROZEN_R50_FULL5912_METHOD_BEFORE_FULL_MEDIA_OR_NEW_PREDICTION'
 assert freeze['selected_configuration']=='wideentry_oldW' and freeze['association'].startswith('literal_oldW')
 for group in ('programs_sha256','inputs_sha256'):
  for rel,h in plan[group].items():
   assert 'round46/final' not in rel and 'FINAL384' not in rel
   assert sha(ROOT/rel)==h,rel
 assert plan['manifest'] in plan['inputs_sha256']
 out=(ROOT/plan['output_root']).resolve();assert out.is_relative_to((ROOT/'crossplatform/round50').resolve())
 rows=read(ROOT/plan['manifest']);assert len(rows)==5912 and len({r['id'] for r in rows})==5912
 assert {p:sum(r['platform']==p for r in rows) for p in ('drone','quad')}=={'drone':2984,'quad':2928}
 keys={'id','platform','scene','pointpath','metapath','imagepath','caption','suffix','seed'}
 assert all(set(r)==keys for r in rows)
 assert len({tuple(r[k] for k in ('imagepath','pointpath','metapath')) for r in rows})==5912
 if a.chunk is not None:
  size=plan['chunk_size'];assert isinstance(a.chunk,int) and 0<=a.chunk<(5912+size-1)//size
  rows=rows[a.chunk*size:(a.chunk+1)*size]
 return plan,rows,out
