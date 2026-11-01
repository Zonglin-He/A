"""R41 fixed U128 cached numerical readout; no torch, media, labels or models."""
import json,hashlib,sys,datetime
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'crossplatform/round41';EXP=ROOT/'research/round41/experiment';REVIEW=ROOT/'research/round41/review'
sys.path.insert(0,str(REVIEW))
from readout41_draft import ARMS,PRIMARY_ARMS,readouts_from_existing_U_cache,proposed_stop_rule
OLD=ROOT/'crossplatform/round35/predictionsU128_attempt001';SOURCE=ROOT/'crossplatform/round33/sourceU128_attempt001';OUT=BASE/'cachedU128_attempt001'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def npz(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def verify(folder):
 s=read(folder/'SEALED.json')
 for f,h in s['files'].items():assert sha(folder/f)==h,(str(folder),f)
 return s
def seal(folder,state,**meta):
 assert not (folder/'SEALED.json').exists();files={str(p.relative_to(folder)):sha(p) for p in sorted(folder.rglob('*')) if p.is_file()};dump(folder/'SEALED.json',dict(state=state,files=files,sealed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**meta))
def check_authorization():
 a=read(EXP/'ROOT_EXECUTION_AUTHORIZATION.json');assert a['state']=='ROOT_AUTHORIZED_CACHED_CPU_DIAGNOSTIC'
 for p,h in a['refs_sha256'].items():assert sha(ROOT/p)==h,p
 lock=read(EXP/'IMPLEMENTATION_LOCK.json');assert lock['root_authorization_sha256']==sha(EXP/'ROOT_EXECUTION_AUTHORIZATION.json')
 assert lock['proposed_plan_sha256']==sha(REVIEW/'PROPOSED_PLAN.json')
 for p,h in dict(lock['program_sha256'],**lock['input_sha256']).items():assert sha(ROOT/p)==h,p
 return lock

def U_rows():
 rr=read(ROOT/'research/NIGHT_20261101/input/U128_PRIVATE.json');assert len(rr)==128 and len({r['id'] for r in rr})==128 and all(r['stage']=='U' for r in rr);return rr

def check_states_and_inputs():
 verify(OLD);verify(SOURCE);verify(ROOT/'crossplatform/round35/partial_keep_attempt001');verify(ROOT/'crossplatform/round30/screen_attempt001')
 for pl in ['drone','quad']:
  a=npz(ROOT/'crossplatform/round35/partial_keep_attempt001/states_PRIVATE'/(pl+'_R30_U_rows.npz'));b=npz(ROOT/'crossplatform/round30/screen_attempt001/states_PRIVATE'/(pl+'_b1.npz'));assert a.keys()==b.keys() and all(np.array_equal(a[k],b[k]) for k in a),'R30_head_not_bitexact'
 assert read(ROOT/'research/round35/experiment/CPU_AUDIT.json')['state']=='PASSED' and read(ROOT/'research/round35/experiment/U128_CPU_AUDIT.json')['state']=='PASSED'
