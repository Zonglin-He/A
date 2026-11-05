"""Independent matched-expert experiment; historical experiments are read-only."""
from pathlib import Path
import os,sys,json,time,hashlib,shutil,fcntl,subprocess
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
R=Path(__file__).resolve().parents[2];S=Path(__file__).parent
D=R/'outputs/expert_evidence_controls_v1';P=R/'results/expert_evidence_controls_v1'
F=R/'outputs/fresh_external_comparison_v1'
V=R/'outputs/review_strengthening_v1/vendor/sam2'
WEIGHT=V.parent/'sam2.1_hiera_large.pt';EXT=R/'outputs/edgetam_comparison_v1/extensions/sam2/_C.so'
PYTHON=R/'outputs/review_strengthening_v1/sam2_runtime/bin/python'
sys.path[:0]=[str(R),str(R/'tools')]
from full import read,sha,atomic,identity
from functools import lru_cache
import numpy as np,torch
fixed=lru_cache(None)(read)
ARMS=['Full','SAM2MatchedMask'];PAIRS=[('Full','SAM2MatchedMask')]
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();t=p.with_suffix('.tmp');torch.save(x,t);os.replace(t,p)
def rows(ds):return fixed(D/'selection.json')[ds]
def reference(ds,row):
 e=fixed(D/'references.json')[ds][row['id']]
 assert sha(F/e['path'])==e['sha256']
 with np.load(F/e['path'])as z:a={k:z[k].copy()for k in ['Source','Source_interval','Ours','Ours_interval','E','frame_ids','positions8']}
 assert a['frame_ids'].tolist()==row['metadata']['frame_ids']
 t=e['current_teacher'];assert sha(t['payload_path'])==t['payload_sha256']and sha(t['receipt_path'])==t['receipt_sha256']
 teacher=torch.load(t['payload_path'],map_location='cpu',weights_only=False,mmap=True)
 meta=read(t['receipt_path']);assert meta['context']==e['input_binding']['context']and meta['key']==e['input_binding']['evidence_key']
 assert teacher['native_same_track_all8']and teacher['nested_mask_all_false']
 return a,teacher,e
def data(ds,row):
 from stvg.decode import decode
 a,t,e=reference(ds,row);frames,ids=decode(row['metadata'],ds);b=e['input_binding'];pos=a['positions8'].tolist()
 assert ids==a['frame_ids'].tolist()and pos==b['positions8']
 assert hashlib.sha256(frames.tobytes()).hexdigest()==b['actual_native_RGB_sha256']
 assert hashlib.sha256(frames[pos].tobytes()).hexdigest()==b['actual_eight_RGB_sha256']
 binding=dict(reference_prediction_sha256=e['sha256'],teacher_payload_sha256=e['current_teacher']['payload_sha256'],teacher_receipt_sha256=e['current_teacher']['receipt_sha256'],native_RGB_sha256=b['actual_native_RGB_sha256'],eight_RGB_sha256=b['actual_eight_RGB_sha256'],positions8=pos,frame_ids=ids,Full_interval_sha256=hashlib.sha256(a['Ours_interval'].tobytes()).hexdigest(),GT_read=False)
 return frames,a,t,binding
def resource(start):
 assert time.perf_counter()-start<72*3600 and shutil.disk_usage(R).free>=80*2**30
 assert next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()if l.startswith('MemAvailable:'))>=8*2**30
 assert sum(f.stat().st_size for f in D.rglob('*')if f.is_file())<160*2**30
def paused_owner():
 proof=read(D/'paused_owner_contract.json');out=[]
 for p in proof['owners']:
  st=Path(f"/proc/{p['pid']}/stat").read_text().split();argv=Path(f"/proc/{p['pid']}/cmdline").read_bytes().replace(b'\0',b' ').decode().strip()
  assert int(st[21])==p['startticks']and st[2]=='T'and argv==p['argv'],('paused owner changed',p['pid'])
  out.append(dict(pid=p['pid'],startticks=int(st[21]),state=st[2],argv=argv))
 assert sha(R/proof['pause_receipt'])==proof['pause_receipt_sha256']
 assert sha(R/proof['old_plan'])==proof['old_plan_sha256']
 # The original flock remains held. Never claim this worker acquired it.
 original=(R/'runtime/gpu.lock').open('a')
 try:
  try:fcntl.flock(original,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:pass
  else:raise AssertionError('Expected preserved stopped owner to retain original lease')
 finally:original.close()
 return out
def foreign():
 import importlib.util
 spec=importlib.util.spec_from_file_location('_matched_protected_gate',R/'studies/expert_subset_followups_v1/protected_service_gate_runtime_v2.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 proof=paused_owner();allowed={p['pid']for p in proof};out=[]
 for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).splitlines():
  if not line.strip():continue
  pid,name=line.split(',',1);pid=int(pid)
  if pid in allowed:continue
  if pid!=os.getpid():out.append(m.identity_from_proc(pid,name.strip()))
 return dict(stopped_owner=proof,other_allowed_services=out)
def lease():
 # Explicit cooperative reservation while a stopped OWN worker retains paid
 # in-memory state and its original flock. No old worker is resumed/signalled.
 f=(R/'runtime/gpu.paused_owner_borrow.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 foreign();return f
def check():
 p=read(D/'propagation_plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for group in ['code_sha256','input_sha256']:
  for f,h in p[group].items():assert sha(R/f)==h,f
 return p
