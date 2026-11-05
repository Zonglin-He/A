"""Full-list extension of the immutable Source-feature VideoWalk128 pilot."""
from pathlib import Path
import os,sys,json,time,hashlib,shutil,subprocess,fcntl
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
R=Path(__file__).resolve().parents[2];S=Path(__file__).parent;D=R/'outputs/videowalk_full_v1';P=R/'results/videowalk_full_v1'
RUN=R/'outputs/simplepath_full_v2';OLD=R/'outputs/spatial_temporal_six_v1';PILOT=OLD/'matching_pilot';PYTHON=R/'.conda/tubedetr/bin/python'
sys.path[:0]=[str(R),str(R/'tools')]
from full import read,sha,atomic,identity
from functools import lru_cache
import numpy as np,torch
fixed=lru_cache(None)(read)
COUNTS={'hc2':1902,'vidstg':10303};ARMS=['Matching','Current','PureMatched'];PAIRS=[('Current','Matching'),('Matching','PureMatched'),('Current','PureMatched')];SHARD=32
import importlib.util
spec=importlib.util.spec_from_file_location('immutable_videowalk_adapter',R/'studies/spatial_temporal_six_v1/matching.py');matching=importlib.util.module_from_spec(spec);spec.loader.exec_module(matching)
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();tmp=p.with_suffix('.tmp');torch.save(x,tmp);os.replace(tmp,p)
def rows(ds):return fixed(D/'selection.json')[ds]
def load_parent(ds,row):
 e=fixed(RUN/ds/'sealed.json')['files'][row['id']];cp=RUN/ds/e['capture_path'];pp=RUN/ds/e['path'];rp=RUN/ds/e['receipt_path']
 assert sha(cp)==e['capture_sha256']and sha(pp)==e['sha256']and sha(rp)==e['receipt_sha256'];cap=torch.load(cp,map_location='cpu',weights_only=False,mmap=True);assert not cap['GT_read']
 return cap,dict(capture=str(cp),capture_sha256=e['capture_sha256'],prediction_sha256=e['sha256'],receipt_sha256=e['receipt_sha256'])
def controls(ds,key):
 e=fixed(OLD/'pure_full'/ds/'sealed.json')['files'][key];assert sha(OLD/e['path'])==e['sha256']
 with np.load(OLD/e['path'])as z:a={k:z[k].copy()for k in ['frame_ids','positions8','Current','Current_interval','PureMatched','PureMatched_interval']}
 assert np.array_equal(a['Current_interval'],a['PureMatched_interval']);return a,e['sha256']
def compact(cap,features,valid,shape,row):
 from stvg.spatial import SpatialGeometry
 g=SpatialGeometry.from_cached_geometry(cap['geometry']);assert tuple(shape)==(g.height,g.width)and features.shape==(len(cap['frame_ids']),g.height*g.width,256)
 return dict(features=features,valid=valid,frame_ids=list(cap['frame_ids']),positions8=list(cap['positions8']),seeds=g.side_maps[cap['positions8'],0].clone(),native_boxes=np.asarray(g.native_boxes).copy(),shape=tuple(shape),image_size=(row['metadata']['height'],row['metadata']['width']),GT_read=False)
def replay(v,ds):return matching.propagate(v['features'],v['valid'],v['positions8'],v['seeds'],v['native_boxes'],v['shape'],v['image_size'],**fixed(D/'plan.json')['source_config'][ds])
def resources(start):
 assert time.perf_counter()-start<72*3600 and shutil.disk_usage(R).free>=80*2**30
 assert next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:'))>=8*2**30
 assert sum(p.stat().st_size for p in D.rglob('*')if p.is_file())<160*2**30
def check():
 p=read(D/'plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for group in ['code_sha256','input_sha256']:
  for f,h in p[group].items():assert sha(R/f)==h,f
 return p
def stopped_owner():
 j=fixed(D/'paused_owner_contract.json')
 for e in j['owners']:
  st=Path(f"/proc/{e['pid']}/stat").read_text().split();cmd=Path(f"/proc/{e['pid']}/cmdline").read_bytes().replace(b'\0',b' ').decode().strip();assert int(st[21])==e['startticks']and st[2]=='T'and cmd==e['argv']
 e=j['GPU_lock_owner'];assert Path(f"/proc/{e['pid']}/fd/{e['fd']}").resolve()==(R/'runtime/gpu.lock').resolve();assert f"WRITE {e['pid']} "in Path(f"/proc/{e['pid']}/fdinfo/{e['fd']}").read_text();return j
def foreign():
 j=stopped_owner();allowed={e['pid']for e in j['owners']};sp=importlib.util.spec_from_file_location('vw_protected_service_gate',R/'studies/expert_subset_followups_v1/protected_service_gate_runtime_v2.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
 for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).splitlines():
  if line.strip():
   pid,name=line.split(',',1);pid=int(pid)
   if pid!=os.getpid()and pid not in allowed:m.identity_from_proc(pid,name.strip())
def lease():
 f=(R/'runtime/gpu.paused_owner_borrow.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);foreign();return f
