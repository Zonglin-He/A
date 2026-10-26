"""Immutable JSON caching + exact reuse of completed CPU-only prediction files."""
from common import *
import run_v1 as original
import argparse,marshal,numpy as np
OLD_READ=original.read;CACHE={}
def cached_read(path):
 p=Path(path)
 if p in [selected.D/f'components_{ds}_sealed.json'for ds in COUNTS]:
  stat=p.stat();sig=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
  if p not in CACHE:CACHE[p]=(sig,sha(p),OLD_READ(p))
  old,h,obj=CACHE[p];assert old==sig,'Frozen input JSON changed during cached read';return obj
 return OLD_READ(p)
def cache_check():
 for p,(stat,h,obj)in CACHE.items():assert sha(p)==h,'Whole frozen JSON SHA recheck'
def execution():
 p=read(D/'execution_v2_plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for field in ['code_sha256','input_sha256']:
  for f,h in p[field].items():assert sha(R/f)==h,f
 validate();return p

def preflight():
 import torch
 torch.set_num_threads(1);gt_guard();start=time.perf_counter();proof=[];raw=marshal.dumps(original.derive.__code__)
 for ds in COUNTS:
  for row in records(ds)[:2]:
   original.read=OLD_READ;t=time.perf_counter();a,p=original.derive(ds,row['id']);before=time.perf_counter()-t;original.read=cached_read;t=time.perf_counter();b,q=original.derive(ds,row['id']);after=time.perf_counter()-t;assert p==q and set(a)==set(b)and all(np.array_equal(v,b[k])for k,v in a.items());proof.append(dict(dataset=ds,original_JSON_parse_seconds=before,cached_JSON_seconds=after,all_arrays_provenance_exact=True))
 assert raw==marshal.dumps(original.derive.__code__);cache_check();atomic(D/'cache_preflight_v2.json',dict(status='passed_four_actual_unchanged_derive_code_all_arrays_and_proof_exact_immutable_JSON_cache',cases=proof,derive_code_object_SHA256=hashlib.sha256(raw).hexdigest(),worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,new_Source=0,new_expert=0,new_model=0,new_GT=0))

def predict(ds):
 import torch
 torch.set_num_threads(1);execution();gt_guard();original.read=cached_read;start=time.perf_counter();inventory=read(D/'preserved_serial_runtime_v1/partial_prediction_SHA_inventory.json');files={};reused=0;created=0;assert not(D/ds/'sealed.json').exists()
 for i,row in enumerate(records(ds),1):
  resources(start);a,p=original.derive(ds,row['id']);path=D/ds/'predictions'/(row['id']+'.npz')
  if path.exists():assert inventory[str(path.relative_to(D))]==sha(path);reused+=1
  else:
   path.parent.mkdir(parents=True,exist_ok=True)
   with path.with_suffix('.runtime_v2.tmp').open('xb')as f:np.savez_compressed(f,**a)
   os.replace(path.with_suffix('.runtime_v2.tmp'),path);created+=1
  with np.load(path)as z:assert set(z.files)==set(a)and all(np.array_equal(v,z[k])for k,v in a.items())
  files[row['id']]=dict(path=str(path.relative_to(D)),sha256=sha(path),provenance=p,existing_serial_prediction_bytes_reused=str(path.relative_to(D))in inventory,actual_v2_worker=identity())
  if i%250==0:atomic(D/ds/'predict_progress_v2.json',dict(completed=i,total=COUNTS[ds],reused=reused,created=created,worker=identity(),GT_read=False))
 cache_check();atomic(D/ds/'sealed.json',dict(status='sealed_whole_domain_before_GT',queries=len(files),files=files,worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,GT_read=False,new_model=0,new_Source=0,new_expert=0,existing_serial_predictions_reused_exact=reused,new_remaining_CPU_predictions=created,execution_v2=read(D/'execution_v2_plan.json')['fingerprint']))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['preflight','predict','score','audit']);p.add_argument('--dataset',choices=COUNTS);a=p.parse_args()
 if a.command=='preflight':preflight()
 else:
  execution();original.read=cached_read
  if a.command=='predict':predict(a.dataset)
  else:getattr(original,a.command)(a.dataset);cache_check()
