"""Retain exact old predictions/metadata while retiring superseded states and feature views."""
from pathlib import Path
import concurrent.futures,hashlib,io,json,os,sys,tarfile,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));BASE=ROOT/'artifacts/storage_cleanup_20261017_decota_tuning_v1'
from scripts.retire_unused_storage_20261017_v2 import opened,available,digest
def write(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def identity(p):
 s=p.stat();return dict(path=str(p.relative_to(ROOT)),bytes=s.st_size,allocated=s.st_blocks*512,device=s.st_dev,inode=s.st_ino,mtime_ns=s.st_mtime_ns,nlink=s.st_nlink)
def same(a,b):
 import torch,numpy as np
 if isinstance(a,torch.Tensor):assert isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
 elif isinstance(a,np.ndarray):assert isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b,equal_nan=True)
 elif isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:same(a[k],b[k])
 elif isinstance(a,(tuple,list)):
  assert type(a) is type(b) and len(a)==len(b)
  for x,y in zip(a,b):same(x,y)
 else:assert a==b or isinstance(a,float) and isinstance(b,float) and a!=a and b!=b
def main():
 import torch
 assert not BASE.exists();baseline=ROOT/'artifacts/decota_paper_experiments_v1/baselines';npzs=sorted(baseline.rglob('*.npz'));cache=[p for r in ['tastvg_coordinate_tuning_v2','tastvg_optuna_v1'] for p in sorted((ROOT/'artifacts'/r).glob('*/H_cache/*.pt'))];assert len(npzs)==93156 and len(cache)==1082
 for root in ['tastvg_coordinate_tuning_v2','tastvg_optuna_v1']:assert json.loads((ROOT/'artifacts'/root/'FINAL_COMPLETION.json').read_text())['status']=='completed_verified_and_published'
 protected=[ROOT/'methods/CURRENT_METHOD.json',ROOT/'methods/PAPER_METHOD.json',ROOT/'artifacts/decota_full_case_iteration_v1/analysis/hcstvg1_test/rows.json',ROOT/'artifacts/decota_full_case_iteration_v1/analysis/hcstvg1_test/summary.json']
 for name in ['artifacts/stvg_r1_geometry32_v1/EXECUTION_RUNTIME002.json','artifacts/stvg_r1_dino_complement32_v1/EXECUTION_RUNTIME.json']:
  protected.extend(ROOT/p for p in json.loads((ROOT/name).read_text())['pins'])
 protected={str(p.relative_to(ROOT)):digest(p) for p in protected};targets=[identity(p) for p in npzs+cache];paths={r['path'] for r in targets};assert not paths.intersection(protected);busy,commands=opened();assert not any((r['device'],r['inode']) in busy for r in targets);before=available();BASE.mkdir();rows=[]
 # Retain every non-state NPY member byte-for-byte. No array is interpreted or rescored.
 def capsule(p):
  original=digest(p)
  with zipfile.ZipFile(p) as z:
   assert set(z.namelist())=={'metadata.npy','Before.npy','After.npy','state_before.npy','state_after.npy'};raw={n:z.read(n) for n in ['metadata.npy','Before.npy','After.npy']}
  stream=io.BytesIO()
  with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
   for n,b in raw.items():z.writestr(n,b)
  blob=stream.getvalue()
  with zipfile.ZipFile(io.BytesIO(blob)) as z:assert all(z.read(n)==b for n,b in raw.items())
  return blob,dict(original_sha256=original,retained_sha256=hashlib.sha256(blob).hexdigest(),member_SHA256={n:hashlib.sha256(b).hexdigest() for n,b in raw.items()},removed=['state_before.npy','state_after.npy'])
 with tarfile.open(BASE/'DECoTA_before_after_predictions.tar','w') as archive,concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  for p,(blob,audit) in zip(npzs,pool.map(capsule,npzs)):
   name=str(p.relative_to(ROOT));info=tarfile.TarInfo(name);info.size=len(blob);info.mode=0o600;archive.addfile(info,io.BytesIO(blob));rows.append(dict(identity(p),**audit,retained_archive='DECoTA_before_after_predictions.tar',retained_member=name))
 # H_cache is source replay input. Preserve prediction and all other fields, retire only views.
 with tarfile.open(BASE/'TUNING_source_predictions_without_views.tar','w') as archive:
  for p in cache:
   original=digest(p);d=torch.load(p,map_location='cpu',weights_only=False);assert isinstance(d,dict) and 'views' in d and 'prediction' in d;kept={k:v for k,v in d.items() if k!='views'};stream=io.BytesIO();torch.save(kept,stream);blob=stream.getvalue();same(kept,torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False));name=str(p.relative_to(ROOT));info=tarfile.TarInfo(name);info.size=len(blob);info.mode=0o600;archive.addfile(info,io.BytesIO(blob));rows.append(dict(identity(p),original_sha256=original,retained_sha256=hashlib.sha256(blob).hexdigest(),removed=['views'],all_other_fields_prediction_dtype_values_bitwise=True,retained_archive='TUNING_source_predictions_without_views.tar',retained_member=name))
 # Independently read all archived members before deleting any original container.
 for name in ['DECoTA_before_after_predictions.tar','TUNING_source_predictions_without_views.tar']:
  with tarfile.open(BASE/name,'r') as archive:
   expected={r['retained_member']:r for r in rows if r['retained_archive']==name};assert len(archive.getmembers())==len(expected)
   for entry in archive:
    assert hashlib.sha256(archive.extractfile(entry).read()).hexdigest()==expected[entry.name]['retained_sha256']
 busy,_=opened();assert not any((r['device'],r['inode']) in busy for r in rows);assert len(rows)==len(targets);write(BASE/'RETIREMENT_MANIFEST.json',dict(status='locked',records=rows,protected=protected,predictions_retained_not_recomputed=True,open_FD_mmap_checks=2,storage_only_no_GT_evaluation=True))
 for r in rows:
  p=ROOT/r['path'];s=p.stat();assert [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]==[r[k] for k in ['device','inode','bytes','mtime_ns']];p.unlink()
 for p,h in protected.items():assert digest(ROOT/p)==h,p
 for root in [baseline,ROOT/'artifacts/tastvg_coordinate_tuning_v2',ROOT/'artifacts/tastvg_optuna_v1']:write(root/'BINARY_STATE_RETIREMENT_20261017.json',dict(status='retired_states_predictions_retained',manifest=str((BASE/'RETIREMENT_MANIFEST.json').relative_to(ROOT)),old_container_paths_removed=True,retained_prediction_archive=str(BASE.relative_to(ROOT)),cannot_replay_original_optimization_without_rebuilding=True))
 after=available();archives=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=digest(p)) for p in BASE.glob('*.tar')];write(BASE/'COMPLETION.json',dict(status='complete',count=len(rows),decota_state_containers=len(npzs),tuning_feature_containers=len(cache),removed_logical_bytes=sum(r['bytes'] for r in rows),removed_allocated_bytes=sum(r['allocated'] for r in rows),retained_archives=archives,retained_archives_bytes=sum(r['bytes'] for r in archives),available_before=before,available_after=after,net_available_increase=after-before,hardlinks_accounted_in_actual_available_not_logical_sum=True,predictions_and_metadata_preserved_exact=True,current_method_and_active_scopes_hashes_pass=True,no_new_model_or_scoring=True));print(json.dumps(json.loads((BASE/'COMPLETION.json').read_text())))
if __name__=='__main__':main()
