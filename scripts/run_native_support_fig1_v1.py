"""One backbone and finite phase, sequential GPU; no oracle or expert imports."""
import sys,time,traceback,hashlib,gc,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.native_support_fig1_common_v1 import *

def run(backbone,phase):
 assert backbone in MODELS and phase in ['smoke','capture']
 p=verify();roster=read(OUT/'ROSTER.json');subjects=read(OUT/'SUBJECTS.json');assert sha(OUT/'SUBJECTS.json')==p['subjects_sha256']
 target=OUT/backbone;start=time.time();done=0;lease=None
 try:
  import numpy as np,torch
  from scripts.run_final_simplification_v1 import lease as gpu_lease
  from vg_tta.exact_frame_decode_audit_v2 import decode
  from scripts.run_tastvg_full_b1_experts_v1 import observation
  from vg_tta.native_support_decode_fig1_v1 import TAAdapter,TubeAdapter,PTDAdapter
  sys.addaudithook(guard);lease=gpu_lease();torch.set_num_threads(4);torch.manual_seed(p['seed']);np.random.seed(p['seed']);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
  assert sha(p['checkpoint_paths'][backbone])==p['checkpoint_sha256'][backbone]
  torch.cuda.reset_peak_memory_stats();adapter=TAAdapter() if backbone=='tastvg' else TubeAdapter(p['checkpoint_paths'][backbone]) if backbone=='tubedetr' else PTDAdapter()
  assert not any(x.requires_grad for x in adapter.model.parameters());versions=[x._version for x in adapter.model.parameters()]
  rows=roster['rows'][:2] if phase=='smoke' else roster['rows'];conditions=['clean'] if phase=='smoke' else roster['conditions'];total=len(rows)*len(conditions)
  for row in rows:
   frames=None
   for cond in conditions:
    verify();dest=target/'predictions'/cond/f"{row['ordinal']:05}.json";receipt=dest.with_suffix('.receipt.json')
    if receipt.exists():assert sha(dest)==read(receipt)['sha256'];done+=1;continue
    assert not dest.exists(),'unreceipted output retained for root review'
    if frames is None:frames,ids=decode(row['input']);assert ids==row['frame_ids']
    pixels,pixel,spec=observation(row,cond,frames);tick=time.time();z=adapter.predict(pixels,row,subjects[str(row['ordinal'])],smoke=phase=='smoke')
    assert len(z['boxes'])==len(z['intervals'])==len(z['spatial_valid'])==len(z['temporal_valid'])==6
    assert z['native_parity'] and all(x is None or len(x)==2 for x in z['intervals'])
    z.update(backbone=backbone,ordinal=row['ordinal'],condition=cond,frame_ids=row['frame_ids'],pixel_sha256=pixel,corruption=spec,seconds=time.time()-tick,GT_read=False,parameter_updates=0,runtime_lock_sha256=sha(OUT/'RUNTIME_LOCK.json'),unique_temporal=len({json.dumps(x) for x in z['intervals']}),unique_spatial=len({json.dumps(x) for x in z['boxes']}))
    # Common raw observation binding is read-only for subsequent backbones.
    cf=OUT/'common_pixels'/cond/f"{row['ordinal']:05}.json";cr=dict(pixel_sha256=pixel,frame_ids=row['frame_ids'],corruption=spec)
    if cf.exists():assert read(cf)==cr
    else:write(cf,cr)
    write(dest,z);write(receipt,dict(sha256=sha(dest),runtime_lock_sha256=z['runtime_lock_sha256'],GT_read=False,time=time.time()))
    done+=1;status(target/'STATUS.json',dict(status='running',phase=phase,done=done,total=total,condition=cond,ordinal=row['ordinal'],seconds=time.time()-start,peak_allocated=torch.cuda.max_memory_allocated(),time=time.time()));print('FIG1',backbone,phase,done,total,'sec',round(time.time()-start,2),flush=True)
    del pixels,z;gc.collect();torch.cuda.empty_cache()
  assert versions==[x._version for x in adapter.model.parameters()]
  fs={str(f.relative_to(target)):sha(f) for f in (target/'predictions').rglob('*.json') if not f.name.endswith('.receipt.json')}
  assert len(fs)==total
  write(target/('SMOKE.json' if phase=='smoke' else 'PREDICTION_BARRIER.json'),dict(status='completed',cells=done,files=fs,GT_read=False,parameter_versions_unchanged=True,native_parity=True,peak_allocated=torch.cuda.max_memory_allocated(),seconds=time.time()-start,time=time.time()))
  status(target/'STATUS.json',dict(status='completed',phase=phase,done=done,total=total,seconds=time.time()-start,time=time.time()))
 except BaseException as e:
  status(target/'STATUS.json',dict(status='failed',phase=phase,done=done,error=repr(e),traceback=traceback.format_exc(),time=time.time()));raise
 finally:
  if lease:lease.close()
if __name__=='__main__':run(*sys.argv[1:])
