"""Original224 native appearance prefix once/view, zero new full Source or teacher."""
from panel_common import *
import argparse,traceback,gc
def predict(ds):
 import torch,numpy as np
 from stvg.runtime import setup,lease,load_source
 from stvg.decode import decode
 from stvg.visual_prefix import native_appearance_grid
 from stvg.teacher_bridge import evidence_key
 from studies.paper_review_suite_v1.shifts import transform
 from panel_core import build,replay
 p=validate();assert read(D/'preflight_receipt.json')['status']=='passed_CPU_contract_and_cached_math_only';assert not(D/f'{ds}_worker.json').exists();setup();lock=lease();who=identity();atomic(D/f'{ds}_worker.json',who);started=time.perf_counter();files={};counts=dict(views=0,appearance_prefix_calls=0,new_full_Source=0,new_expert=0);load=None
 try:
  cfg=read(R/'configs/default.json')['datasets'][ds];tick=time.perf_counter();atomic(D/f'{ds}_model_load_attempt.json',dict(worker=who,before_model=True))
  try:model=load_source(cfg);model.cpu()
  except BaseException:atomic(D/f'{ds}_model_load_failure.json',dict(worker=who,failed_load_seconds=time.perf_counter()-tick,error=traceback.format_exc()));raise
  load=dict(worker=who,actual_model_load_seconds=time.perf_counter()-tick,loads=1);atomic(D/f'{ds}_model_load.json',load)
  for row in rows(ds):
   frames,ids=decode(row['metadata'],ds);original_sha=hashlib.sha256(frames.tobytes()).hexdigest()
   for view in VIEWS:
    resources(started);assert sum(f.stat().st_size for f in D.rglob('*')if f.is_file())<p['limits']['private_bytes'];key=row['id']+'.'+view;c,e=checked_parent(ds,row['id'],view);assert ids==c['frame_ids']and original_sha==e['original_RGB_sha256'];rgb=transform(frames,view);rgb_sha=hashlib.sha256(rgb.tobytes()).hexdigest();assert rgb_sha==e['transformed_RGB_sha256'];scratch=D/'active'/ds/key;attempt=scratch/'attempt.json';rawfile=scratch/'prefix.pt';meta=scratch/'prefix.json';assert not attempt.exists()and not rawfile.exists();atomic(attempt,dict(before_prefix=True,worker=who,original_RGB_sha256=original_sha,view_RGB_sha256=rgb_sha,GT_read=False));tick=time.perf_counter();grid=native_appearance_grid(model,rgb,ids,row['metadata']);prefix_seconds=time.perf_counter()-tick;raw=dict(frame_ids=ids,native_local_grid=grid,GT_read=False);save(rawfile,raw);atomic(meta,dict(sha256=sha(rawfile),worker=who,prefix_seconds=prefix_seconds,before_teacher_context_validation_and_ROI_pool=True,GT_read=False));counts['appearance_prefix_calls']+=1
    te=e['teacher'];assert sha(te['path'])==te['sha256']and sha(te['receipt'])==te['receipt_sha256'];tm=read(te['receipt']);key2,context=evidence_key(rgb[c['positions8']],row['metadata']['caption'],fingerprint=p['teacher_runtime_fingerprint'],positions=c['positions8'],frame_ids=ids,dataset=ds);assert tm['key']==key2 and tm['context']==context and tm['sha256']==te['sha256'],'Exact same corrupted RGB teacher context mandatory; no new expert fallback'
    a,compact,proof,cost=build(ds,row['id'],view,raw);compact.update(proof=proof,teacher=te,original_RGB_sha256=original_sha,view_RGB_sha256=rgb_sha);out=D/ds/'predictions'/(key+'.npz');cp=D/ds/'compact'/(key+'.pt');rp=D/ds/'receipts'/(key+'.json');assert not out.exists()and not cp.exists()and not rp.exists();save(cp,compact);cold=torch.load(cp,map_location='cpu',weights_only=False,mmap=True);again=replay(cold);assert all(np.array_equal(a[k],again[k])for k in a);out.parent.mkdir(parents=True,exist_ok=True)
    with out.with_suffix('.tmp').open('wb')as f:np.savez_compressed(f,**a)
    os.replace(out.with_suffix('.tmp'),out)
    with np.load(out)as z:assert all(np.array_equal(a[k],z[k])for k in a)
    rec=dict(path=str(out.relative_to(D)),sha256=sha(out),compact_path=str(cp.relative_to(D)),compact_sha256=sha(cp),raw_prefix_sha256=sha(rawfile),worker=who,view=view,query=row['id'],teacher=te,proof=proof,original_RGB_sha256=original_sha,view_RGB_sha256=rgb_sha,actual_teacher_context_exact=True,prefix_seconds=prefix_seconds,pipeline_seconds=time.perf_counter()-tick,**cost,new_full_Source=0,new_expert=0,GT_read=False,all_arrays_cold_replay_exact=True,fingerprint=p['fingerprint']);atomic(rp,rec);files[key]={**rec,'receipt_path':str(rp.relative_to(D)),'receipt_sha256':sha(rp)};assert sha(rawfile)==read(meta)['sha256'];rawfile.unlink();atomic(scratch/'disposed.json',dict(raw_prefix_sha256=rec['raw_prefix_sha256'],compact_sha256=rec['compact_sha256'],receipt_sha256=sha(rp),all_arrays_cold_replay_exact=True));counts['views']+=1;atomic(D/f'{ds}_progress.json',dict(completed=counts['views'],total=384,counts=counts,worker=who,GT_read=False));print(ds,key,counts,flush=True);del rgb,grid,raw,a,compact,cold,again;model.cpu();gc.collect();torch.cuda.empty_cache()
   del frames
  assert counts==dict(views=384,appearance_prefix_calls=384,new_full_Source=0,new_expert=0);atomic(D/f'{ds}_sealed.json',dict(queries=128,views=384,files=files,counts=counts,worker=who,seconds=time.perf_counter()-started,model_load=load,GT_read=False,fingerprint=p['fingerprint']));atomic(D/f'{ds}_GPU_state.json',dict(status='complete_sealed'))
 except BaseException:atomic(D/f'{ds}_GPU_state.json',dict(status='failed_preserved',worker=who,error=traceback.format_exc(),counts=counts));raise
 finally:lock.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=COUNTS,required=True);predict(p.parse_args().dataset)
