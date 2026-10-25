"""One CG Source-or-prefix reconstruction per missing raw query; zero new expert."""
from cg_common import *
import argparse,traceback,gc,copy
def predict(ds):
 import torch,numpy as np
 from stvg.runtime import setup,lease
 from stvg.decode import decode
 from stvg.positions import observation_positions
 from stvg.teacher_bridge import evidence_key
 from cg_core import build,replay
 old=legacy();from cgstvg_runtime_v9 import load_model,capture
 from cgstvg_interfaces_v9 import frame_grid
 p=validate();assert read(D/'preflight_receipt.json')['status']=='passed';assert not(D/f'{ds}_worker.json').exists();setup();lock=lease();who=identity();atomic(D/f'{ds}_worker.json',who);started=time.perf_counter();model=None;files={};reuse=fixed(CGOLD/'reuse_index.json')[ds];counts=dict(queries=0,reused_raw=0,new_Source=0,new_Source_forwards=0,new_expert=0);load=None
 try:
  for i,row in enumerate(old.rows(ds),1):
   resources(started)
   if i%32==1:assert sum(f.stat().st_size for f in D.rglob('*')if f.is_file())<p['limits']['private_bytes']
   key=row['id'];assert not committed(ds,key),'No duplicate or blind recovery';scratch=D/'active'/ds/key;raw=scratch/'Source.pt';rawmeta=scratch/'Source.json';attempt=scratch/'attempt.json';assert not attempt.exists()and not raw.exists();tick=time.perf_counter()
   if key in reuse:
    cap,native,origin=old.load_legacy(reuse[key],row,ds);counts['reused_raw']+=1;source_s=0.
   else:
    if model is None:
     start=time.perf_counter();atomic(D/f'{ds}_model_load_attempt.json',dict(worker=who,before_model=True))
     try:model,load=load_model(ds)
     except BaseException:atomic(D/f'{ds}_model_load_failure.json',dict(worker=who,actual_failed_load_seconds=time.perf_counter()-start,error=traceback.format_exc()));raise
     load={**load,'actual_load_seconds':time.perf_counter()-start,'worker':who};atomic(D/f'{ds}_model_load.json',load)
    metadata={**row['metadata'],'frame_ids':frame_grid(row['metadata'],ds)};frames,ids=decode(metadata,ds);assert ids==metadata['frame_ids'];atomic(attempt,dict(worker=who,before_Source=True,input_binding=old.row_binding(row),GT_read=False));cap=capture(model,frames,ids,metadata['caption'],noop=False);assert cap['source_forwards']==1;cap['positions8']=observation_positions(ids,[ids.index(cap['native_map'][0]),ids.index(cap['native_map'][1]-1)]);save(raw,cap);source_s=cap['source_seconds'];atomic(rawmeta,dict(path=str(raw.relative_to(D)),sha256=sha(raw),worker=who,Source_seconds=source_s,Source_forwards=1,input_binding=old.row_binding(row),before_teacher_or_ROI_pool=True,GT_read=False));counts['new_Source']+=1;counts['new_Source_forwards']+=1
    oe=fixed(CGOLD/f'{ds}_sealed.json')['files'][key];ac=oe['acquisition'];tp=Path(ac['teacher_path']);assert sha(tp)==ac['teacher_sha256'];tm=read(tp.with_suffix('.json'));key2,context=evidence_key(frames[cap['positions8']],metadata['caption'],fingerprint=p['teacher_runtime_fingerprint'],positions=cap['positions8'],frame_ids=ids,dataset=ds);assert tm['key']==key2==ac['teacher_key']and tm['context']==context and tm['sha256']==ac['teacher_sha256'],'Original exact teacher context required; no expert fallback';native=torch.load(tp,map_location='cpu',weights_only=False,mmap=True);native={**native,'positions':cap['positions8']};origin=dict(capture_path=str(raw),capture_sha256=sha(raw),teacher_path=str(tp),teacher_sha256=ac['teacher_sha256'],teacher_receipt_sha256=sha(tp.with_suffix('.json')),teacher_key=key2,actual_context_exact=True,Source_checkpoint_before_teacher=True,old_acquisition=ac,new_expert=0);del frames
   arrays,compact,proof,cost=build(cap,native,ds,key);compact.update(proof=proof,input_binding=old.row_binding(row),acquisition=origin);dest=D/ds;out=dest/'predictions'/(key+'.npz');cp=dest/'compact'/(key+'.pt');rp=dest/'receipts'/(key+'.json');assert not out.exists()and not cp.exists()and not rp.exists();save(cp,compact);cold=torch.load(cp,map_location='cpu',weights_only=False,mmap=True);again=replay(cold);assert all(np.array_equal(arrays[k],again[k])for k in arrays);out.parent.mkdir(parents=True,exist_ok=True)
   with out.with_suffix('.tmp').open('wb')as f:np.savez_compressed(f,**arrays)
   os.replace(out.with_suffix('.tmp'),out)
   with np.load(out)as z:assert all(np.array_equal(arrays[k],z[k])for k in arrays)
   rec=dict(path=str(out.relative_to(D)),sha256=sha(out),compact_path=str(cp.relative_to(D)),compact_sha256=sha(cp),worker=who,input_binding=old.row_binding(row),acquisition=origin,proof=proof,legacy_raw_reused=key in reuse,Source_seconds=source_s,pipeline_seconds=time.perf_counter()-tick,**cost,GT_read=False,new_expert=0,all_arrays_cold_replay_exact=True,fingerprint=p['fingerprint']);atomic(rp,rec);files[key]={**rec,'receipt_path':str(rp.relative_to(D)),'receipt_sha256':sha(rp)}
   if raw.exists():
    assert raw.resolve().is_relative_to((D/'active').resolve())and sha(raw)==read(rawmeta)['sha256'];raw.unlink();atomic(scratch/'disposed.json',dict(raw_sha256=origin['capture_sha256'],prediction_sha256=rec['sha256'],compact_sha256=rec['compact_sha256'],receipt_sha256=sha(rp),all_arrays_cold_replay_exact=True))
   counts['queries']+=1;atomic(D/f'{ds}_progress.json',dict(completed=i,total=COUNTS[ds],counts=counts,worker=who,GT_read=False,seconds=time.perf_counter()-started));print(ds,i,counts,flush=True);del cap,native,arrays,compact,cold,again;gc.collect();torch.cuda.empty_cache()
  assert counts['queries']==COUNTS[ds]and counts['new_Source']==COUNTS[ds]-len(reuse)and counts['new_Source_forwards']==counts['new_Source']and counts['new_expert']==0
  atomic(D/f'{ds}_sealed.json',dict(queries=len(files),files=files,counts=counts,worker=who,seconds=time.perf_counter()-started,model_load=load,GT_read=False,fingerprint=p['fingerprint']));atomic(D/f'{ds}_GPU_state.json',dict(status='complete_sealed'))
 except BaseException:atomic(D/f'{ds}_GPU_state.json',dict(status='failed_preserved',worker=who,error=traceback.format_exc(),counts=counts));raise
 finally:lock.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=COUNTS,required=True);predict(p.parse_args().dataset)
