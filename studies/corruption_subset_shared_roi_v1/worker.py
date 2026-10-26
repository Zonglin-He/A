"""One corruption/domain, paid checkpoints before validation and expert use."""
from subset_common import *
import argparse,traceback
def predict(ds,name):
 import torch,numpy as np
 from stvg.runtime import setup,lease,load_source
 from stvg.decode import decode
 from stvg.readout import prediction
 from stvg.visual_prefix import native_appearance_grid
 from stvg.teacher_bridge import evidence_key
 from stvg.query import QuerySubjectParser
 p=validate();assert read(D/'preflight_receipt.json')['status']=='passed_real_pixels_cached_selected_math_no_model_GT'
 root=unit(ds,name);assert not(root/'worker.json').exists(),'No automatic paid retry';setup();lock=lease();who=identity();atomic(root/'worker.json',who)
 start=time.perf_counter();files={};model=None;teacher=None;parser=None;Klass=checkpoint_class();counts=dict(queries=0,new_Source=0,new_appearance_prefix=0,new_expert=0,legacy_source_views=0,clean_identity_views=0);bridged=0
 bridge_mode='legacy_prefix'if name.startswith('legacy_')else'new_Source';domain_bridge=D/f'{ds}_{bridge_mode}_root_bridge.json';root_needed=not domain_bridge.exists()
 try:
  for i,row in enumerate(rows(ds),1):
   resources(start);tick=time.perf_counter();key=row['id'];frames,ids=decode(row['metadata'],ds);view,meta=operators.apply(frames,ids,row['metadata'],ds,condition(name));rgbsha=hashlib.sha256(view.tobytes()).hexdigest();originalsha=hashlib.sha256(frames.tobytes()).hexdigest()
   out=root/'predictions'/(key+'.npz');rp=root/'receipts'/(key+'.json');cp=root/'compact'/(key+'.pt');scratch=root/'active'/key;rawfile=scratch/'processed_raw.pt';fullfile=scratch/'full_Source_checkpoint.pt';attempt=scratch/'attempt.json';assert not out.exists()and not cp.exists()and not attempt.exists()
   teacherproof=None;proof={};cost={};paid=False
   if np.array_equal(view,frames):
    a,cl=clean(ds,row);c=dict(kind='clean_identity',dataset=ds,query=key,clean_proof=cl,GT_read=False);counts['clean_identity_views']+=1;cost=dict(new_Source=0,new_expert=0,new_appearance_prefix=0,inherited_original_acquisition_nonzero=cl)
   else:
    if model is None:
     atomic(root/'model_load_attempt.json',dict(worker=who,before_model=True));lt=time.perf_counter();model=Klass(ds);model.source=load_source(model.config);model.source.cpu();atomic(root/'model_load.json',dict(worker=who,seconds=time.perf_counter()-lt,scope='Actual Source construction once/PID/start; teacher separately resident on first miss'))
     teacher=Teacher(root);model.teacher_service=teacher
    atomic(attempt,dict(worker=who,before_paid=True,RGB_sha256=rgbsha,condition=name,GT_read=False));paid=True
    if name.startswith('legacy_'):
     old,e=old_entry(ds,key,condition(name)['name']);assert originalsha==e['original_RGB_sha256']and rgbsha==e['transformed_RGB_sha256'];pt=time.perf_counter();grid=native_appearance_grid(model.source,view,ids,row['metadata']);raw=dict(frame_ids=ids,native_local_grid=grid,GT_read=False);save(rawfile,raw);atomic(scratch/'processed_raw.json',dict(sha256=sha(rawfile),before_teacher_context_validation_and_ROI=True,worker=who));prefixsec=time.perf_counter()-pt;counts['new_appearance_prefix']+=1;counts['legacy_source_views']+=1
     te=e['teacher'];assert sha(te['path'])==te['sha256']and sha(te['receipt'])==te['receipt_sha256'];tkey,context=evidence_key(view[old['positions8']],row['metadata']['caption'],fingerprint=teacher.fingerprint,positions=old['positions8'],frame_ids=ids,dataset=ds);tr=read(te['receipt']);assert tr['key']==tkey and tr['context']==context and tr['sha256']==te['sha256'];teacherproof=te
     ct=time.perf_counter();c,proof=from_legacy(ds,key,condition(name)['name'],raw);a=replay(c);cost=dict(new_Source=0,new_expert=0,new_appearance_prefix=1,appearance_prefix_and_checkpoint_seconds=prefixsec,compatibility_and_selected_finish_seconds=time.perf_counter()-ct,inherited_nonzero_Source_expert=e)
    else:
     def full_checkpoint(z):
      save(fullfile,z);atomic(scratch/'full_Source_checkpoint.json',dict(sha256=sha(fullfile),worker=who,before_native_validation_spatial_parsing_teacher_ROI=True,GT_read=False));counts['new_Source']+=1
     def processed_checkpoint(z):
      save(rawfile,z);atomic(scratch/'processed_raw.json',dict(sha256=sha(rawfile),worker=who,before_teacher_and_ROI=True,GT_read=False))
     model.full_checkpoint=full_checkpoint;model.processed_checkpoint=processed_checkpoint
     if parser is None:parser=QuerySubjectParser(R/'checkpoints/stanza')
     parses=dict(subject=parser(row['metadata']['caption'])['subject']);torch.manual_seed(20260920);np.random.seed(20260920)
     cap=model.capture(view,ids,row['metadata'],parses=parses,collect_mask_diagnostics=True);native=prediction(cap['native_logits'],torch.as_tensor(cap['Source']),cap['records'],ids)['physical_interval'];ft=time.perf_counter();c=from_capture(cap,native);a=replay(c);teacherproof=teacher.last;cost=dict(new_Source=1,new_expert=cap['cost']['ReferDINO_query_forwards'],new_appearance_prefix=0,capture=cap['cost'],selected_finish_seconds=time.perf_counter()-ft);counts['new_expert']+=cost['new_expert'];proof=dict(complete_Source_checkpoint_before_native_validation=True,actual_E_ROI_pool=True,first_aux_DP_in_selected_path=False);del cap
   c.update(dataset=ds,query=key);save(cp,c);again=replay(torch.load(cp,map_location='cpu',weights_only=False,mmap=True));assert all(np.array_equal(a[k],again[k])for k in a)
   out.parent.mkdir(parents=True,exist_ok=True)
   with out.with_suffix('.tmp').open('wb')as f:np.savez_compressed(f,**a)
   os.replace(out.with_suffix('.tmp'),out)
   with np.load(out)as z:assert all(np.array_equal(a[k],z[k])for k in a)
   e=dict(path=str(out.relative_to(D)),sha256=sha(out),compact_path=str(cp.relative_to(D)),compact_sha256=sha(cp),worker=who,kind=c['kind'],query=key,teacher=teacherproof,proof=proof,original_RGB_sha256=originalsha,transformed_RGB_sha256=rgbsha,transform=meta,cost=cost,query_seconds=time.perf_counter()-tick,GT_read=False,all_arrays_coldreplay_exact=True,processed_raw_sha256=sha(rawfile)if paid else None,full_Source_checkpoint_sha256=sha(fullfile)if fullfile.exists()else None)
   atomic(rp,e);files[key]={**e,'receipt_path':str(rp.relative_to(D)),'receipt_sha256':sha(rp)}
   if paid and root_needed and bridged<2:
    bridged+=1;atomic(root/'bridge_waiting.json',dict(files={k:v for k,v in files.items()if v['processed_raw_sha256']},paid_count=bridged,worker=who,GT_read=False))
    if bridged==2:
     print('WAIT_ROOT_ACTUAL_TWO_PAID_BRIDGE',ds,name,flush=True)
     while not(root/'root_actual_bridge.json').exists():resources(start);time.sleep(3)
     ack=read(root/'root_actual_bridge.json');assert ack['status']=='passed_actual_two_paid_raw_repool_cold_exact'and ack['bridge_waiting_sha256']==sha(root/'bridge_waiting.json')
     for k in ack['queries']:
      entry=files[k];sc=root/'active'/k
      for f in [sc/'processed_raw.pt',sc/'full_Source_checkpoint.pt']:
       if f.exists():h=sha(f);f.unlink();atomic(sc/(f.name+'.disposed.json'),dict(original_sha256=h,committed_receipt_sha256=entry['receipt_sha256'],root_bridge_sha256=sha(root/'root_actual_bridge.json')))
   elif paid:
    for f in [rawfile,fullfile]:
     if f.exists():h=sha(f);f.unlink();atomic(scratch/(f.name+'.disposed.json'),dict(original_sha256=h,committed_receipt_sha256=sha(rp),all_arrays_coldreplay_exact=True))
   counts['queries']=i;atomic(root/'progress.json',dict(counts=counts,total=128,worker=who,GT_read=False));print(ds,name,i,counts,flush=True)
   del frames,view,a,c,again;model.source.cpu()if model else None;gc.collect();torch.cuda.empty_cache()
  assert len(files)==128
  atomic(root/'sealed.json',dict(files=files,queries=128,counts=counts,seconds=time.perf_counter()-start,worker=who,GT_read=False,fingerprint=p['fingerprint']));atomic(root/'GPU_state.json',dict(status='complete_sealed'))
 except BaseException:atomic(root/'GPU_state.json',dict(status='failed_preserved',error=traceback.format_exc(),worker=who,counts=counts));raise
 finally:
  if teacher is not None:teacher.close()
  lock.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=COUNTS,required=True);p.add_argument('--condition',required=True);a=p.parse_args();predict(a.dataset,a.condition)
