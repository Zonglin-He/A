"""One full-list Source producer; exact pilot reuse, no expert model access."""
from common import *
from compatibility import current_exact
import gc,traceback,argparse
def subprocess_audit(ds,shard,bridge=False):
 name=f'{ds}_shard{shard:05d}';log=D/'logs'/(name+'_input_audit.log');log.parent.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
 with log.open('xb')as f:
  p=subprocess.Popen([str(PYTHON),'-B',str(S/'evaluate.py'),'shard_audit','--dataset',ds,'--shard',str(shard)],cwd=R,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},stdout=f,stderr=subprocess.STDOUT);who=identity(p.pid);code=p.wait()
 atomic(D/'exits'/(name+'_input_audit.json'),dict(worker=who,actual_subprocess_wait=True,exit_code=code,elapsed_wall_seconds=time.perf_counter()-start,log_sha256=sha(log)));assert code==0,('Preserve all paid state; no retry',name)
def run(ds):
 plan=check();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0';dest=D/ds;assert not(dest/'worker.json').exists();foreign();lock=lease();start=time.perf_counter();files={};pending=[];shard=0;first_new=[];model=None;parser=None;counts=dict(queries=0,reused_pilot=0,reused_raw_grid=0,new_Source_attempts=0,new_Source_completed=0,new_expert=0,new_optimizer=0)
 from stvg.runtime import setup,load_source
 from stvg.native import frozen_native_forward
 from stvg.representation import observe_native_grid
 from stvg.evidence import extract_visual_grids
 from stvg.tensors import detached
 from stvg.decode import decode
 from stvg.query import QuerySubjectParser
 setup();torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False;torch.set_num_threads(1);atomic(dest/'worker.json',dict(**identity(),plan_fingerprint=plan['fingerprint']))
 pilot=fixed(PILOT/ds/'sealed.json')['files'];raws=fixed(D/'raw_inventory.json')[ds]
 def audit_pending():
  nonlocal shard,pending
  if not pending:return
  atomic(dest/'shards'/f'{shard:05d}.json',dict(keys=pending.copy(),files={k:files[k]for k in pending},GT_read=False,plan_fingerprint=plan['fingerprint'],worker=identity()))
  subprocess_audit(ds,shard);ack=read(dest/'shards'/f'{shard:05d}_audit.json');assert ack['all_arrays_cold_exact']
  # Retain ALL minimal full-grid matching inputs for future reuse. Only this
  # worker's redundant native checkpoint is released after independent audit.
  for key in pending:
   e=files[key]
   if e['origin']['kind']=='new_Source':
    path=D/e['origin']['native_path'];assert sha(path)==e['origin']['native_sha256'];assert sha(D/e['compact_path'])==e['compact_sha256'];path.unlink();atomic(path.with_name('disposed.json'),dict(only_new_redundant_native_checkpoint_disposed=True,native_Source_sha256=e['origin']['native_sha256'],retained_full_grid_matching_input_sha256=e['compact_sha256'],prediction_sha256=e['sha256'],independent_shard_audit_sha256=sha(dest/'shards'/f'{shard:05d}_audit.json'),original_Source_nativeheads_currentpool_geometry_finalDP_exact=True,first_two_actual_bridge_verified=not first_new or(dest/'actual_first2_Source_bridge.json').exists()))
  pending=[];shard+=1
 try:
  for row in rows(ds):
   resources(start);key=row['id'];tick=time.perf_counter();foreign()
   if key in pilot:
    e=pilot[key]
    for f,h in [('path','sha256'),('compact_path','compact_sha256'),('receipt_path','receipt_sha256')]:assert sha(PILOT/e[f])==e[h]
    controls_a,controlsha=controls(ds,key)
    with np.load(PILOT/e['path'])as z:
     a={k:z[k].copy()for k in ['Matching','Current','PureMatched','frame_ids','positions8','Matching_interval','Current_interval','PureMatched_interval']};assert all(np.array_equal(a[k],v)for k,v in controls_a.items());assert np.array_equal(a['Matching_interval'],a['Current_interval'])
    # Keep original audited prediction/complete features read-only and refer to it.
    ref=dict(path=str(PILOT/e['path']),sha256=e['sha256'],compact_path=str(PILOT/e['compact_path']),compact_sha256=e['compact_sha256'],origin=dict(kind='original128_exact_reuse',pilot_receipt_sha256=e['receipt_sha256']),control_sha256=controlsha,all_array_schema='pilot',new_Source=0,new_expert=0,GT_read=False,worker=identity(),query_wall_seconds=time.perf_counter()-tick);counts['reused_pilot']+=1
   else:
    cap,binding=load_parent(ds,row);control,controlsha=controls(ds,key);assert np.array_equal(control['frame_ids'],cap['frame_ids'])and np.array_equal(control['positions8'],cap['positions8']);native_file=dest/'active'/key/'native_Source.pt';source_seconds=0.;compat={}
    if key in raws:
     old=raws[key];assert sha(old['path'])==old['sha256'];raw=torch.load(old['path'],map_location='cpu',weights_only=False,mmap=True);assert raw['parent_capture_sha256']==binding['capture_sha256'];f,valid,shape=raw['spatial_features'],raw['spatial_valid'],raw['spatial_grid_shape'];origin=dict(kind='original_retained_full_grid',path=old['path'],sha256=old['sha256']);counts['reused_raw_grid']+=1
    else:
     assert not native_file.exists()and not native_file.with_name('attempt.json').exists(),'No blind Source reacquisition'
     if model is None:
      load=time.perf_counter();atomic(dest/'model_load_attempt.json',dict(before_model=True,worker=identity()));model=load_source(fixed(R/'configs/default.json')['datasets'][ds]);model.cpu();parser=QuerySubjectParser(R/'checkpoints/stanza');atomic(dest/'model_load.json',dict(worker=identity(),elapsed_wall_seconds=time.perf_counter()-load,loads=1,source_checkpoint_sha256=plan['source_checkpoints'][ds]))
     frames,ids=decode(row['metadata'],ds);assert ids==cap['frame_ids'];parsed={'subject':parser(row['metadata']['caption'])['subject']};atomic(native_file.with_name('attempt.json'),dict(before_Source=True,worker=identity(),parent=binding,RGB_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),parses=parsed,GT_read=False));torch.set_num_threads(4);torch.manual_seed(20260920);np.random.seed(20260920);foreign();model.cuda();counts['new_Source_attempts']+=1;assert counts['new_Source_attempts']<=plan['maximum_new_Source'][ds];st=time.perf_counter()
     with torch.no_grad(),observe_native_grid(model)as grid:batch,recs,base=frozen_native_forward(model,frames,dict(input=row['metadata'],frame_ids=ids,parses=parsed))
     torch.cuda.synchronize();source_seconds=time.perf_counter()-st
     raw=dict(Source=base.zero['boxes'].detach().cpu().numpy(),native_logits=detached(base.zero['logits'],'cpu'),records=recs,spatial_inputs=detached(base.spatial_inputs,'cpu'),native_local_grid=grid,frame_ids=ids,parent_capture_sha256=binding['capture_sha256'],GT_read=False)
     save(native_file,raw);atomic(native_file.with_suffix('.json'),dict(sha256=sha(native_file),checkpoint_before_any_compatibility_or_pool=True,actual_Source_seconds=source_seconds,worker=identity(),GT_read=False));counts['new_Source_completed']+=1;model.cpu();del frames,batch,base;gc.collect();torch.cuda.empty_cache();torch.set_num_threads(1)
     f,valid,shape,compat=current_exact(cap,raw,control);origin=dict(kind='new_Source',native_path=str(native_file.relative_to(D)),native_sha256=sha(native_file),Source_seconds=source_seconds,original_native_and_current_exact=compat);first_new.append(key)
    torch.set_num_threads(1);v=compact(cap,f,valid,shape,row);st=time.perf_counter();boxes,diag=replay(v,ds);matching_seconds=time.perf_counter()-st;assert np.array_equal(boxes[v['positions8']],v['native_boxes'])
    a={**control,'Matching':boxes,'Matching_interval':control['Current_interval'].copy()};cp=dest/'compact'/(key+'.pt');save(cp,v);cold=torch.load(cp,map_location='cpu',weights_only=False,mmap=True);again,diagnostics=replay(cold,ds);assert np.array_equal(again,boxes)and diagnostics==diag
    out=dest/'predictions'/(key+'.npz');out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('xb')as fh:np.savez_compressed(fh,**a)
    with np.load(out)as z:assert set(z.files)==set(a)and all(np.array_equal(x,z[k])for k,x in a.items())
    ref=dict(path=str(out.relative_to(D)),sha256=sha(out),compact_path=str(cp.relative_to(D)),compact_sha256=sha(cp),origin=origin,parent=binding,control_sha256=controlsha,diagnostics=diag,matching_CPU_seconds=matching_seconds,Source_seconds=source_seconds,all_array_schema='full',new_Source=int(origin['kind']=='new_Source'),new_expert=0,GT_read=False,worker=identity(),query_wall_seconds=time.perf_counter()-tick)
   rp=dest/'receipts'/(key+'.json');atomic(rp,ref);files[key]=dict(ref,receipt_path=str(rp.relative_to(D)),receipt_sha256=sha(rp));pending.append(key);counts['queries']+=1
   # First two NEW Source outputs are independently re-extracted and replayed
   # by a real separate CPU process before another paid Source is requested.
   if len(first_new)==2 and not(dest/'actual_first2_Source_bridge.json').exists():
    atomic(dest/'first2_Source_input.json',dict(keys=first_new.copy(),files={k:files[k]for k in first_new},plan_fingerprint=plan['fingerprint']));log=dest/'first2_Source_bridge.log';bt=time.perf_counter()
    with log.open('xb')as fh:p=subprocess.Popen([str(PYTHON),'-B',str(S/'evaluate.py'),'bridge','--dataset',ds],cwd=R,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},stdout=fh,stderr=subprocess.STDOUT);who=identity(p.pid);code=p.wait()
    atomic(dest/'first2_Source_bridge_actual_exit.json',dict(worker=who,actual_subprocess_wait=True,exit_code=code,elapsed_wall_seconds=time.perf_counter()-bt,log_sha256=sha(log)));assert code==0,'All paid Source preserved on bridge failure'
   if len(pending)>=SHARD:audit_pending()
   atomic(dest/'progress.json',dict(completed=counts['queries'],total=COUNTS[ds],counts=counts,completed_independent_shards=shard,retained_fullgrid_bytes=sum(p.stat().st_size for p in(dest/'compact').glob('*.pt')),worker=identity(),GT_read=False));print(ds,counts,flush=True)
   if key not in pilot:del cap,raw,f,valid,v,cold,boxes,again
   gc.collect()
  audit_pending();assert counts['queries']==COUNTS[ds]and counts['reused_pilot']==128;atomic(dest/'sealed.json',dict(queries=COUNTS[ds],files=files,counts=counts,independent_shards=shard,GT_read=False,plan_fingerprint=plan['fingerprint'],worker=identity(),elapsed_wall_seconds=time.perf_counter()-start));print(ds,'whole full-list sealed, all complete grids retained',flush=True)
 except BaseException:atomic(dest/'failure.json',dict(status='failed_preserve_all_paid_attempt_native_features_predictions_no_retry',error=traceback.format_exc(),counts=counts,worker=identity()));raise
 finally:
  if model is not None:model.cpu()
  lock.close()
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--dataset',choices=COUNTS,required=True);run(a.parse_args().dataset)
