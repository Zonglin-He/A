"""New bounded Source acquisition; never resumes any historical dispatcher."""
import os
_visibility=os.environ.get('CUDA_VISIBLE_DEVICES')
from common_v1 import *
if _visibility is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
else:os.environ['CUDA_VISIBLE_DEVICES']=_visibility
import subprocess,fcntl,gc,traceback
import numpy as np,torch
from full_core_v1 import F,Q,build,replay,ARMS,PAIRS
def save(path,x):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(x,tmp);os.replace(tmp,path)
def validate_full():
 p=read(F/'plan_v1.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for group in ['code_sha256','input_sha256']:
  for f,h in p[group].items():assert sha(R/f)==h,f
 return p
def prepare():
 assert not(F/'plan_v1.json').exists();inputs=[D/'plan_v1.json',selected.PREV/'raw_reuse_index.json',R/'configs/default.json']
 for ds in COUNTS:inputs += [selected.RUN/ds/'sealed.json',selected.PREV/'spatial'/f'{ds}_sealed.json',selected.D/f'components_{ds}_sealed.json',selected.D/f'components_{ds}_reporting_receipt.json']
 files=[Path(__file__),S/'full_core_v1.py',S/'full_controller_v1.py',S/'root_gpu_bridge_v1.py',S/'common_v1.py',R/'stvg/native.py',R/'stvg/representation.py',R/'stvg/evidence.py',R/'stvg/spatial.py',R/'stvg/initialization.py',R/'stvg/predictor.py',R/'stvg/prototype.py']
 p=dict(schema='full_regional_evidence_and_uniform_gamma_v1',authorization='2026-11-03 explicit user: missing-cache experiments may use GPU after existing task finishes; all reviewer experiments. This is a new bounded study, not historical full-gamma/CG/corruption/generic-TTA resumption.',counts=COUNTS,regions=['Source original box','shared-corrected box','all-valid global','outside expanded corrected box background','SHA temporal-permutation of whole corrected boxes'],region_output='same full E trajectory, each own q0->prototype->posterior->legal jointMAP; comparisons isolate intervals; source-only deployable baseline is CPU temporal task',uniform_gamma=[.5,1.],gamma='Both fixed values applied in BOTH directions, all other original per-domain config frozen; reconstruct E/raw ROI pool/geometry/posterior/own-interval DP. No best setting/default reselection. Historical direction-specific gamma used exposed target development labels, not source-only calibration.',params=dict(kappa=0,beta=1,context_scale=1.5,observations8=True,first_aux_DP=False),cached_raw={'hc2':192,'vidstg':192},maximum_new_Source=11821,maximum_new_expert=0,maximum_new_optimizer=0,acquisition='One original TA224 native two-offset Source per missing query shared by all regional and uniform-gamma arms. Original FP16 visual prefix/FP32 readout and cuDNN TF32 True/matmul TF32 False. Save complete native outputs, raw pretext grid, postfusion inputs and attempt SHA BEFORE any compatibility validation/pooling. Verify original Source/native logits/records/positions/frame IDs/q0 and original source/E pools. No extra teacher; original eight masks/native boxes retained in bound parent geometry.',first_paid_bridge='First two actually NEW queries/domain retain full paid raw and compact. Stop owned GPU worker in WAIT until root actual CPU raw grid re-pool, current/default/readout and double cold-array exact audit writes SHA-bound ack. No handwritten ack or blind repeat.',release='Only own new paid raw after committed prediction/compact/receipt readback+cold exact; first2 only after root ack. Preserve attempts, all failures/paid raw; all old roots immutable.',resource='One GPU worker, max2 heavy jobs total; wait complete CPU prediction-control suite; do not signal unrelated tasks. Before load/query require no foreign compute PID except exact ToDesk service and globalGPU flock. RAM8GiB/disk80GiB/private160GiB/domain72h, SourceTorch4, CPU14-15nice19idleIO except actual GPUproducer original cores allowed.',scoring='Each complete domain seal before separate official CPU scoring and independent all-array/means/inputSHA audit. Root all signed/negative/severe/rescue/shortlong/parent10000CI and fixed-gamma contrasts; not deployment latency.',code_sha256={str(f.relative_to(R)):sha(f)for f in files},input_sha256={str(f):sha(f)for f in inputs});p['fingerprint']=digest(p);atomic(F/'plan_v1.json',p);atomic(Q/'protocol_v1.json',{k:v for k,v in p.items()if k!='input_sha256'});print(p['fingerprint'])
def foreign_gpu():
 v=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True,check=True);out=[]
 for line in v.stdout.splitlines():
  if not line.strip():continue
  pid,name=line.split(',',1);pid=int(pid.strip());name=name.strip()
  if pid==os.getpid()or name in ['/opt/todesk/bin/ToDesk_Session','/opt/todesk/bin/ToDesk_Service']:continue
  out.append(dict(pid=pid,name=name))
 return out
def preflight():
 p=validate_full();torch.set_num_threads(1);infer_guard();cases=[]
 for ds in COUNTS:
  by={r['id']:r for r in selected.records(ds)}
  for key,e in list(read(selected.PREV/'raw_reuse_index.json')[ds].items())[:2]:
   assert sha(e['path'])==e['sha256'];raw=torch.load(e['path'],map_location='cpu',weights_only=False,mmap=True);cap,par=selected.suite().load_parent(ds,by[key]);a,c,proof,diag=build(ds,by[key],cap,raw);b=replay(c);assert set(a)==set(b)and all(np.array_equal(x,b[k])for k,x in a.items());cases.append(dict(dataset=ds,raw_sha256=e['sha256'],default_and_all_cold_arrays_exact=True,actual_all_regions_and_two_gamma_repool=True))
 atomic(F/'preflight_v1.json',dict(status='passed_CPU_cache_only_not_new_GPU',cases=cases,new_Source=0,new_expert=0,new_model=0));print('full cached preflight passed')
def predict(ds):
 p=validate_full();assert read(F/'preflight_v1.json')['status']=='passed_CPU_cache_only_not_new_GPU';assert not(F/ds/'worker.json').exists()
 from stvg.runtime import setup,lease,load_source
 from stvg.native import frozen_native_forward
 from stvg.representation import observe_native_grid,extract
 from stvg.tensors import detached
 from stvg.evidence import extract_visual_grids
 from stvg.decode import decode
 from stvg.query import QuerySubjectParser
 from stvg.temporal import probabilities,mass_table
 from stvg.positions import observation_positions
 from stvg.readout import prediction
 setup();torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False;start=time.perf_counter();lock=None
 while lock is None:
  resources(start);other=foreign_gpu()
  if not other:
   try:lock=lease()
   except BlockingIOError:pass
  if lock is None:atomic(F/ds/'waiting_resources.json',dict(status='waiting_existing_GPU_task_exit',foreign_compute=other));time.sleep(15)
 atomic(F/ds/'worker.json',identity());model=None;parser=None;files={};counts=dict(queries=0,new_Source=0,retained_raw=0,new_expert=0);paid=[];reuse=read(selected.PREV/'raw_reuse_index.json')[ds];dest=F/ds
 try:
  for row in selected.records(ds):
   resources(start);assert sum(f.stat().st_size for f in F.rglob('*')if f.is_file())<160*2**30;key=row['id'];cap,par=selected.suite().load_parent(ds,row);attempt=dest/'active'/key/'attempt.json';rawfile=attempt.with_name('Source.pt');meta=attempt.with_name('Source.json');out=dest/'predictions'/(key+'.npz');cp=dest/'compact'/(key+'.pt');rp=dest/'receipts'/(key+'.json');assert not out.exists()and not cp.exists()and not rp.exists();tick=time.perf_counter();source_seconds=0.
   if key in reuse:
    e=reuse[key];assert sha(e['path'])==e['sha256'];raw=torch.load(e['path'],map_location='cpu',weights_only=False,mmap=True);origin=dict(kind='retained_raw',path=e['path'],sha256=e['sha256']);counts['retained_raw']+=1
   else:
    assert not attempt.exists()and not rawfile.exists(),'Paid attempt preserved, no blind reacquisition'
    while foreign_gpu():model.cpu()if model is not None else None;torch.cuda.empty_cache();resources(start);time.sleep(15)
    if model is None:
     st=time.perf_counter();atomic(dest/'model_load_attempt.json',dict(worker=identity(),before_model=True))
     try:model=load_source(read(R/'configs/default.json')['datasets'][ds]);model.cpu()
     except BaseException:atomic(dest/'model_load_failure.json',dict(elapsed_wall_seconds=time.perf_counter()-st,error=traceback.format_exc()));raise
     atomic(dest/'model_load.json',dict(worker=identity(),elapsed_wall_seconds=time.perf_counter()-st,loads=1));parser=QuerySubjectParser(R/'checkpoints/stanza')
    frames,ids=decode(row['metadata'],ds);assert ids==cap['frame_ids'];parsed={'subject':parser(row['metadata']['caption'])['subject']};atomic(attempt,dict(worker=identity(),before_Source=True,frames_RGB_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),parent=par,GT_read=False));torch.manual_seed(20260920);np.random.seed(20260920);model.cuda();st=time.perf_counter()
    with torch.no_grad(),observe_native_grid(model)as grid:batch,recs,base=frozen_native_forward(model,frames,dict(input=row['metadata'],frame_ids=ids,parses=parsed))
    torch.cuda.synchronize();source_seconds=time.perf_counter()-st
    # Full paid output survives even if the first numerical assertion fails.
    paidstate=dict(frame_ids=ids,Source=base.zero['boxes'].detach().cpu().numpy(),native_logits=detached(base.zero['logits'],'cpu'),records=recs,spatial_inputs=detached(base.spatial_inputs,'cpu'),native_local_grid=grid,GT_read=False,parent_capture_sha256=par['capture_sha256'])
    save(rawfile.with_name('native_Source.pt'),paidstate);save(rawfile,paidstate);atomic(meta,dict(sha256=sha(rawfile),native_checkpoint_sha256=sha(rawfile.with_name('native_Source.pt')),worker=identity(),Source_seconds=source_seconds,checkpoint_before_compatibility_and_pool=True,GT_read=False));counts['new_Source']+=1;del batch,base;model.cpu();gc.collect();torch.cuda.empty_cache()
    assert np.array_equal(paidstate['Source'],cap['Source'])and all(torch.equal(x,y)for x,y in zip(paidstate['native_logits'],cap['native_logits']))and paidstate['records']==cap['records']
    ids2,q0=mass_table(probabilities(paidstate['records'],paidstate['native_logits']));assert ids2==ids and np.array_equal(q0,cap['native_prior']);pr=prediction(paidstate['native_logits'],torch.as_tensor(paidstate['Source']),paidstate['records'],ids);assert observation_positions(ids,pr['indices'])==cap['positions8']
    spatial,valid,_,shape,_=extract_visual_grids(paidstate['spatial_inputs'],ids);raw={**paidstate,'spatial_features':spatial,'spatial_valid':valid,'spatial_grid_shape':shape};save(rawfile,raw);atomic(meta,dict(sha256=sha(rawfile),worker=identity(),Source_seconds=source_seconds,checkpoint_before_compatibility_and_pool=True,processed_raw_full_outputs_retained=True,native_checkpoint_sha256=sha(rawfile.with_name('native_Source.pt')),GT_read=False));origin=dict(kind='new_Source',sha256=sha(rawfile),Source_seconds=source_seconds);del frames,paidstate
   a,c,proof,diag=build(ds,row,cap,raw);save(cp,c);cold=torch.load(cp,map_location='cpu',weights_only=False,mmap=True);b=replay(cold);assert all(np.array_equal(v,b[k])for k,v in a.items());out.parent.mkdir(parents=True,exist_ok=True)
   with out.with_suffix('.tmp').open('wb')as f:np.savez_compressed(f,**a)
   os.replace(out.with_suffix('.tmp'),out)
   with np.load(out)as z:assert set(a)==set(z.files)and all(np.array_equal(v,z[k])for k,v in a.items())
   rec=dict(path=str(out.relative_to(F)),sha256=sha(out),compact_path=str(cp.relative_to(F)),compact_sha256=sha(cp),parent=par,proof=proof,diagnostics=diag,raw_origin=origin,worker=identity(),query_elapsed_wall_seconds=time.perf_counter()-tick,GT_read=False,new_expert=0,all_arrays_cold_replay_exact=True);atomic(rp,rec);files[key]=dict(rec,receipt_path=str(rp.relative_to(F)),receipt_sha256=sha(rp));counts['queries']+=1
   if origin['kind']=='new_Source':
    if len(paid)<2:paid.append(key)
    if len(paid)==2 and not(dest/'root_actual_GPU_bridge.json').exists():
     atomic(dest/'bridge_waiting.json',dict(status='WAIT_root_first_two_paid_raw_array_bridge',keys=paid,worker=identity()))
     while not(dest/'root_actual_GPU_bridge.json').exists():resources(start);time.sleep(3)
    if len(paid)==2:
     ack=read(dest/'root_actual_GPU_bridge.json');assert ack['status']=='passed_actual_two_new_Source_bridge'and ack['plan_fingerprint']==p['fingerprint']
     for k in set(paid+[key]):
      f=dest/'active'/k/'Source.pt'
      if f.exists():
       rr=read(dest/'receipts'/(k+'.json'));assert sha(f)==rr['raw_origin']['sha256']and sha(F/rr['compact_path'])==rr['compact_sha256']and sha(F/rr['path'])==rr['sha256'];native_file=f.with_name('native_Source.pt');native_sha=sha(native_file);assert native_sha==read(f.with_name('Source.json'))['native_checkpoint_sha256'];f.unlink();native_file.unlink();atomic(f.with_name('disposed.json'),dict(original_raw_sha256=rr['raw_origin']['sha256'],native_checkpoint_sha256=native_sha,committed_compact_sha256=rr['compact_sha256'],root_bridge_passed=True))
   atomic(dest/'progress.json',dict(completed=counts['queries'],total=COUNTS[ds],counts=counts,worker=identity(),GT_read=False));print(ds,counts,flush=True);del cap,raw,a,c,cold,b;gc.collect()
  assert counts['queries']==COUNTS[ds]and counts['new_Source']<=COUNTS[ds]-192;atomic(dest/'sealed.json',dict(queries=len(files),files=files,counts=counts,worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,GT_read=False,plan_fingerprint=p['fingerprint']));atomic(dest/'GPU_state.json',dict(status='complete_sealed',counts=counts))
 except BaseException:atomic(dest/'GPU_state.json',dict(status='failed_preserved_no_retry',counts=counts,error=traceback.format_exc(),worker=identity()));raise
 finally:
  if model is not None:model.cpu()
  lock.close()
def score(ds):
 os.environ['CUDA_VISIBLE_DEVICES']='';torch.set_num_threads(1);validate_full();seal=read(F/ds/'sealed.json');assert seal['queries']==COUNTS[ds]and not seal['GT_read'];truth=selected.labels(ds);out=[];checks=0;mx=0.
 from full_score import metric
 for row in selected.records(ds):
  e=seal['files'][row['id']];assert sha(F/e['path'])==e['sha256']and sha(F/e['receipt_path'])==e['receipt_sha256'];g=truth[row['id']];vals={}
  with np.load(F/e['path'])as z:
   r={**row,'metadata':{**row['metadata'],'frame_ids':z['frame_ids'].tolist()}}
   for arm in ARMS+['Current']:vals[arm],err=metric(z[arm],r,{int(k):v for k,v in g['boxes'].items()},g['interval'],z[arm+'_interval'].tolist(),z['positions8']);checks+=1;mx=max(mx,err)
  out.append(dict(id=row['id'],parent=str(g['parent']),duration_group='short_le5s'if(g['interval'][1]-g['interval'][0])/row['metadata']['fps']<=5 else'long_gt5s',values=vals))
 res=selected.suite().parent_report(out,ARMS+['Current'],PAIRS,Q/ds);selected.suite().signed_tails(out,PAIRS,Q/ds);atomic(F/ds/'metrics.json',dict(private_rows=out));atomic(F/ds/'score_receipt.json',dict(status='scored',queries=len(out),official_checks=checks,max_discrepancy=mx,seal_sha256=sha(F/ds/'sealed.json'),metrics_sha256=sha(F/ds/'metrics.json')))
def audit(ds):
 os.environ['CUDA_VISIBLE_DEVICES']='';torch.set_num_threads(1);validate_full();infer_guard();seal=read(F/ds/'sealed.json');sc=read(F/ds/'score_receipt.json');assert sc['seal_sha256']==sha(F/ds/'sealed.json')and sc['metrics_sha256']==sha(F/ds/'metrics.json')
 for key,e in seal['files'].items():
  for name,h in [('path','sha256'),('compact_path','compact_sha256'),('receipt_path','receipt_sha256')]:assert sha(F/e[name])==e[h]
  c=torch.load(F/e['compact_path'],map_location='cpu',weights_only=False,mmap=True);a=replay(c)
  with np.load(F/e['path'])as z:assert set(a)==set(z.files)and all(np.array_equal(v,z[k])for k,v in a.items())
 rr=read(F/ds/'metrics.json')['private_rows'];report=read(Q/ds/'report.json')
 for arm,mm in report['means'].items():
  for k,v in mm.items():
   vv=[r['values'][arm][k]for r in rr if r['values'][arm][k]is not None];assert(v is None and not vv)or abs(v-float(np.mean(vv)))<1e-12
 receipt=dict(status='verified_full_domain_ROI_and_uniform_gamma',queries=len(rr),all_arrays_cold_exact=True,all_file_SHA_and_means_exact=True,score_sha256=sha(F/ds/'score_receipt.json'),metrics_sha256=sha(F/ds/'metrics.json'),seal_sha256=sha(F/ds/'sealed.json'),inherited_Source_expert_cost_nonzero=True);atomic(F/ds/'reporting_receipt.json',receipt);atomic(Q/ds/'reporting_receipt.json',receipt)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','preflight','predict','score','audit']);ap.add_argument('--dataset',choices=COUNTS);x=ap.parse_args();globals()[x.stage](*([]if x.stage in ['prepare','preflight']else[x.dataset]))

