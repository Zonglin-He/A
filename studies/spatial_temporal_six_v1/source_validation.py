"""New source-domain validation extraction and calibration, never target tuning."""
import os
_visibility=os.environ.get('CUDA_VISIBLE_DEVICES')
from cpu import *
if _visibility is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
else:os.environ['CUDA_VISIBLE_DEVICES']=_visibility
import gc,subprocess,traceback
import matching
C=D/'source_validation';CONFIGS=[dict(temperature=t,topk=k,context=20)for t in [.05,.1]for k in [10,20]]
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');torch.save(x,tmp);os.replace(tmp,p)
def source_plan():
 p=read(C/'plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for g in ['code_sha256','input_sha256']:
  for f,h in p[g].items():assert sha(R/f)==h,f
 return p
def prepare_source():
 assert not(C/'plan.json').exists();assert read(D/'matching_contract_preflight.json')['moving_patch_positive_control_exact'];inputs=[D/'source_validation_cohort_v1.json',D/'matching_contract_preflight.json',D/'matching_preparation_v1.json',R/'configs/default.json',RUN/'plan.json']
 for ds in COUNTS:inputs += [RUN/ds/'sealed.json']
 inputs += [R/read(R/'configs/default.json')['datasets'][t]['source_checkpoint']for t in COUNTS]
 p=dict(schema='source_validation36_each_direction_frozen_four_configs_v1',scope='72 actually new Source extractions on SOURCE-DOMAIN VALIDATION inputs, correct source-trained checkpoint for each direction. Not source-training36, not target-domain configuration selection. Opposite-direction prior target exposure disclosed. Same8existingexpert masks/boxes,0newteacher.',configs=CONFIGS,
  selection='Source validation cohort was metadata/SHA-selected before calibration. For targetHC calibrate only on36Vid using Vid-trained Source; for targetVid only36HC using HC-trained Source. Never average the two source sets to select shared parameters. No target labels may be read in acquisition or target config selection.',
  inference='Original frozen_native_forward at original224FP16visual/FP32heads, cudnnTF32True matmulFalse Torch4. Save native_Source.pt and SHA BEFORE ANY compatibility/feature check. Extract same postfusion visual tokens/valid grid; use exact old same8expert Source-grid soft masks and raw boxes, no E or Source boxes input to propagation. Reset Torch1 for matching and restore4 before every Source. Keep all source paid raw through calibration and root source acceptance.',
  bridge='First2 paid queries PER TARGET-MODEL source cohort normally WAIT until ROOT rereads full paid native state, re-extracts postfusion feature grid, compares serialized arrays/source-native checkpoint SHA, reruns all4 matching configs twice exact and acknowledges. These are new source-input/model pairs, so no claim of equality to the opposite-model historical Source outputs.',
  calibration='All72*4 predictions sealed before source labels. Source GT-supported unobserved native-frame IoU queryequal; report all configs and pure expert/Source references. Mean across queries per source domain; choose max, deterministic original config order on ties. Source performance/empty masks and feature-value controls root-reviewed before target configuration freeze. No blind target launch after source score.',
  max_new_Source=72,max_new_expert=0,max_new_optimizer=0,resources='GPU1, CPU14-15nice19idleIO, originalSourceTorch4/matching1, RAM8GiB/disk80GiB/private160GiB/72h. Wait foreignGPU and globallease; signal no other process.',
  code_sha256={str(f.relative_to(R)):sha(f)for f in [Path(__file__),S/'matching.py',S/'cpu.py',R/'stvg/native.py',R/'stvg/evidence.py',R/'stvg/runtime.py',R/'stvg/representation.py']},input_sha256={str(f.relative_to(R)):sha(f)for f in inputs})
 p['fingerprint']=digest(p);atomic(C/'plan.json',p);atomic(P/'source_validation_protocol.json',p);print(p['fingerprint'],flush=True)
def foreign():
 s=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True);out=[]
 for line in s.splitlines():
  if not line.strip():continue
  pid,name=line.split(',',1);pid=int(pid);name=name.strip()
  if pid!=os.getpid()and name not in ['/opt/todesk/bin/ToDesk_Session','/opt/todesk/bin/ToDesk_Service']:out.append(dict(pid=pid,name=name))
 return out
def match_arrays(v):
 arrays=dict(frame_ids=np.asarray(v['frame_ids']),positions8=np.asarray(v['positions8']),PureExpert=pure.pure_interpolation(v['frame_ids'],v['positions8'],v['native_boxes']),Source=v['Source'].copy());diags={}
 for j,cfg in enumerate(CONFIGS):arrays[f'Config{j}'],diags[f'Config{j}']=matching.propagate(v['features'],v['valid'],v['positions8'],v['seeds'],v['native_boxes'],v['shape'],v['image_size'],**cfg)
 return arrays,diags
def acquire():
 plan=source_plan();assert not(C/'worker.json').exists()
 from stvg.runtime import setup,lease,load_source
 from stvg.native import frozen_native_forward
 from stvg.representation import observe_native_grid
 from stvg.evidence import extract_visual_grids
 from stvg.tensors import detached
 from stvg.query import QuerySubjectParser
 from stvg.decode import decode
 setup();torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False;start=time.perf_counter();lock=None;model=None;parser=None
 while lock is None:
  parent.resources(start)
  if not foreign():
   try:lock=lease()
   except BlockingIOError:pass
  if lock is None:atomic(C/'waiting_resources.json',dict(foreign_compute=foreign(),status='waiting_no_foreign_GPU_and_globallease'));time.sleep(10)
 atomic(C/'worker.json',identity());cohorts=read(D/'source_validation_cohort_v1.json');counts=dict(new_Source=0,new_expert=0,queries=0)
 try:
  for target,cohort in cohorts.items():
   ds=cohort['source_domain'];cfg=read(R/'configs/default.json')['datasets'][target];assert cfg['source_dataset']==ds;dest=C/target;files={};paid=[];tick=time.perf_counter();atomic(dest/'model_load_attempt.json',dict(worker=identity(),before_load=True))
   model=load_source(cfg);model.cpu();atomic(dest/'model_load.json',dict(worker=identity(),elapsed_wall_seconds=time.perf_counter()-tick,checkpoint_sha256=sha(R/cfg['source_checkpoint']),loads=1));parser=QuerySubjectParser(R/'checkpoints/stanza')
   for row in cohort['rows']:
    parent.resources(start);assert sum(f.stat().st_size for f in C.rglob('*')if f.is_file())<160*2**30;key=row['id'];folder=dest/'paid'/key;assert not folder.exists();folder.mkdir(parents=True);qstart=time.perf_counter()
    while foreign():model.cpu();torch.cuda.empty_cache();parent.resources(start);time.sleep(10)
    e=fixed(RUN/ds/'sealed.json')['files'][key];cp=RUN/ds/e['capture_path'];assert sha(cp)==e['capture_sha256'];cap=torch.load(cp,map_location='cpu',weights_only=False,mmap=True);assert not cap['GT_read']
    frames,ids=decode(row['metadata'],ds);assert ids==cap['frame_ids'];parsed={'subject':parser(row['metadata']['caption'])['subject']};atomic(folder/'attempt.json',dict(worker=identity(),GT_read=False,before_Source=True,RGB_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),parent_capture_sha256=e['capture_sha256']))
    torch.set_num_threads(4);torch.manual_seed(20260920);np.random.seed(20260920);model.cuda();t=time.perf_counter()
    with torch.no_grad(),observe_native_grid(model)as grid:batch,recs,base=frozen_native_forward(model,frames,dict(input=row['metadata'],frame_ids=ids,parses=parsed))
    torch.cuda.synchronize();source_seconds=time.perf_counter()-t
    native=dict(Source=base.zero['boxes'].detach().cpu().numpy(),native_logits=detached(base.zero['logits'],'cpu'),records=recs,spatial_inputs=detached(base.spatial_inputs,'cpu'),native_local_grid=grid,frame_ids=ids,GT_read=False)
    save(folder/'native_Source.pt',native);atomic(folder/'native_Source.json',dict(sha256=sha(folder/'native_Source.pt'),Source_seconds=source_seconds,checkpoint_before_validation=True,worker=identity(),GT_read=False));counts['new_Source']+=1;model.cpu();del frames,batch,base;gc.collect();torch.cuda.empty_cache()
    feats,valid,ids2,shape,metadata=extract_visual_grids(native['spatial_inputs'],ids);assert ids2==ids;g=cap['geometry'];seeds=g['side_maps'][cap['positions8'],0].clone();assert tuple(seeds.shape[1:])==tuple(shape)
    v=dict(features=feats,valid=valid,frame_ids=ids,shape=shape,positions8=cap['positions8'],seeds=seeds,native_boxes=np.asarray(g['native_boxes']),image_size=(row['metadata']['height'],row['metadata']['width']),Source=native['Source'],GT_read=False,native_Source_sha256=sha(folder/'native_Source.pt'),parent_capture_sha256=e['capture_sha256'],source_domain=ds,target_model_key=target)
    save(folder/'matching_input.pt',v);torch.set_num_threads(1);arrays,diags=match_arrays(v);cold=torch.load(folder/'matching_input.pt',map_location='cpu',weights_only=False,mmap=True);again,_=match_arrays(cold);assert set(arrays)==set(again)and all(np.array_equal(x,again[k])for k,x in arrays.items())
    out=dest/'predictions'/(key+'.npz');out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('wb')as f:np.savez_compressed(f,**arrays)
    with np.load(out)as z:assert set(arrays)==set(z.files)and all(np.array_equal(x,z[k])for k,x in arrays.items())
    rec=dict(path=str(out.relative_to(C)),sha256=sha(out),input_path=str((folder/'matching_input.pt').relative_to(C)),input_sha256=sha(folder/'matching_input.pt'),native_path=str((folder/'native_Source.pt').relative_to(C)),native_sha256=v['native_Source_sha256'],diagnostics=diags,GT_read=False,Source_seconds=source_seconds,query_elapsed_wall_seconds=time.perf_counter()-qstart,worker=identity());rp=dest/'receipts'/(key+'.json');atomic(rp,rec);files[key]=dict(rec,receipt_path=str(rp.relative_to(C)),receipt_sha256=sha(rp));paid.append(key);counts['queries']+=1
    if len(paid)==2:
     atomic(dest/'bridge_waiting.json',dict(status='WAIT_root_source_validation_actual_bridge',keys=paid,worker=identity()))
     while not(dest/'root_actual_bridge.json').exists():parent.resources(start);time.sleep(3)
     assert read(dest/'root_actual_bridge.json')['plan_fingerprint']==plan['fingerprint']
    atomic(C/'progress.json',dict(target_model=target,source_domain=ds,completed_in_domain=len(files),total_in_domain=36,counts=counts,worker=identity(),GT_read=False));del cap,native,v,cold,arrays,again;gc.collect()
   atomic(dest/'sealed.json',dict(queries=36,files=files,GT_read=False,source_domain=ds,target_model_key=target,worker=identity(),plan_fingerprint=plan['fingerprint']));model.cpu();del model;model=None;gc.collect();torch.cuda.empty_cache()
  atomic(C/'state.json',dict(status='all72_source_features_fourconfig_predictions_sealed_pending_source_only_calibration',counts=counts,worker=identity(),whole_six_complete=False))
 except BaseException:atomic(C/'failure.json',dict(error=traceback.format_exc(),counts=counts,worker=identity(),all_paid_state_preserved=True));raise
 finally:
  if model is not None:model.cpu()
  lock.close()
def bridge(target):
 os.environ['CUDA_VISIBLE_DEVICES']='';plan=source_plan();parent.infer_guard();torch.set_num_threads(1);dest=C/target;wait=read(dest/'bridge_waiting.json');assert len(wait['keys'])==2 and not(dest/'root_actual_bridge.json').exists();cases=[]
 from stvg.evidence import extract_visual_grids
 for key in wait['keys']:
  rp=dest/'receipts'/(key+'.json');e=read(rp)
  for f,h in [('path','sha256'),('input_path','input_sha256'),('native_path','native_sha256')]:assert sha(C/e[f])==e[h]
  native=torch.load(C/e['native_path'],map_location='cpu',weights_only=False,mmap=True);v=torch.load(C/e['input_path'],map_location='cpu',weights_only=False,mmap=True);assert not native['GT_read']and not v['GT_read'];f,valid,ids,shape,_=extract_visual_grids(native['spatial_inputs'],native['frame_ids']);assert torch.equal(f,v['features'])and torch.equal(valid,v['valid'])and list(ids)==list(v['frame_ids'])and tuple(shape)==tuple(v['shape'])
  a,_=match_arrays(v);b,_=match_arrays(v);assert all(np.array_equal(x,b[k])for k,x in a.items())
  with np.load(C/e['path'])as z:assert set(a)==set(z.files)and all(np.array_equal(x,z[k])for k,x in a.items())
  cases.append(dict(native_sha256=e['native_sha256'],input_sha256=e['input_sha256'],all_actual_postfusion_features_and_all4config_arrays_cold_exact=True,GT_read=False))
 ack=dict(status='passed_root_actual_source_domain_two_paid_Source_bridge',plan_fingerprint=plan['fingerprint'],cases=cases,worker=identity(),new_Source=0,new_expert=0,new_GT=0);atomic(dest/'root_actual_bridge.json',ack);atomic(P/'source_bridges'/f'{target}.json',ack);print(target,'root actual source bridge passed',flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare_source','acquire','bridge']);ap.add_argument('--target',choices=COUNTS);x=ap.parse_args();globals()[x.stage](*([x.target]if x.stage=='bridge'else[]))
