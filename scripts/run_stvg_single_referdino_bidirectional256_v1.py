"""Actual single-expert replacement; frozen inputs and first-query full profiles."""
import sys,os,time,gc,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_single_referdino_bidirectional256_common_v1 import ROOT,D,P,OLD,ROWS,read,put,sha,load,save,verify
PROFILE_ACTIVE=False
def deny_independent_detector(event,args):
 if event=='open'and args and isinstance(args[0],(str,bytes,Path)):
  s=str(args[0]).lower()
  if '/checkpoints/groundingdino'in s or 'grounding-dino-tiny'in s:raise PermissionError('Independent detector forbidden: '+s)
  if PROFILE_ACTIVE and any(x in s for x in ['/artifacts/stvg_spatial_strong_bidirectional256_v1/','/artifacts/stvg_referdino_condition_hc128_v1/','/artifacts/stvg_vid_joint_parameter_search128_v1/']):raise PermissionError('Fresh single-expert profile cannot read historical prediction/detector caches: '+s)
def main(ds):
 import torch,numpy as np
 from methods.stvg_referdino_opd_v1.runtime import setup,lease
 from methods.stvg_single_referdino_opd_v1.predictor import SingleReferDINOOPDPredictor
 from methods.stvg_single_referdino_opd_v1.runtime import load_source
 from methods.stvg_single_referdino_opd_v1.positions import observation_positions
 from methods.stvg_referdino_opd_v1._source_forward import frozen_forward
 from methods.decota_final_simplified_v1.objectives import prediction
 from methods.decota_final_simplified_v1.tensors import detached,state_hash
 from scripts.run_decota_paper_main_v1 import frames_for
 from main_method.runtime import resource_guard
 global PROFILE_ACTIVE
 setup();sys.addaudithook(deny_independent_detector);p=verify();lock=lease();start=time.perf_counter();cfg=p['configs'][ds];records=[r for r in p['records']if r['dataset']==ds];raw=read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows'];new_source=updates=count=0;failure=None;model=None;mh=None
 def check():resource_guard();assert time.perf_counter()-start<3600
 try:
  for f,h in p['checkpoints'].items():assert sha(ROOT/f)==h
  for i,rec in enumerate(records):
   check();a=rec['alias'];dest=D/f'predictions/{ds}/{a}.pt'
   if dest.exists():continue
   cachefile=D/f'inputs/{ds}/{a}.pt';cache=load(cachefile);row=dict(raw[rec['query_ordinal']]);subject=read(ROOT/rec['subject_path']);row['parses']=dict(subject=subject['parses']['subject']);assert not subject['GT_read']
   if i==0 and not(D/f'PROFILE_{ds}.json').exists():
    tick=time.perf_counter();frames,ids,_=frames_for(ds,row,collections.OrderedDict());decode_s=time.perf_counter()-tick;predictor=SingleReferDINOOPDPredictor(ds,config=cfg,check=check);torch.cuda.reset_peak_memory_stats();tick=time.perf_counter();PROFILE_ACTIVE=True
    try:
     fresh=predictor.capture(frames,ids,row['input']);out=predictor.adapt(fresh,a);torch.cuda.synchronize();wall=time.perf_counter()-tick+decode_s
    finally:PROFILE_ACTIVE=False
    assert torch.equal(fresh['Source'],cache['Source']),'fresh Source differs'
    assert fresh['planned_positions']==cache['planned_positions']
    assert torch.equal(fresh['teacher_native']['boxes'],cache['teacher_native']['boxes']),'single-checkpoint ReferDINO differs from cached native'
    assert np.array_equal(fresh['E'],cache['E']),'ERef initialization differs'
    assert all(np.array_equal(fresh['posterior'][k],cache['posterior'][k])and fresh['intervals'][k]==cache['intervals'][k]for k in ['Native','Strong'])
    assert fresh['teacher_native']['load_audit']['independent_GroundingDINO_checkpoint_loaded']is False
    fresh.update(alias=a,dataset=ds,query_ordinal=rec['query_ordinal'],needs_source_capture=False,provenance=dict(single_expert_fresh_profile=True));save(cachefile,fresh);cache=fresh;model=predictor.source;mh=state_hash(model.state_dict());new_source+=1
    profile=dict(status='pass',alias=a,complete_single_seed_wall_seconds=wall,decode_seconds=decode_s,capture_cost=fresh['cost'],adapt_cost=out['cost'],peak_GPU_bytes=max(torch.cuda.max_memory_allocated(),fresh['cost']['peak_GPU_bytes']),new_Source224_queries=1,new_Source448_queries=1,new_ReferDINO_queries=1,new_ReferDINO_frames=8,independent_GroundingDINO_queries=0,historical_prediction_cache_reads_during_fresh_pipeline=0,Source_posterior_teacher_and_ERef_exact=True,expert_count=1,not_panel_average=True)
    put(D/f'PROFILE_{ds}.json',profile);put(P/f'COMPLETE_PROFILE_{ds}_PUBLIC.json',profile);del frames,predictor
   else:
    if model is None:model=load_source(cfg);mh=state_hash(model.state_dict())
    if cache['needs_source_capture']:
     tick=time.perf_counter();frames,ids,_=frames_for(ds,row,collections.OrderedDict());model.cuda();batch,recs,base=frozen_forward(model,frames,row);native=prediction(base.zero['logits'],base.zero['boxes'],recs,ids);torch.cuda.synchronize()
     assert torch.equal(native['boxes'].cpu(),cache['Source'])and ids==cache['frame_ids'];assert observation_positions(ids,native['indices'])==cache['planned_positions']
     cache['spatial_inputs']=detached(base.spatial_inputs,'cpu');cache['needs_source_capture']=False;cache['new_Source_capture_seconds']=time.perf_counter()-tick;save(cachefile,cache);new_source+=1;del frames,batch,base;model.cpu();gc.collect();torch.cuda.empty_cache()
    predictor=SingleReferDINOOPDPredictor(ds,config=cfg,source=model,check=check);out=predictor.adapt(cache,a);del predictor
   updates+=out['cost']['actual_updates'];count+=1
   assert out['cost']['actual_updates']==cfg['steps'];assert out['initialization']['evidence_sha256']==cache['initialization']['evidence_sha256']
   prediction_cache=dict(alias=a,dataset=ds,query_ordinal=rec['query_ordinal'],frame_ids=cache['frame_ids'],planned_positions=cache['planned_positions'],Source=out['Source'],E_Ref=out['E_Ref'],Live=out['Live'],intervals=cache['intervals'],posterior=cache['posterior'],initialization=out['initialization'],shared_ReferDINO_evidence_hash=out['initialization']['evidence_sha256'],expert_count=1,GT_read=False,cost=out['cost'])
   save(D/f'traces/{ds}/{a}.pt',dict(fit=out['fit'],teacher=cache['teacher_native'],initialization=out['initialization'],input_sha256=sha(cachefile),expert_count=1,GT_read=False));save(dest,prediction_cache)
   put(D/'STATE.json',dict(status='running',dataset=ds,done=len(list((D/f'predictions/{ds}').glob('*.pt'))),total=128,new_updates_this_invocation=updates,new_Source_queries_this_invocation=new_source,pid=os.getpid(),new_independent_GroundingDINO_queries=0));print('SINGLE_REF',ds,a,cfg['steps'],round(out['cost']['fit_seconds'],3),out['initialization']['support_frames'],flush=True)
   del cache,out,prediction_cache;gc.collect();torch.cuda.empty_cache()
  model.cpu();assert state_hash(model.state_dict())==mh;assert all(not x.requires_grad and x.grad is None for x in model.parameters());verify()
  files={str(f.relative_to(ROOT)):sha(f)for f in sorted((D/f'predictions/{ds}').glob('*.pt'))};assert len(files)==128
  put(D/f'BARRIER_{ds}.json',dict(status='sealed',files=files,count=128,expert_count=1,GT_read=False,epoch=time.time()))
  if all((D/f'BARRIER_{k}.json').exists()for k in ['hc2','vidstg']):put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='sealed',parents=256,all_before_GT=True,barriers={k:sha(D/f'BARRIER_{k}.json')for k in ['hc2','vidstg']},expert_count=1,epoch=time.time()))
  print('DATASET_DONE',ds,count,updates,flush=True)
 except BaseException as e:failure=repr(e);put(D/f'FAILURE_{ds}_{time.time_ns()}.json',dict(error=failure,new_updates=updates,new_Source_queries=new_source,completed=count));raise
 finally:
  put(D/f'COST_{ds}_{time.time_ns()}.json',dict(worker_wall_seconds=time.perf_counter()-start,new_primary_fits=count,new_primary_updates=updates,new_Source224_queries=new_source,new_Source448_queries=int((D/f'PROFILE_{ds}.json').exists()),new_ReferDINO_query_forwards=int((D/f'PROFILE_{ds}.json').exists()),new_independent_GroundingDINO_forwards=0,peak_GPU_bytes=torch.cuda.max_memory_allocated(),failure=failure));lock.close()
if __name__=='__main__':main(sys.argv[1])
