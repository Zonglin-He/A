"""Frozen Source cache and actual per-query state-only optimizer search.

The GPU process never loads GT. Scoring and adaptive fine-plan selection are
separate CPU stages after a complete phase barrier.
"""
import sys,os,time,gc,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_vid_joint_parameter_common_v1 import ROOT,D,P,OLD,read,put,sha,load,save,verify
def main(phase):
 import numpy as np,torch
 from methods.stvg_referdino_opd_v1.runtime import setup,lease
 from scripts.run_stvg_spatial_strong_bidirectional256_v1 import source_for
 from methods.stvg_referdino_opd_v1._source_forward import frozen_forward
 from methods.decota_final_simplified_v1.objectives import prediction
 from methods.decota_final_simplified_v1.tensors import detached,state_hash
 from methods.stvg_referdino_opd_v1.state import Replay
 from methods.stvg_referdino_opd_v1.policy import fit
 from scripts.run_decota_paper_main_v1 import frames_for
 from main_method.runtime import resource_guard
 setup();p=verify();lock=lease();start=time.perf_counter();prior=sum(x['wall_seconds']for x in read(D/'GPU_LEDGER.json')['invocations'])if(D/'GPU_LEDGER.json').exists()else 0.
 count=updates=0;m=None;initial_hash=None;failure=None;phase_files={};torch.cuda.reset_peak_memory_stats()
 def check():
  resource_guard();assert prior+time.perf_counter()-start<p['cumulative_GPU_worker_wall_limit_seconds'],'GPU worker wall budget'
 def new_source():
  nonlocal m,initial_hash
  tick=time.perf_counter();m=source_for('vidstg');initial_hash=state_hash(m.state_dict());return time.perf_counter()-tick
 try:
  for f,h in p['checkpoint'].items():assert sha(ROOT/f)==h
  raw=read(ROOT/'artifacts/decota_paper_experiments_v1/vidstg/PLAN.json')['rows'];loading=new_source();source_cost=[]
  configs=p['configs']if phase=='coarse'else read(D/'FINE_PLAN_LOCK.json')['configs']
  if phase=='coarse':
   for rec in p['records']:
    check();a=rec['alias'];dest=D/'inputs'/f'{a}.pt'
    if dest.exists():continue
    old=load(OLD/f'predictions/vidstg/{a}.pt');tr=load(OLD/f'traces/vidstg/{a}.pt');row=dict(raw[rec['query_ordinal']]);sf=ROOT/rec['subject_path'];assert sha(sf)==rec['subject_sha256'];row['parses']=read(sf)['parses'];tick=time.perf_counter();frames,ids,_=frames_for('vidstg',row,collections.OrderedDict());decode_s=time.perf_counter()-tick;assert ids==old['frame_ids']
    torch.cuda.synchronize();tick=time.perf_counter();batch,recs,base=frozen_forward(m,frames,row);native=prediction(base.zero['logits'],base.zero['boxes'],recs,ids);torch.cuda.synchronize();forward_s=time.perf_counter()-tick;source=native['boxes'].detach().cpu()
    assert np.array_equal(source.double().numpy(),old['Source']),(a,'Source exact reproduction failed')
    inp=dict(alias=a,frame_ids=ids,Source=source,E=old['E'],interval=old['intervals']['Strong'],planned_positions=old['planned_positions'],spatial_inputs=detached(base.spatial_inputs,'cpu'),teacher_native=tr['teacher'],GT_read=False,
       provenance=dict(original_prediction_sha256=sha(OLD/f'predictions/vidstg/{a}.pt'),original_trace_sha256=sha(OLD/f'traces/vidstg/{a}.pt'),Source_model_sha256=p['checkpoint'],physical_frames_unchanged=True),cost=dict(Source_forward_seconds=forward_s,decode_seconds=decode_s,new_Source_offset_forwards=2,new_expert_calls=0))
    save(dest,inp);del frames,batch,base,source,tr,old;gc.collect();torch.cuda.empty_cache()
    if a==p['records'][0]['alias']and not(D/'BASELINE_COMPATIBILITY.json').exists():
     rp0=Replay(m,inp,inp['Source'],inp['E'],inp['planned_positions'],check);oldfit=load(OLD/f'traces/vidstg/{a}.pt')['fits']['Live_41'];rep=fit(rp0,inp['teacher_native']['boxes'],a,'Live',41,p['configs'][p['baseline']]);updates+=40
     assert all(torch.equal(x['boxes'],y['boxes'])and torch.equal(x['state'],y['state'])for x,y in zip(rep['path'],oldfit['path'])),'baseline paths differ'
     put(D/'BASELINE_COMPATIBILITY.json',dict(status='pass',alias=a,all41_saved_states_and_trajectories_bitexact=True,new_control_updates=40,GT_read=False));del rp0,oldfit,rep;gc.collect();torch.cuda.empty_cache();print('BASELINE_COMPATIBILITY_PASS',a,flush=True)
    del inp;source_cost.append(dict(alias=a,forward_seconds=forward_s,decode_seconds=decode_s));put(D/'STATE.json',dict(status='running',phase='Source_cache',done=len(list((D/'inputs').glob('*.pt'))),total=128,pid=os.getpid()));print('SOURCE_CACHED',a,round(forward_s,3),flush=True)
   put(D/'SOURCE_BARRIER.json',dict(files={str(f.relative_to(ROOT)):sha(f)for f in sorted((D/'inputs').glob('*.pt'))},all128_exact_original_Source=True,GT_read=False,epoch=time.time(),cost=source_cost));assert len(read(D/'SOURCE_BARRIER.json')['files'])==128
  else:assert (D/'SOURCE_BARRIER.json').exists()
  m.cpu();gc.collect();torch.cuda.empty_cache();m.ground_decoder.decoder.cuda();decoder_hash=state_hash(m.ground_decoder.decoder.state_dict())
  for rec in p['records']:
   check();a=rec['alias'];inpfile=D/'inputs'/f'{a}.pt';assert sha(inpfile)==read(D/'SOURCE_BARRIER.json')['files'][str(inpfile.relative_to(ROOT))];cache=load(inpfile);rp=Replay(m,cache,cache['Source'],cache['E'],cache['planned_positions'],check)
   try:
    if phase=='coarse'and a==p['records'][0]['alias']and not(D/'BASELINE_COMPATIBILITY.json').exists():
     old=load(OLD/f'traces/vidstg/{a}.pt')['fits']['Live_41'];rep=fit(rp,cache['teacher_native']['boxes'],a,'Live',41,p['configs'][p['baseline']]);updates+=40
     assert all(torch.equal(x['boxes'],y['boxes'])and torch.equal(x['state'],y['state'])for x,y in zip(rep['path'],old['path'])), 'baseline paths differ'
     put(D/'BASELINE_COMPATIBILITY.json',dict(status='pass',alias=a,all41_saved_states_and_trajectories_bitexact=True,new_control_updates=40,GT_read=False));del old,rep
    for key,cfg in configs.items():
     dest=D/'fits'/phase/key/f'{a}.pt'
     if phase=='coarse'and key==p['baseline']:
      phase_files[f'{key}/{a}']=dict(path=str((OLD/f'traces/vidstg/{a}.pt').relative_to(ROOT)),sha256=sha(OLD/f'traces/vidstg/{a}.pt'),historical_reuse=True);continue
     if dest.exists():
      phase_files[f'{key}/{a}']=dict(path=str(dest.relative_to(ROOT)),sha256=sha(dest),historical_reuse=False);continue
     check();out=fit(rp,cache['teacher_native']['boxes'],a,'Live',41,cfg);updates+=out['actual_updates'];assert out['actual_updates']==40
     h=save(dest,dict(alias=a,config_id=key,fit=out,input_sha256=sha(inpfile),GT_read=False));phase_files[f'{key}/{a}']=dict(path=str(dest.relative_to(ROOT)),sha256=h,historical_reuse=False);count+=1;print('FIT',phase,key,a,round(out['seconds'],3),flush=True);del out;gc.collect();torch.cuda.empty_cache()
    assert not rp.state.detach().any()
   finally:del rp,cache;gc.collect();torch.cuda.empty_cache()
   put(D/'STATE.json',dict(status='running',phase=phase,last_alias=a,new_fits_this_invocation=count,new_updates_this_invocation=updates,pid=os.getpid()))
  assert len(phase_files)==len(configs)*128
  assert state_hash(m.ground_decoder.decoder.state_dict())==decoder_hash
  m.cpu();assert state_hash(m.state_dict())==initial_hash;verify()
  put(D/f'{phase.upper()}_FIT_BARRIER.json',dict(status='sealed',files=phase_files,configs=configs,parents=128,GT_read=False,epoch=time.time(),plan_sha256=sha(D/'PLAN_LOCK.json')))
  put(D/'STATE.json',dict(status='phase_predictions_complete_pending_CPU',phase=phase,new_fits_this_invocation=count,new_updates_this_invocation=updates,new_expert_forwards=0));print('PHASE_DONE',phase,count,updates,flush=True)
 except BaseException as e:
  failure=repr(e);put(D/f'FAILURE_{phase}_{time.time_ns()}.json',dict(error=failure,new_fits=count,new_updates=updates,GT_read=False));raise
 finally:
  ledger=read(D/'GPU_LEDGER.json')if(D/'GPU_LEDGER.json').exists()else dict(invocations=[]);ledger['invocations'].append(dict(phase=phase,wall_seconds=time.perf_counter()-start,new_fits=count,new_updates=updates,Source_load_seconds=locals().get('loading'),peak_GPU_bytes=torch.cuda.max_memory_allocated(),failure=failure));put(D/'GPU_LEDGER.json',ledger);lock.close()
if __name__=='__main__':main(sys.argv[1])
