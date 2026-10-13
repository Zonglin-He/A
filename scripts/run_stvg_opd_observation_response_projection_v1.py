"""Finite actual decoder qualification; no signals to cancelled baselines."""
import collections,gc,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_observation_response_projection_common_v1 import *

def scientific(x):
 # Remove instrumentation only; keep every model/math/proposal/readout value.
 if isinstance(x,dict):return {k:scientific(v) for k,v in x.items() if k not in ['seconds','elapsed_seconds','fit_seconds','CUDA_peak_allocated','time']}
 if isinstance(x,(tuple,list)):return [scientific(v) for v in x]
 return x

def qualification():
 import torch
 from scripts.stvg_opd_design_ablation_common_v1 import activate_old
 activate_old();verify();standby()
 from scripts.run_stvg_opd_expanded_after_p8_priority001 import gpu
 handle=gpu()
 from scripts.run_decota_paper_main_v1 import model_for,read_row,frames_for,unpack_expert
 from scripts.stvg_opd_paper_inputs_v1 import observation
 from scripts.stvg_opd_legacy_input_bridge003 import normalized
 from vg_tta.stvg_opd_frozen_mask_cache012 import frozen_forward
 from methods.decota_final_simplified_v1.objectives import prediction
 from methods.decota_final_simplified_v1.tensors import state_hash
 from vg_tta.stvg_opd_common_state_geometry_arithmetic004 import bare
 from scripts.run_stvg_opd_LN_state_intervention_v1 import fit_one
 from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal
 from vg_tta.stvg_opd_observation_response_projection_v1 import TrickReplay,readout,shadows,step
 from vg_tta.decota_optimizer_posterior_r1_v1 import Energy
 from scripts.decota_paper_common_v1 import budget
 d=read(BASE/'DESIGN_LOCK.json');rr=[r for r in d['records'] if r['index'] in d['qualification_indices']]
 count=0;replays=0;captures=0;bindings={};model=None;dataset=None;cache=collections.OrderedDict();start=time.time()
 try:
  for r in rr:
   budget();standby()
   if dataset!=r['dataset']:
    if model is not None:assert state_hash(model.state_dict())==mh;del model;cache.clear();gc.collect();torch.cuda.empty_cache()
    torch.manual_seed(20260920);np.random.seed(20260920);model=model_for(r['source']);mh=model._fixed_full_state_hash;dataset=r['dataset']
   pair=old.pair_for(r);initial=old.states(pair)['Inherited_order1'];reference_old=load(ROOT/r['A_path'])
   inp=normalized(load(ROOT/r['originals'][0]['input_path']));expert=unpack_expert(inp['expert'])
   assert mh==inp['source_model_state_sha256']==r['source_model_sha256'] and old.digest(inp['expert'])==pair['evidence_sha256'][0]
   for repeat in [0,1]:
    row=read_row(dataset,r['query_ordinal']);frames,ids,_=frames_for(dataset,row,cache);shifted,pixel,spec=observation(dataset,row,'clean',frames)
    batch,records,base=frozen_forward(model,shifted,row);captures+=1
    native=prediction(base.zero['logits'],base.zero['boxes'],records,ids)
    assert pixel==r['pixel_sha256']==inp['pixel_sha256'] and ids==inp['frame_ids'] and spec is None
    assert torch.equal(native['boxes'].cpu(),inp['native_boxes']) and native['physical_interval']==inp['interval']
    assert old.digest(base.initial)==old.digest(pair['source_initial_state'])
    del frames,shifted,batch;tick=time.perf_counter()
    original,auditA=fit_one(base,row,expert,initial,load(ROOT/r['originals'][0]['path'])['fit'],True)
    cmp=dict(tensors=0,tensor_coordinates=0,scalar_values=0);equal(bare(original),bare(reference_old['fit']),cmp)
    equal(original['geometry'],reference_old['fit']['geometry'],cmp)
    a_path=BASE/'qualification'/f"{r['index']:05}"/f'A_complete_repeat{repeat}.pt'
    az=dict(fit=original,audit=auditA,complete_original_comparison=cmp,index=r['index'],dataset=dataset,
      repeat=repeat,runtime_sha256=sha(BASE/'RUNTIME.json'),GT_read=False,elapsed_seconds=time.perf_counter()-tick)
    if a_path.exists():
     previous=load(a_path);equal(bare(original),bare(previous['fit']),cmp);equal(original['geometry'],previous['fit']['geometry'],cmp)
     receipt=BASE/'Adam_snapshot_storage002'/'ROOT_REUSED_A_COMPLETE.json'
     if not receipt.exists():write(receipt,dict(status='pass',scope='fresh exact complete A replay matches the previously accepted original complete fit; preserved file reused',count=1,sha256=sha(a_path),GT_read=False,time=time.time()))
    else:save(a_path,az)
    bindings[str(a_path.relative_to(ROOT))]=sha(a_path);replays+=1
    lambda_R=None
    for k in [0,r['config']['steps']//2,r['config']['steps']-1]:
     state=original['path'][k]['state'];base.restore(state);rp=TrickReplay(base,ids,row['key'],{})
     with torch.no_grad():entering=reference_old['fit']['before'].to(base.delta);current=readout(rp,[])[0]
     assert torch.equal(current.cpu(),original['path'][k]['boxes'])
     support=Energy(expert,entering,'top1');positions=[int(t[0]) for t in support.frames]
     targets=torch.cat([t[1] for t in support.frames],0).detach() if positions else entering.new_empty((0,4))
     ab=original['geometry']['rounds'][k]['Adam_before'] if not original['empty'] else None
     for arm in ARMS:
      budget();standby();rp.restore(state);pp,opt=shadows(rp,r['config'],ab)
      h,w,lam=step(rp,expert,positions,targets,r['visited'],row['key'],r['config'],k,entering,arm,pp,opt,
       lambda_R=lambda_R,need_curvature=(arm=='P' and k==0),check_derivatives=(arm=='P' and repeat==0),seed=SEED+r['index']*100+k)
      if arm=='P' and k==0:lambda_R=lam if lam is not None else 0.
      if positions and w['updated'] and arm in ['P','F']:
       assert torch.equal(w['gradient'],original['path'][k]['update']['gradient'])
       assert np.array_equal(w['d_A'].numpy(),original['path'][k]['update']['raw'].numpy().astype(float))
       assert torch.equal(w['original_proposal_boxes'],original['path'][k+1]['boxes'])
      if arm=='R' and k==0 and positions and w['updated']:
       assert torch.equal(w['gradient'],original['path'][k]['update']['gradient'])
       assert np.array_equal(w['d_A'].numpy(),original['path'][k]['update']['raw'].numpy().astype(float))
      if not positions:
       assert not w['updated'] and torch.equal(h['boxes'],reference_old['fit']['before'])
       assert old.digest(rp.state())==old.digest(state)
      z=dict(index=r['index'],dataset=dataset,repeat=repeat,arm=arm,boundary=k,entry=h,witness=w,
        after_state=old.canonical(detached_state(rp)),reference_complete_fit_sha256=r['A_sha256'],
        initial_reference_sha256=old.digest(initial),pixel_sha256=pixel,source_model_sha256=mh,
        frame_ids=ids,Native_interval=inp['interval'],runtime_sha256=sha(BASE/'RUNTIME.json'),revision_runtime_bindings=revision_bindings(),
        GT_read=False,actual_new_DINO_calls=0,actual_full_fit=False,scope='one-step qualification at original A saved state and moments')
      p=BASE/'qualification'/f"{r['index']:05}"/(f'{arm}_k{k}_repeat{repeat}.pt')
      if repeat:
       first=load(p.with_name(f'{arm}_k{k}_repeat0.pt'))
       # Qualification repeats use the same mathematical code and full grid;
       # FD instrumentation exists only in repeat0 and is compared separately.
       firstw=dict(first['witness']);nextw=dict(w)
       firstw.pop('derivative_checks',None);nextw.pop('derivative_checks',None)
       # P repeat0 includes full J_U for derivative checks; P repeat1 only
       # needs J_O except at entry. Compare the overlapping Jacobian exactly.
       firstw.pop('J_U',None);nextw.pop('J_U',None);firstw.pop('Jacobian',None);nextw.pop('Jacobian',None)
       cc=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
       try:
        equal(scientific(first['entry']),scientific(h),cc);equal(scientific(firstw),scientific(nextw),cc)
        assert first['after_state']==z['after_state'];z['repeat_scientific_status']='pass_bitwise'
       except AssertionError:
        # A failed repeat is a measured qualification result, not permission
        # to omit this query or abort its remaining fixed checks.
        z['repeat_scientific_status']='failed_bitwise'
        first_raw=first['entry'].get('update',{}).get('raw');next_raw=h.get('update',{}).get('raw')
        z['repeat_max_raw_difference']=float((first_raw.double()-next_raw.double()).abs().max()) if first_raw is not None else 0.
        z['repeat_max_boxes_difference']=float((first['entry']['boxes'].double()-h['boxes'].double()).abs().max())
       z['repeat_scientific_comparison']=cc
      if p.exists():
       prior=load(p);cc=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
       equal(scientific(prior['entry']),scientific(h),cc);equal(scientific(prior['witness']),scientific(w),cc)
       assert prior['after_state']==z['after_state']
       receipt=BASE/'repeat_scope_readback004'/'reused_prefix'/f"{r['index']:05}"/p.name
       write(receipt,dict(status='pass',scope='full accepted one-step scientific witness bitwise rechecked; no payload overwritten',count=1,sha256=sha(p),comparisons=cc,GT_read=False,time=time.time()))
      else:
       save(p,z)
       write(p.with_suffix('.json'),dict(status='accepted_complete_one_step_qualification',index=r['index'],dataset=dataset,arm=arm,
        count=count+1,total=144,sha256=sha(p),bytes=p.stat().st_size,GT_read=False,time=time.time()))
      bindings[str(p.relative_to(ROOT))]=sha(p);count+=1
      status(BASE/'GPU_STATUS.json',dict(status='running_actual_technical_qualification',dataset=dataset,done=count,total=144,
       original_complete_replays=replays,captures=captures,pid=os.getpid(),GT_read=False,time=time.time()))
      print('OBS_RESPONSE_QUALIFICATION',count,144,dataset,arm,k,'fallback',w.get('fallback'),flush=True)
      del pp,opt,h,w,z
     del rp
    del base,original,auditA;gc.collect();torch.cuda.empty_cache()
   del pair,initial,reference_old,inp,expert
  assert count==144 and captures==16 and replays==16 and state_hash(model.state_dict())==mh
  verify();standby()
  write(BASE/'QUALIFICATION_BARRIER.json',dict(status='sealed',scope='all8 fixed queries/two fresh captures/16 exact full A replays/144 fixed local proposals',
   files=bindings,count=count,total=144,qualification_queries=8,original_complete_replays=replays,actual_frozen_captures=captures,
   source_state_restored=True,runtime_sha256=sha(BASE/'RUNTIME.json'),revision_runtime_bindings=revision_bindings(),GT_read=False,new_DINO_calls=0,elapsed_seconds=time.time()-start,time=time.time()))
 finally:
  if model is not None:del model
  cache.clear();gc.collect();torch.cuda.empty_cache();handle.close()

def detached_state(rp):
 from methods.decota_final_simplified_v1.tensors import detached
 return detached(rp.state(),'cpu')

if __name__=='__main__':
 try:
  mode=sys.argv[1]
  if mode=='prepare':prepare()
  elif mode=='qualification':qualification()
  else:raise ValueError(mode)
 except BaseException:
  p=BASE/'failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=False);(p/'traceback.txt').write_text(traceback.format_exc())
  status(BASE/'STATUS.json',dict(status='failed_preserved_pending_root',stage=sys.argv[1],evidence=str(p.relative_to(ROOT)),GT_read=False,time=time.time()));raise
