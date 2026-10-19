"""Finite cached temporal-head probe. No RGB/expert/weight/spatial updates."""
from pathlib import Path
import sys,os,time,gc,json
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=R/'artifacts/stvg_temporal_roi_sampler128_v1';P=R/'results/stvg_error_state/2026-10-20/temporal_roi_sampler128'
OLD=R/'artifacts/stvg_temporal_object_memory128_v1';PRE=R/'artifacts/stvg_referdino_condition_hc128_v1';FIN=R/'artifacts/stvg_referdino_finalize_v1';DEV=R/'artifacts/stvg_temporal_increment128_v1'
CONDITIONS=['NativeGlobal','ValidGlobal','E_ROI','Live41_ROI','Live42_ROI','Live43_ROI','ReferROI','ShiftedReferROI']

def prepare():
 p=read(OLD/'PLAN_LOCK.json');records=[]
 for r in p['records']:
  files=[OLD/'temporal_inputs'/f"{r['alias']}.pt",OLD/'predictions'/f"{r['alias']}.pt",DEV/'priors'/f"{r['alias']}.pt"]
  records.append(dict(r,cache_pins={str(f.relative_to(R)):sha(f)for f in files}))
 own=['protocols/stvg_temporal_roi_sampler128_v1.md','vg_tta/stvg_temporal_roi_sampler128_v1.py','scripts/run_stvg_temporal_roi_sampler128_v1.py','scripts/score_stvg_temporal_roi_sampler128_v1.py','scripts/publish_stvg_temporal_roi_sampler128_v1.py','vg_tta/stvg_temporal_object_memory128_v1.py','vg_tta/stvg_temporal_foreground128_v1.py','methods/decota_final_simplified_v1/objectives.py','methods/stvg_referdino_opd_v1/config.json']
 put(D/'PLAN_LOCK.json',dict(records=records,count=128,seeds=[41,42,43],conditions=CONDITIONS,arms=['NativeMBR']+[c+n for c in CONDITIONS for n in ['Full','Fluct']],eligible=['NativeGlobalFull','NativeGlobalFluct','ReferROIFull','ReferROIFluct'],primary='ReferROIFluct',GPU_seconds_cap=900,deadline_epoch=time.time()+14400,created_epoch=time.time(),pins={f:sha(R/f)for f in own},new_experts=0,new_RGB=0,spatial_frozen=True,pretrained_weights_frozen=True,new_OPD=False))
 put(P/'PROTOCOL_PUBLIC.json',dict(count=128,exposure='RC128 exposed development, old components64 not independent',conditions=CONDITIONS,heads='existing source-supervised motion TemporalSampling, native endpoints same prior all conditions',readouts=['historical Full','historical Fluct'],primary='ReferROIFluct',eligible=['NativeGlobalFull','NativeGlobalFluct','ReferROIFull','ReferROIFluct'],all_spatial_coordinates_frozen=True,new_experts=0,new_RGB=0,new_training=0,GPU_seconds_cap=900,confirmation_gate='whole-parent fullv>=2pp and CI lower>0; no perquery selection',not_OPD=True))
 print('locked128',flush=True)

def measure():
 import torch,numpy as np
 from methods.stvg_referdino_opd_v1.runtime import setup,lease,load_source
 from main_method.runtime import resource_guard
 from methods.stvg_referdino_opd_v1._capture import transfer
 from methods.decota_final_simplified_v1.tensors import state_hash
 from vg_tta.stvg_temporal_object_memory128_v1 import prior_mask
 from vg_tta.stvg_temporal_foreground128_v1 import project
 from vg_tta.stvg_temporal_roi_sampler128_v1 import pools,properties
 from vg_tta.stvg_referdino_native32_v1 import interpolate
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
 from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,mass_table
 setup();assert properties()['status']=='pass';p=read(D/'PLAN_LOCK.json');assert all(sha(R/f)==h for f,h in p['pins'].items());lock=lease();beg=time.perf_counter();calls=0;files=[];times=[]
 def check():
  resource_guard();assert time.time()<p['deadline_epoch'] and time.perf_counter()-beg+10<900
 try:
  check();m=load_source();mh=state_hash(m.state_dict());loading=time.perf_counter()-beg
  for r in p['records']:
   check();tick=time.perf_counter();a=r['alias'];assert all(sha(R/f)==h for f,h in r['cache_pins'].items());z=load(PRE/'inputs'/f'{a}.pt');sp=load(FIN/'Live/predictions'/f'{a}.pt');ca=load(OLD/'temporal_inputs'/f'{a}.pt');old=load(OLD/'predictions'/f'{a}.pt');prior=load(DEV/'priors'/f'{a}.pt');teacher=load(PRE/'teacher0'/f'{a}.pt');ids=z['frame_ids'];ref=interpolate(ids,r['positions8'],teacher['native']['boxes'].numpy());shift=ref.copy();shift[:,0]=(shift[:,0]+.5)%1
   boxes=dict(E_ROI=np.asarray(z['E']),**{f'Live{s}_ROI':np.asarray(sp['outputs'][f'T20_Live_{s}'])for s in p['seeds']},ReferROI=ref,ShiftedReferROI=shift);ll={k:[]for k in CONDITIONS};aa={k:[]for k in CONDITIONS}
   pool_audit={};native_checks=0
   for off,(kw0,g)in enumerate(zip(ca['inputs'],ca['grids'])):
    kw=transfer(kw0,'cuda');valid=g['valid'].cuda();memory=kw['encoded_feature'];L=g['h']*g['w'];motion=memory[-L:];text=memory[:-L].mean(1).unsqueeze(0);fmap=motion.permute(1,2,0).reshape(len(valid),256,g['h'],g['w'])
    with torch.no_grad():
     native=m.t_temporal_clas(fmap,text).reshape(1,-1,1).cpu();pooled=m.t_temporal_clas(torch.nn.functional.adaptive_avg_pool2d(fmap,(1,1)),text).reshape(1,-1,1).cpu()
    assert torch.equal(native,pooled),'native classifier prepool must be exact';native_checks+=1;calls+=1;ll['NativeGlobal'].append(prior['logits'][off]);aa['NativeGlobal'].append(native)
    for c in CONDITIONS[1:]:
     check();bxs=boxes['ReferROI']if c=='ValidGlobal'else boxes[c]
     _,global_pool,roi,meta=pools(motion,valid,torch.as_tensor(bxs[off::2],device='cuda',dtype=torch.float64),g['h'],g['w']);pool=global_pool if c=='ValidGlobal'else roi
     with torch.no_grad():action=m.t_temporal_clas(pool,text).reshape(1,-1,1).cpu()
     ll[c].append(prior['logits'][off]);aa[c].append(action);pool_audit[c+str(off)]=meta;calls+=1
    del kw;torch.cuda.empty_cache()
   grid,q=mass_table(probabilities(z['records'],ll['NativeGlobal']));assert grid==ids and np.array_equal(q,prior['q']);iv=final_mbr(ids,q)['interval'];assert iv==z['interval'];outputs={'NativeMBR':dict(interval=iv)};checks={}
   for c in CONDITIONS:
    vv,audit=project(ll[c],aa[c],z['records']);outputs.update({c+n:v for n,v in vv.items()});checks[c]=audit
   files.append(save(D/'predictions'/f'{a}.pt',dict(alias=a,query_ordinal=r['query_ordinal'],frame_ids=ids,trajectories=old['trajectories'],outputs=outputs,actions=aa,endpoint_logits=ll,audit=checks,pooling_audit=pool_audit,native_equivalent_checks=native_checks,GT_read=False,spatial_updated=False)));times.append(time.perf_counter()-tick);put(D/'STATUS.json',dict(stage='measure',done=len(files),pid=os.getpid(),epoch=time.time()),True);print('foreground',a,round(time.perf_counter()-beg,2),flush=True)
   if len(files)==2:
    estimate=loading+max(times)*128+30;put(D/'PILOT2.json',dict(seconds=times,model_load=loading,estimated_GPU_seconds=estimate,GT_read=False));assert estimate<900
   del z,sp,ca,old,prior,teacher,ll,aa,boxes,outputs;gc.collect();torch.cuda.empty_cache()
  assert state_hash(m.state_dict())==mh and all(not x.requires_grad and x.grad is None for x in m.parameters());put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(count=128,files=files,all_before_GT=True,plan_sha256=sha(D/'PLAN_LOCK.json'),weights_unchanged=True));print('sealed128',flush=True)
 finally:
  put(D/f'COST_{time.time_ns()}.json',dict(GPU_seconds=time.perf_counter()-beg+3,temporal_offset_decodes=0,temporal_classifier_forwards=calls,equivalent_native_classifier_checks=2*len(files),new_Source_encoder=0,new_DINO=0,new_ReferDINO=0,network_updates=0,spatial_updates=0));lock.close()

if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException as e:
  put(D/f'FAILURE_{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e),partials_preserved=True));raise
