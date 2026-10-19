"""Finite frozen temporal-memory probe; all predictions before GT, no experts."""
from pathlib import Path
import sys,time,gc,os
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=R/'artifacts/stvg_temporal_object_memory128_v1';P=R/'results/stvg_error_state/2026-10-19/temporal_object_memory128'
PRE=R/'artifacts/stvg_referdino_condition_hc128_v1';FIN=R/'artifacts/stvg_referdino_finalize_v1';DEV=R/'artifacts/stvg_temporal_increment128_v1'

def prepare():
 from vg_tta.stvg_temporal_object_memory128_v1 import properties
 assert not(D/'PLAN_LOCK.json').exists();D.mkdir(exist_ok=True);P.mkdir(parents=True,exist_ok=True)
 records=read(DEV/'PLAN_LOCK.json')['records'];rs=[]
 for r in records:
  a=r['alias'];rs.append(dict(r,teacher_sha256=sha(PRE/'teacher0'/f'{a}.pt'),spatial_sha256=sha(FIN/'Live/predictions'/f'{a}.pt'),input_sha256=sha(PRE/'inputs'/f'{a}.pt'),prior_sha256=sha(DEV/'priors'/f'{a}.pt')))
 code=['scripts/run_stvg_temporal_object_memory128_v1.py','vg_tta/stvg_temporal_object_memory128_v1.py','protocols/stvg_temporal_object_memory128_v1.md','methods/stvg_referdino_opd_v1/config.json','methods/stvg_referdino_opd_v1/_source_forward.py','external/TA-STVG/models/grounding_model/query_decoder.py']
 p=dict(records=rs,count=128,seeds=[41,42,43],arms=['Native','E_ROI','Live41_ROI','Live42_ROI','Live43_ROI','ReferROI','ShiftedReferROI'],primary='ReferROI minus Native, correct vsShifted for grounding dependence',fixed_prior_context_mass=.5,new_experts=0,new_training=0,spatial_branch_frozen=True,primary_spaces='saved Live41/42/43; metric seed average within parent',pins={f:sha(R/f)for f in code},GPU_seconds_cap=900,deadline_epoch=time.time()+14400,created_epoch=time.time(),GT_read=False)
 put(D/'PLAN_LOCK.json',p);put(D/'PROPERTIES.json',properties());put(P/'PROTOCOL_PUBLIC.json',{k:v for k,v in p.items()if k not in ['records','pins']});status('prepared',0)

def status(stage,done,**kw):put(D/'STATUS.json',dict(stage=stage,done=done,total=128,pid=os.getpid(),epoch=time.time(),**kw),True)
def verify():
 from main_method.runtime import resource_guard
 p=read(D/'PLAN_LOCK.json');assert time.time()<p['deadline_epoch'];assert all(sha(R/f)==h for f,h in p['pins'].items());resource_guard();return p

def measure():
 import torch,numpy as np
 from methods.stvg_referdino_opd_v1.runtime import setup,lease,load_source,decode
 from methods.stvg_referdino_opd_v1._source_forward import frozen_forward
 from methods.stvg_referdino_opd_v1._capture import transfer
 from methods.decota_final_simplified_v1.tensors import state_hash,detached
 from vg_tta.stvg_referdino_native32_v1 import interpolate
 from vg_tta.stvg_temporal_object_memory128_v1 import replay
 from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,mass_table
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
 setup();p=verify();raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];lock=lease();start=time.perf_counter();files=[];calls=0;captures=0;readouts=0;previous_costs=[read(x)for x in (D/'cost').glob('*.json')];previous_GPU=sum(x['GPU_seconds']for x in previous_costs)
 def check():
  verify();assert previous_GPU+time.perf_counter()-start+8<p['GPU_seconds_cap'];assert torch.cuda.mem_get_info()[0]>4*2**30
 try:
  model=load_source();mh=state_hash(model.state_dict());load_seconds=time.perf_counter()-start;pilot_case_seconds=[]
  for r in p['records']:
   check();a=r['alias'];t=time.perf_counter();assert sha(PRE/'inputs'/f'{a}.pt')==r['input_sha256']and sha(FIN/'Live/predictions'/f'{a}.pt')==r['spatial_sha256'];z=load(PRE/'inputs'/f'{a}.pt');s=load(FIN/'Live/predictions'/f'{a}.pt');prior=load(DEV/'priors'/f'{a}.pt');row=dict(raw[r['query_ordinal']]);row['parses']=read(R/r['subject_path'])['parses'];
   if (D/'temporal_inputs'/f'{a}.pt').exists():
    old_cache=load(D/'temporal_inputs'/f'{a}.pt');assert old_cache['Source_exact']and old_cache['native_logits_exact'];inputs=old_cache['inputs'];grids=old_cache['grids'];ids=z['frame_ids'];recs=z['records'];gc.collect();torch.cuda.empty_cache()
   else:
    frames,ids=decode(row['input']);batch,recs,base=frozen_forward(model,frames,row);captures+=1;assert torch.equal(base.zero['boxes'].cpu(),z['Source']);assert all(torch.equal(x.cpu(),y)for x,y in zip(base.zero['logits'],prior['logits']));cap=[]
    def hook(module,args,kw):cap.append(detached(kw,'cpu'))
    hnd=base.decoder.time_decoder.register_forward_pre_hook(hook,with_kwargs=True)
    try:
     with torch.no_grad():native=base.reference_values()
    finally:hnd.remove()
    assert len(cap)==4 and all(torch.equal(x.cpu(),y)for x,y in zip(native['logits'],prior['logits']));inputs=[cap[1],cap[3]];grids=[]
    for v in base.views:
     hh,ww=v['info']['fea_map_size'];grids.append(dict(h=hh,w=ww,valid=(~v['info']['encoded_mask'][:,-hh*ww:]).cpu()))
    save(D/'temporal_inputs'/f'{a}.pt',dict(inputs=inputs,grids=grids,Source_exact=True,native_logits_exact=True,GT_read=False))
    del base,batch,frames,cap,native;gc.collect();torch.cuda.empty_cache()
   teacher=load(PRE/'teacher0'/f'{a}.pt');assert sha(PRE/'teacher0'/f'{a}.pt')==r['teacher_sha256'];ref=interpolate(ids,r['positions8'],teacher['native']['boxes'].numpy());shift=ref.copy();shift[:,0]=(shift[:,0]+.5)%1
   boxes=dict(E_ROI=np.asarray(z['E']),**{f'Live{seed}_ROI':np.asarray(s['outputs'][f'T20_Live_{seed}'])for seed in p['seeds']},ReferROI=ref,ShiftedReferROI=shift);logits={k:[]for k in boxes};meta={k:[]for k in boxes};native_meta=[]
   for view,(kw0,g)in enumerate(zip(inputs,grids)):
    kw=transfer(kw0,'cuda');assert torch.equal(kw['query_tgt'],kw['query_tgt'][:1].expand_as(kw['query_tgt']));kw['query_tgt']=kw['query_tgt'][:1].expand_as(kw['query_tgt']);valid=g['valid'].cuda()
    with torch.no_grad():out=model.temp_embed(model.ground_decoder.time_decoder(**kw))[-1].cpu()
    assert torch.equal(out,prior['logits'][view]);native_meta.append(True)
    for arm,b in boxes.items():
     check();out,info=replay(model,kw,torch.as_tensor(b[view::2],device='cuda',dtype=torch.float64),valid,g['h'],g['w']);logits[arm].append(out);meta[arm].append(info);readouts+=1
    del kw;gc.collect();torch.cuda.empty_cache()
   logits['Native']=prior['logits'];outputs={}
   for arm,ll in logits.items():
    grid,q=mass_table(probabilities(recs,ll));assert grid==ids;outputs[arm]=dict(interval=final_mbr(ids,q)['interval'],q=q,logits=ll,max_logit_change=max(float(abs(x-y).max())for x,y in zip(ll,prior['logits'])))
   assert outputs['Native']['interval']==z['interval']and np.array_equal(outputs['Native']['q'],prior['q']);trajectories=dict(E=np.asarray(z['E']),**{f'Live{seed}':np.asarray(s['outputs'][f'T20_Live_{seed}'])for seed in p['seeds']})
   files.append(save(D/'predictions'/f'{a}.pt',dict(alias=a,query_ordinal=r['query_ordinal'],frame_ids=ids,positions8=r['positions8'],original_interval=z['interval'],trajectories=trajectories,outputs=outputs,prior_meta=meta,Source_exact=True,native_temporal_replay_exact=all(native_meta),compute=dict(seconds=time.perf_counter()-t,new_Source=1,new_experts=0,temporal_readouts=14),GT_read=False)));status('measure',len(files));print('memory',a,round(time.perf_counter()-start,2),flush=True)
   pilot_case_seconds.append(time.perf_counter()-t)
   if len(files)==2:
    est=sum(pilot_case_seconds)/2*128+load_seconds+60+previous_GPU;put(D/'PILOT2.json',dict(zero_Source_and_time_exact=True,primary_response=[outputs['ReferROI']['max_logit_change']],conservative128_seconds=est,GT_read=False));assert est<900,'two-case projected budget exceeds cap'
   del z,s,prior,teacher,logits,meta,outputs,inputs;gc.collect();torch.cuda.empty_cache()
  assert state_hash(model.state_dict())==mh and all(not x.requires_grad and x.grad is None for x in model.parameters());put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(count=128,files=files,plan_sha256=sha(D/'PLAN_LOCK.json'),all_before_new_GT=True,weights_unchanged=True));status('sealed_before_GT',128)
 finally:
  now=dict(GPU_seconds=time.perf_counter()-start+3,new_Source_queries=captures,Source_offset_forwards=2*captures,new_DINO=0,new_ReferDINO=0,temporal_conditioned_offset_decodes=readouts,network_weight_updates=0,spatial_updates=0);put(D/'cost'/f'measure_{time.time_ns()}.json',now);allcost=previous_costs+[now];put(D/'COST.json',dict(GPU_seconds=sum(x['GPU_seconds']for x in allcost),new_Source_queries=sum(x.get('new_Source_queries',0)for x in allcost),Source_offset_forwards=sum(x.get('Source_offset_forwards',0)for x in allcost),new_DINO=0,new_ReferDINO=0,temporal_conditioned_offset_decodes=sum(x.get('temporal_conditioned_offset_decodes',0)for x in allcost),network_weight_updates=0,spatial_updates=0,attempts=allcost));lock.close()

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','measure']);a=ap.parse_args()
 try:globals()[a.stage]()
 except BaseException as e:
  D.mkdir(parents=True,exist_ok=True);put(D/f'FAILURE_{time.time_ns()}.json',dict(stage=a.stage,type=type(e).__name__,error=str(e),partials_preserved=True));raise
