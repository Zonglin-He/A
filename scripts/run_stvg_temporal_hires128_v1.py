"""Bounded high-resolution frozen-Source probe, with exact original control."""
from pathlib import Path
import sys,os,time,gc
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=R/'artifacts/stvg_temporal_hires128_v1';P=R/'results/stvg_error_state/2026-10-20/temporal_hires128';OLD=R/'artifacts/stvg_temporal_object_memory128_v1';PRE=R/'artifacts/stvg_referdino_condition_hc128_v1';FIN=R/'artifacts/stvg_referdino_finalize_v1';DEV=R/'artifacts/stvg_temporal_increment128_v1';FG=R/'artifacts/stvg_temporal_foreground128_v1'
ARMS=['Native','HiRes','HiResMix','RootStart','RootEnd','StudentStartEnd','NativeFull','NativeFluct','HiResFull','HiResFluct']

def prepare():
 from vg_tta.stvg_temporal_hires128_v1 import properties
 records=[]
 for r in read(OLD/'PLAN_LOCK.json')['records']:records.append(dict(r,prior_sha256=sha(DEV/'priors'/f"{r['alias']}.pt"),native_foreground_sha256=sha(FG/'predictions'/f"{r['alias']}.pt")))
 own=['protocols/stvg_temporal_hires128_v1.md','vg_tta/stvg_temporal_hires128_v1.py','vg_tta/stvg_temporal_foreground128_v1.py','scripts/run_stvg_temporal_hires128_v1.py','scripts/score_stvg_temporal_hires128_v1.py','scripts/publish_stvg_temporal_hires128_v1.py','methods/decota_final_simplified_v1/backbone.py','methods/stvg_referdino_opd_v1/_source_forward.py','methods/stvg_referdino_opd_v1/config.json']
 p=dict(records=records,count=128,seeds=[41,42,43],arms=ARMS,eligible=['HiRes','HiResMix','RootStart','RootEnd','StudentStartEnd','HiResFull','HiResFluct'],primary='HiRes',input_height=448,GPU_seconds_cap=1800,process_seconds_cap=900,deadline_epoch=time.time()+21600,created_epoch=time.time(),pins={f:sha(R/f)for f in own},new_experts=0,new_training=0,spatial_frozen=True,not_OPD=True)
 put(D/'PLAN_LOCK.json',p);put(P/'PROTOCOL_PUBLIC.json',{k:v for k,v in p.items()if k not in ['records','deadline_epoch','created_epoch']});put(P/'PROPERTIES_PUBLIC.json',properties());print('locked128',flush=True)

def measure():
 import torch,numpy as np
 from methods.stvg_referdino_opd_v1.runtime import setup,lease,load_source,decode
 from methods.stvg_referdino_opd_v1._source_forward import frozen_forward
 from methods.decota_final_simplified_v1.tensors import state_hash
 from methods.decota_final_simplified_v1 import backbone
 from main_method.runtime import resource_guard
 from vg_tta.stvg_temporal_hires128_v1 import resolution,hybrid,student_start,properties
 from vg_tta.stvg_temporal_foreground128_v1 import project
 from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,mass_table
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
 setup();assert properties()['status']=='pass';p=read(D/'PLAN_LOCK.json');assert all(sha(R/f)==h for f,h in p['pins'].items());raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];lock=lease();beg=time.perf_counter();prev=[read(f)for f in (D/'cost').glob('*.json')];previous=sum(x['GPU_seconds']for x in prev);high_calls=low_calls=0;files=[];times=[]
 def check():
  resource_guard();assert time.time()<p['deadline_epoch']and previous+time.perf_counter()-beg+60<p['GPU_seconds_cap']and time.perf_counter()-beg+60<p['process_seconds_cap']
 try:
  check();free,total=torch.cuda.mem_get_info();fraction=min(.8,(free-5*2**30)/total);assert fraction>.4;torch.cuda.set_per_process_memory_fraction(fraction);m=load_source();mh=state_hash(m.state_dict());loading=time.perf_counter()-beg;native_once_seconds=0.
  for r in p['records']:
   check();a=r['alias'];dest=D/'predictions'/f'{a}.pt'
   if dest.exists():load(dest);files.append(read(dest.with_suffix('.receipt.json')));continue
   tick=time.perf_counter();old=load(DEV/'priors'/f'{a}.pt');z=load(PRE/'inputs'/f'{a}.pt');sp=load(FIN/'Live/predictions'/f'{a}.pt');fg=load(FG/'predictions'/f'{a}.pt');assert sha(DEV/'priors'/f'{a}.pt')==r['prior_sha256']and sha(FG/'predictions'/f'{a}.pt')==r['native_foreground_sha256']and sha(PRE/'inputs'/f'{a}.pt')==r['input_sha256']and sha(FIN/'Live/predictions'/f'{a}.pt')==r['spatial_sha256'];row=dict(raw[r['query_ordinal']]);row['parses']=read(R/r['subject_path'])['parses'];frames,ids=decode(row['input']);assert ids==old['ids']
   if not (D/'NATIVE_CONTROL.json').exists():
    control=time.perf_counter();original=backbone.make_batch;b0=original(frames,ids,row['input'],m)
    with resolution(224):b1=backbone.make_batch(frames,ids,row['input'],m)
    assert torch.equal(b0['videos'].tensors,b1['videos'].tensors)and backbone.make_batch is original;del b0,b1;gc.collect();torch.cuda.empty_cache();low_calls+=1;batch,recs,base=frozen_forward(m,frames,row);assert torch.equal(base.zero['boxes'].cpu(),z['Source'])and all(torch.equal(x.cpu(),y)for x,y in zip(base.zero['logits'],old['logits']));_,qq=mass_table(probabilities(recs,[x.cpu()for x in base.zero['logits']]));assert np.array_equal(qq,old['q']);base.restore(base.initial);del batch,base;gc.collect();torch.cuda.empty_cache();native_once_seconds=time.perf_counter()-control;put(D/'NATIVE_CONTROL.json',dict(status='pass',alias=a,batch224_exact=True,Source_boxes_logits_q_exact=True,seconds=native_once_seconds,GT_read=False));tick+=native_once_seconds
   check();assert sum(x['highres_Source_attempts']for x in prev)+high_calls<128;torch.cuda.reset_peak_memory_stats();original=backbone.make_batch;high_calls+=1
   with resolution(p['input_height']):batch,recs,base=frozen_forward(m,frames,row)
   assert backbone.make_batch is original;actual_shape=list(batch['videos'].tensors.shape);assert actual_shape[-2]==448;ll=[x.detach().cpu()for x in base.zero['logits']];actions=[x.detach().cpu()for x in base.zero['actions']];grid,qh=mass_table(probabilities(recs,ll));assert grid==ids;qs=dict(Native=old['q'],HiRes=qh,**hybrid(old['q'],qh));qs['StudentStartEnd'],prefix_meta=student_start(old['q'],qh,ids,z['interval']);outputs={k:dict(interval=final_mbr(ids,q)['interval'],q=q)for k,q in qs.items()};assert outputs['Native']['interval']==z['interval'];assert not prefix_meta['available']or outputs['StudentStartEnd']['interval'][0]==z['interval'][0];proj,pa=project(ll,actions,recs)
   for n in ['Full','Fluct']:outputs['HiRes'+n]=proj[n];outputs['Native'+n]=fg['outputs']['Native'+n]
   shape=[v['info']['fea_map_size']for v in base.views];peak=torch.cuda.max_memory_allocated();base.restore(base.initial);del batch,base;gc.collect();torch.cuda.empty_cache();trajectories=dict(E=np.asarray(z['E']),**{f'Live{s}':np.asarray(sp['outputs'][f'T20_Live_{s}'])for s in p['seeds']});files.append(save(dest,dict(alias=a,query_ordinal=r['query_ordinal'],frame_ids=ids,outputs=outputs,trajectories=trajectories,observation=dict(logits=ll,actions=actions,q=qh,records=recs,normalized_tensor_shape=actual_shape,raw_RGB_shape=list(frames.shape),feature_shapes=shape,peak_GPU_allocated=peak,RGB_sha256=__import__('hashlib').sha256(frames.tobytes()).hexdigest()),projection_audit=pa,student_prefix=prefix_meta,GT_read=False)));times.append(time.perf_counter()-tick);put(D/'STATUS.json',dict(stage='measure',done=len(files),pid=os.getpid(),epoch=time.time()),True);print('hires',a,round(time.perf_counter()-beg,2),actual_shape[-2:],flush=True)
   if len(times)==2 and not(D/'PILOT2.json').exists():
    estimate=previous+loading+native_once_seconds+max(times)*128+30;put(D/'PILOT2.json',dict(case_seconds=times,model_loading=loading,native_control_seconds=native_once_seconds,estimated_GPU_seconds=estimate,allocator_fraction=fraction,GT_read=False));assert estimate<p['GPU_seconds_cap']
   del frames,z,sp,fg,old,outputs,qs,qh;gc.collect();torch.cuda.empty_cache()
  assert state_hash(m.state_dict())==mh and all(not x.requires_grad and x.grad is None for x in m.parameters());put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(count=128,files=files,plan_sha256=sha(D/'PLAN_LOCK.json'),all_before_GT=True,weights_unchanged=True));print('sealed128',flush=True)
 finally:
  now=dict(GPU_seconds=time.perf_counter()-beg+3,highres_Source_attempts=high_calls,native224_Source_attempts=low_calls,Source_offset_forwards=2*(high_calls+low_calls),new_DINO=0,new_ReferDINO=0,network_training=0,spatial_updates=0);put(D/'cost'/f'attempt_{time.time_ns()}.json',now);put(D/'COST.json',dict(GPU_seconds=previous+now['GPU_seconds'],attempts=prev+[now],highres_Source_attempts=sum(x['highres_Source_attempts']for x in prev)+high_calls,native224_Source_attempts=sum(x['native224_Source_attempts']for x in prev)+low_calls),True);lock.close()

if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException as e:
  put(D/f'FAILURE_{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e),partials_preserved=True));raise
