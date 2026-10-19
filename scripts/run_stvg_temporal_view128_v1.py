"""Bounded two-view frozen Source time probe. No new expert or spatial update."""
from pathlib import Path
import sys,time,gc,os
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=R/'artifacts/stvg_temporal_view128_v1';P=R/'results/stvg_error_state/2026-10-19/temporal_view128'
DEV=R/'artifacts/stvg_temporal_increment128_v1';PRE=R/'artifacts/stvg_referdino_condition_hc128_v1';FIN=R/'artifacts/stvg_referdino_finalize_v1'

def prepare():
 from vg_tta.stvg_temporal_view128_v1 import views,properties
 assert not(D/'PLAN_LOCK.json').exists();D.mkdir(exist_ok=True);P.mkdir(parents=True,exist_ok=True);records=[]
 for r in read(DEV/'PLAN_LOCK.json')['records']:
  a=r['alias'];z=load(DEV/'priors'/f'{a}.pt');v,meta=views(z['ids'],z['interval']);records.append(dict(r,observation_grids=v,view_meta=meta,prior_sha256=sha(DEV/'priors'/f'{a}.pt'),input_sha256=sha(PRE/'inputs'/f'{a}.pt'),spatial_sha256=sha(FIN/'Live/predictions'/f'{a}.pt')))
 code=['scripts/run_stvg_temporal_view128_v1.py','vg_tta/stvg_temporal_view128_v1.py','protocols/stvg_temporal_view128_v1.md','methods/stvg_referdino_opd_v1/config.json','methods/stvg_referdino_opd_v1/_source_forward.py','methods/stvg_referdino_opd_v1/runtime.py']
 p=dict(records=records,count=128,seeds=[41,42,43],arms=['Native','CoarseDirect','WindowDirect','CoarseMix','WindowMix'],mixture_weights=[.5,.5],context_expansion=2,common_original_decisions=True,spatial_frozen=True,new_experts=0,new_training=0,new_temporal_OPD=False,created_epoch=time.time(),deadline_epoch=time.time()+14400,GPU_seconds_cap=900,pins={f:sha(R/f)for f in code})
 put(D/'PLAN_LOCK.json',p);put(D/'PROPERTIES.json',properties());put(P/'PROTOCOL_PUBLIC.json',{k:v for k,v in p.items()if k not in ['records','pins']});put(P/'OBSERVATION_LOCK_PUBLIC.json',dict(rows=[dict(alias=r['alias'],**r['view_meta'])for r in records],GT_read=False));status('prepared',0)

def status(stage,done,**kw):put(D/'STATUS.json',dict(stage=stage,done=done,total=128,pid=os.getpid(),epoch=time.time(),**kw),True)
def verify():
 from main_method.runtime import resource_guard
 p=read(D/'PLAN_LOCK.json');assert time.time()<p['deadline_epoch'];assert all(sha(R/f)==h for f,h in p['pins'].items());resource_guard();return p

def measure():
 import torch,numpy as np
 from methods.stvg_referdino_opd_v1.runtime import setup,lease,load_source,decode
 from methods.stvg_referdino_opd_v1._source_forward import frozen_forward
 from methods.decota_final_simplified_v1.tensors import state_hash
 from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,mass_table
 from vg_tta.stvg_temporal_view128_v1 import readout
 from vg_tta.stvg_posterior_decision_tree_v1 import brute_utility
 setup();p=verify();raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];lock=lease();start=time.perf_counter();files=[];newcalls=0;reused=0;costs=[];previous_costs=[read(x)for x in (D/'cost').glob('*.json')];previous_GPU=sum(x['GPU_seconds']for x in previous_costs)
 def check():verify();assert previous_GPU+time.perf_counter()-start+6<p['GPU_seconds_cap']and time.perf_counter()-start+6<p['process_seconds_cap'];assert torch.cuda.mem_get_info()[0]>4*2**30
 try:
  model=load_source();mh=state_hash(model.state_dict());load_s=time.perf_counter()-start
  for r in p['records']:
   check();a=r['alias'];dest=D/'predictions'/f'{a}.pt'
   if dest.exists():load(dest);files.append(read(dest.with_suffix('.receipt.json')));continue
   tick=time.perf_counter();old=load(DEV/'priors'/f'{a}.pt');z=load(PRE/'inputs'/f'{a}.pt');sp=load(FIN/'Live/predictions'/f'{a}.pt');assert sha(DEV/'priors'/f'{a}.pt')==r['prior_sha256']and sha(PRE/'inputs'/f'{a}.pt')==r['input_sha256']and sha(FIN/'Live/predictions'/f'{a}.pt')==r['spatial_sha256'];row=dict(raw[r['query_ordinal']]);row['parses']=read(R/r['subject_path'])['parses'];grids=r['observation_grids'];union=sorted(set(old['ids'])|set(grids['Coarse'])|set(grids['Window']));meta=dict(row['input'],frame_ids=union);rgb,decoded_ids=decode(meta);assert union==decoded_ids;cache={tuple(old['ids']):dict(ids=old['ids'],q=old['q'],records=old['records'],logits=old['logits'],new_Source=False)};obs={};case_calls=0;new_RGB_counts={}
   for name,ids in grids.items():
    key=tuple(ids)
    if key in cache:obs[name]=cache[key];reused+=1;continue
    check();indices=np.searchsorted(union,ids);frames=rgb[indices];rr=dict(row,frame_ids=ids,input=dict(row['input'],frame_ids=ids));batch,recs,base=frozen_forward(model,frames,rr);newcalls+=1;case_calls+=1;ll=[x.detach().cpu()for x in base.zero['logits']];U,q=mass_table(probabilities(recs,ll));assert U==ids;obs[name]=dict(ids=ids,q=q,records=recs,logits=ll,new_Source=True,RGB_sha256=__import__('hashlib').sha256(frames.tobytes()).hexdigest());cache[key]=obs[name];base.restore(base.initial);del batch,base,frames;gc.collect();torch.cuda.empty_cache()
   originals=(old['ids'],old['q']);outputs={};components={}
   outputs['Native']=readout(old['ids'],[(1.,*originals)]);assert outputs['Native']['interval']==old['interval']
   # Same original p0 embedded on all view endpoints is a grid-only numerical control.
   G=sorted(set(union));gq=np.zeros((len(G),len(G)));ix=np.searchsorted(G,old['ids']);gq[np.ix_(ix,ix)]=old['q'];gridonly=readout(old['ids'],[(1.,G,gq)]);assert gridonly['interval']==old['interval']
   for name,v in obs.items():
    outputs[name+'Direct']=readout(old['ids'],[(1.,v['ids'],v['q'])]);outputs[name+'Mix']=readout(old['ids'],[(.5,*originals),(.5,v['ids'],v['q'])])
   for name,o in outputs.items():
    U=np.array(o['hypothesis_ids']);ii,jj=np.triu_indices(len(U),1);chosen=int(np.where((U[ii]==o['interval'][0])&(U[jj]+1==o['interval'][1]))[0][0]);exact=float(brute_utility(U,o['q'],[chosen])[0]);assert abs(exact-o['utility'])<1e-11
   trajectories=dict(E=np.asarray(z['E']),**{f'Live{s}':np.asarray(sp['outputs'][f'T20_Live_{s}'])for s in p['seeds']});files.append(save(D/'predictions'/f'{a}.pt',dict(alias=a,query_ordinal=r['query_ordinal'],frame_ids=old['ids'],original_interval=old['interval'],trajectories=trajectories,outputs=outputs,observations=obs,view_meta=r['view_meta'],grid_only_exact=True,new_Source=case_calls,GT_read=False)));costs.append(dict(alias=a,seconds=time.perf_counter()-tick,new_Source=case_calls));status('measure',len(files));print('view',a,case_calls,round(time.perf_counter()-start,2),flush=True)
   if len(files)==2:put(D/'PILOT2.json',dict(case_seconds=[x['seconds']for x in costs],model_load_seconds=load_s,conservative128_seconds=max(x['seconds']for x in costs)*128+load_s+30,GT_read=False));assert max(x['seconds']for x in costs)*128+load_s+30<900,'projected view budget exceeds cap'
   del rgb,z,sp,old,obs,cache,outputs;gc.collect();torch.cuda.empty_cache()
  assert state_hash(model.state_dict())==mh and all(not x.requires_grad and x.grad is None for x in model.parameters());put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(count=128,files=files,plan_sha256=sha(D/'PLAN_LOCK.json'),all_before_new_GT=True,weights_unchanged=True));status('sealed_before_GT',128)
 finally:
  now=dict(GPU_seconds=time.perf_counter()-start+3,new_Source_view_queries=newcalls,Source_view_offset_forwards=2*newcalls,exact_duplicate_views_reused=reused,new_DINO=0,new_ReferDINO=0,network_weight_updates=0,spatial_updates=0,case_costs=costs);put(D/'cost'/f'attempt_{time.time_ns()}.json',now);allcost=previous_costs+[now];put(D/'COST.json',dict(GPU_seconds=sum(x['GPU_seconds']for x in allcost),new_Source_view_queries=sum(x['new_Source_view_queries']for x in allcost),Source_view_offset_forwards=sum(x['Source_view_offset_forwards']for x in allcost),exact_duplicate_views_reused=sum(x['exact_duplicate_views_reused']for x in allcost),new_DINO=0,new_ReferDINO=0,network_weight_updates=0,spatial_updates=0,case_costs=[x for c in allcost for x in c['case_costs']],attempts=[{k:v for k,v in c.items()if k!='case_costs'}for c in allcost]));lock.close()

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','measure']);a=ap.parse_args()
 try:globals()[a.stage]()
 except BaseException as e:
  D.mkdir(parents=True,exist_ok=True);put(D/f'FAILURE_{time.time_ns()}.json',dict(stage=a.stage,type=type(e).__name__,error=str(e),partials_preserved=True));raise
