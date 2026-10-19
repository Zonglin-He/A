"""Finite final temperature development; private cache input, no scoring imports."""
from pathlib import Path
import sys,time,gc,json,hashlib,os
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=R/'artifacts/stvg_referdino_finalize_v1';P=R/'results/stvg_error_state/2026-10-19/referdino_final';PRE=R/'artifacts/stvg_referdino_condition_hc128_v1';OLD=R/'artifacts/stvg_referdino_dualscale_hc128_v1'
def prepare():
 assert not D.exists();D.mkdir();P.mkdir(parents=True)
 import random,torch
 old=read(PRE/'PLAN_LOCK.json');rows=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];exclude={r['parent']for r in old['records']+old['confirmation_reserved']}
 for r in old.get('engineering',[]):
  if isinstance(r,dict)and'parent'in r:exclude.add(r['parent'])
 # Exclude all old small development cases known from locked fine/ROI rosters.
 for f in [R/'artifacts/stvg_fine_feature32_v1/PLAN_LOCK.json',R/'artifacts/stvg_roi_layout_opd32_v1/PLAN_LOCK.json']:
  if f.exists():
   z=read(f)
   for rr in z.get('records',z.get('rows',[])):
    if rr.get('dataset')=='hc2':exclude.add(rr.get('parent',rows[rr['query_ordinal']]['source']if'query_ordinal'in rr else None))
 groups={}
 for r in rows:
  if r['source']not in exclude:groups.setdefault(r['source'],[]).append(r)
 keys=sorted(groups);random.Random(2026101907).shuffle(keys);assert len(keys)>=32,len(keys)
 formal=[]
 for i,key in enumerate(keys[:32]):
  raw=min(groups[key],key=lambda x:x['ordinal']);q=raw['ordinal'];prior=R/f'artifacts/stvg_opd_paper_hc2_revision_v2/inputs/vidstg_to_hc2/clean/{q:05}.pt';x=torch.load(prior,map_location='cpu',weights_only=False);pos=x['expert']['positions4'];from methods.stvg_referdino_opd_v1.experts import positions8
  formal.append(dict(alias=f'RF{i+1:03}',dataset='hc2',parent=key,query_ordinal=q,positions4=pos,positions8=positions8(raw['frame_ids'],pos),prior_input=str(prior.relative_to(R)),prior_sha256=sha(prior),development_overlap=False,historical_exposure='earlier full-cohort Source/old method evaluated; not globally untouched'))
 files=list((R/'methods/stvg_referdino_opd_v1').glob('*.py'))+[Path(__file__),R/'protocols/stvg_referdino_finalize_v1.md']
 plan=dict(status='locked_before_new_predictions_or_GT',records=old['records'],formal_records=formal,seeds=[41,42,43],temperatures=[.05,.1,.2],new_temperatures=[.1,.2],configuration=dict(scales=[.05,.25],tau=.05,lr=.01,steps=40,samples=32),selection='highest full-vIoU averaged across all three seeds within parent then 128 parents; tie 1e-12 retains .05 then smaller tau; if no improvement retain .05; no further search',formal_count=32,formal_parent_disjoint_from_RC128_RH32=True,formal_history='prior full-cohort Source/old method exposure; no new temperature selection on RF32',new_GPU_cap=3600,chunk_GPU_cap=900,deadline_epoch=time.time()+24000,old_tau05_results_sha256=sha(R/'results/stvg_error_state/2026-10-19/referdino_dualscale_hc128/RESULTS_PUBLIC.json'),old_prediction_barrier_sha256=sha(OLD/'GLOBAL_PREDICTION_BARRIER.json'),pins={str(f.relative_to(R)):sha(f)for f in files},GT_read=False)
 put(D/'PLAN_LOCK.json',plan);put(D/'STATUS.json',dict(stage='prepared',done=0,total=128),True)
 put(P/'PROTOCOL_PUBLIC.json',{k:v for k,v in plan.items()if k not in ['records','formal_records','pins']});put(P/'FORMAL_ROSTER_PUBLIC.json',dict(count=32,roster_sha256=hashlib.sha256(json.dumps(formal,sort_keys=True).encode()).hexdigest(),rows=[{k:v for k,v in r.items()if k not in ['parent','query_ordinal','prior_input']}for r in formal],locked_before_temperature_GT=True))
 print('LOCKED tau .05/.10/.20;128dev;32 prelocked formal;seeds41/42/43')
def measure(chunk,arm='Live'):
 import torch
 from scripts.run_decota_paper_main_v1 import gpu,model_for
 from methods.stvg_referdino_opd_v1.policy import Replay,fit
 from methods.decota_final_simplified_v1.tensors import state_hash
 from main_method.runtime import resource_guard
 p=read(D/'PLAN_LOCK.json');assert all(sha(R/f)==h for f,h in p['pins'].items());lease=gpu();start=time.perf_counter();complete=False;files=[]
 used=sum(read(f)['GPU_seconds']for f in (D/'cost').glob('*.json'))
 def check():
  resource_guard();assert time.time()<p['deadline_epoch'];assert time.perf_counter()-start+3<900;assert used+time.perf_counter()-start+3<p['new_GPU_cap'];assert torch.cuda.mem_get_info()[0]>4*2**30
 try:
  model=model_for('vidstg');mh=state_hash(model.state_dict())
  for j,r in enumerate(p['records'][chunk*16:(chunk+1)*16]):
   check();a=r['alias'];z=load(PRE/'inputs'/f'{a}.pt');target=load(PRE/'teacher0'/f'{a}.pt')['native']['boxes'];rp=Replay(model,z,z['Source'],z['E'],r['positions8'],check);fits={};outputs=dict(E_MBR=z['E'],Source_MBR=z['Source'].numpy());temps=p['new_temperatures']if arm=='Live'else[read(D/'SELECTION_LOCK.json')['tau']]
   for tau in temps:
    for seed in p['seeds']:
     cfg=dict(p['configuration'],tau=tau);name=f'T{int(tau*100):02}_{arm}_{seed}';v=fit(rp,target,a,arm,seed,cfg);fits[name]=v;outputs[name]=v['final'].numpy()
     if chunk==0 and j==0 and arm=='Live' and tau==temps[0]and seed==41:
      from vg_tta.stvg_referdino_dualscale_hc128_v1 import fit as legacy
      exact=legacy(rp,target,a,arm,seed,cfg)
      assert torch.equal(v['path'][0]['boxes'],exact['path'][0]['boxes'])and torch.equal(v['path'][1]['state'],exact['path'][1]['state'])and torch.equal(v['final'],exact['final'])
      old=load(OLD/'predictions'/f'{a}.pt')['fits']['N005F025_Live_41'];rep=fit(rp,target,a,'Live',41,p['configuration']);assert torch.equal(rep['final'],old['final'])and torch.equal(rep['path'][1]['state'],old['path'][1]['state'])
      put(P/'EXTRACTION_REGRESSION.json',dict(status='pass',zero_E_exact=True,first_update_exact=True,final_trajectory_exact=True,legacy_tau05_exact=True,first_parent=a,new_tau=tau,seeds_not_independent_parents=True))
   files.append(save(D/arm/'predictions'/f'{a}.pt',dict(alias=a,dataset='hc2',query_ordinal=r['query_ordinal'],frame_ids=z['frame_ids'],interval=z['interval'],planned_positions=r['positions8'],outputs=outputs,fits=fits,GT_read=False,compute=dict(trajectory_decoder_calls=rp.calls,offset_decoder_calls=2*rp.calls,actual_updates=rp.backwards,new_Source=0,new_DINO=0,new_ReferDINO=0))))
   put(D/'STATUS.json',dict(stage=arm,chunk=chunk,done=chunk*16+j+1,total=128,GT_read=False),True);print('sealed',arm,a,round(time.perf_counter()-start,1),flush=True);del rp,z,fits,outputs;gc.collect();torch.cuda.empty_cache()
  assert state_hash(model.state_dict())==mh;put(D/arm/f'CHUNK_{chunk}_BARRIER.json',dict(files=files,count=16,GT_read=False));complete=True
  if all((D/arm/f'CHUNK_{i}_BARRIER.json').exists()for i in range(8)):put(D/arm/'GLOBAL_PREDICTION_BARRIER.json',dict(files=[x for i in range(8)for x in read(D/arm/f'CHUNK_{i}_BARRIER.json')['files']],count=128,all_before_new_GT=True))
 finally:put(D/'cost'/f'{arm}_{chunk}_{time.time_ns()}.json',dict(GPU_seconds=time.perf_counter()-start+3,complete=complete,count=len(files)));lease.close()
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','measure']);p.add_argument('--chunk',type=int);p.add_argument('--arm',default='Live');a=p.parse_args()
 try:prepare()if a.stage=='prepare'else measure(a.chunk,a.arm)
 except BaseException as e:
  put(D/f'FAILURE_{time.time_ns()}.json',dict(error=str(e),type=type(e).__name__,stage=a.stage));raise
