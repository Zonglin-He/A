"""Finite coefficient-only Live OPD: full128, no new model or expert calls."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='4'
from pathlib import Path
import sys,time,signal,fcntl,gc
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
from scripts.run_stvg_temporal_strong_relative_preference_opd128_v1 import forbid_GT,SEEDS
D=R/'artifacts/stvg_temporal_preference_kl_strength128_v1';P=R/'results/stvg_error_state/2026-10-20/temporal_preference_kl_strength128'
OLD=R/'artifacts/stvg_temporal_strong_relative_preference_opd128_v1';OP=R/'results/stvg_error_state/2026-10-20/temporal_strong_relative_preference_opd128'
COEFFICIENTS={'KL0p1':.1,'KL0p3':.3};ALL=['KL0p1','KL0p3','KL1'];FITS=[m+'Live'+str(s)for m in COEFFICIENTS for s in SEEDS]
ARMS=['Strong']+[m+'Live'+str(s)for m in ALL for s in SEEDS]

def properties():
 import numpy as np
 from vg_tta.stvg_temporal_preference_kl_strength128_v1 import feedback,loss_gradient,manual,fit
 from vg_tta.stvg_temporal_strong_relative_preference_opd128_v1 import loss_gradient as oldloss
 B=np.array([.2,.3,.5]);r=np.array([.1,-.05,.02]);ij=np.array([[0,1],[1,2],[0,0]]);lb=np.log(B)
 ev=dict(mapping=np.array([[0,1,2],[0,1,2]]),logD=[np.log([.1,.3,.6])]*2,logB=[lb,lb],support=np.argwhere(np.triu(np.ones((3,3)),1)))
 fb=feedback(ev,ij);base=oldloss(B,r,ij,fb);same=loss_gradient(B,r,ij,fb,1.)
 for k in base:assert np.array_equal(base[k],same[k]),k
 errors={};grid=np.zeros((3,3));grid[tuple(ev['support'].T)]=B
 empty=dict(ev,logD=ev['logB'])
 for c in [.1,.3,1.]:
  calc=loss_gradient(B,r,ij,fb,c);g=calc['gradient'];assert max(abs(g-manual(B,r,ij,fb,c)))<1e-12
  num=[]
  for i in range(3):
   d=np.eye(3)[i]*1e-6;num.append((loss_gradient(B,r+d,ij,fb,c)['loss']-loss_gradient(B,r-d,ij,fb,c)['loss'])/2e-6)
  errors[str(c)]=float(max(abs(g-num)));assert errors[str(c)]<1e-9
  z=fit(grid,empty,'Live',41,kl_coefficient=c);assert np.array_equal(z['q'],grid)and not z['eta'].any()
 return dict(status='pass',coefficient1_saved_objective_bit_exact=True,finite_difference_error=errors,zero_feedback_all_coefficients_exact=True)

def prepare():
 import torch
 from main_method.runtime import resource_guard
 torch.set_num_threads(4);sys.addaudithook(forbid_GT);resource_guard();old=read(OLD/'PLAN_LOCK.json');bar=read(OLD/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==128
 assert read(OP/'ROOT_AUDIT.json')['status']=='pass';assert all(sha(R/f)==h for f,h in old['pins'].items())
 own=['protocols/stvg_temporal_preference_kl_strength128_v1.md','vg_tta/stvg_temporal_preference_kl_strength128_v1.py',
 'scripts/run_stvg_temporal_preference_kl_strength128_v1.py','scripts/score_stvg_temporal_preference_kl_strength128_v1.py','scripts/publish_stvg_temporal_preference_kl_strength128_v1.py',
 'scripts/score_stvg_temporal_strong_relative_preference_opd128_audit_revision001.py']
 pins=dict(old['pins']);pins.update({f:sha(R/f)for f in own})
 for f in [OLD/'PLAN_LOCK.json',OLD/'GLOBAL_PREDICTION_BARRIER.json',OLD/'COST.json',OP/'RESULTS_PUBLIC.json',OP/'ROOT_AUDIT.json',OP/'ROOT_SAVED_SCORE_REAGGREGATION_PUBLIC.json']:
  pins[str(f.relative_to(R))]=sha(f)
 receipts={r['path']:r['sha256']for r in bar['files']};records=[]
 for r in old['records']:
  f=OLD/'predictions'/f"{r['alias']}.pt";rel=str(f.relative_to(R));assert sha(f)==receipts[rel];pins[rel]=receipts[rel];records.append(dict(r,prior_preference_sha256=receipts[rel]))
 p=dict(records=records,count=128,seeds=SEEDS,new_coefficients=COEFFICIENTS,reused_coefficient=1.,arms=ARMS,steps=40,lr=.1,pairs_per_step=32,
 scheduled_fits=768,scheduled_updates=30720,only_change='global KL coefficient',new_mode='Live only; old Direct/Fixed not relabeled matched new controls',
 selection='max parent-mean complete vIoU, ties stronger protection1 then.3 then.1; development only',
 source_checkpoint=old['source_checkpoint'],source_checkpoint_sha256=old['source_checkpoint_sha256'],pins=pins,CPU_seconds_cap=900,deadline_epoch=time.time()+14400,
 new_GPU=0,new_visual_calls=0,new_expert=0,new_network_training=0,spatial_frozen=True,not_confirmation=True,
 no_coefficients_beyond_predeclared_three=True,no_automatic_next_experiment=True,created_epoch=time.time())
 put(D/'PLAN_LOCK.json',p);put(P/'PROTOCOL_PUBLIC.json',{k:v for k,v in p.items()if k not in ['records','created_epoch','deadline_epoch']});put(P/'PROPERTIES_PUBLIC.json',properties());print('locked two new full128 Live KL coefficients;1 reused',flush=True)

def measure():
 import numpy as np,torch
 from main_method.runtime import resource_guard
 from methods.stvg_referdino_opd_v1.runtime import guard
 from vg_tta.stvg_temporal_preference_kl_strength128_v1 import fit,loss_gradient
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
 sys.addaudithook(guard);sys.addaudithook(forbid_GT);torch.set_num_threads(4);p=read(D/'PLAN_LOCK.json');beg=time.perf_counter();files=[];fits=updates=first=controls=0;times=[];success=False;costs={m:0. for m in COEFFICIENTS};partial=None
 def check():resource_guard();assert time.perf_counter()-beg<880 and time.time()<p['deadline_epoch']
 def pins():assert all(sha(R/f)==h for f,h in p['pins'].items())
 pins();loading=time.perf_counter()-beg
 try:
  for rec in p['records']:
   check();tick=time.perf_counter();a=rec['alias'];old=load(OLD/'predictions'/f'{a}.pt');B=old['q0'];ev=old['evidence']
   outputs={'Strong':dict(q=B,interval=old['outputs']['Strong']['interval'])}
   for s in SEEDS:
    out=old['outputs']['Live'+str(s)];outputs['KL1Live'+str(s)]=dict(q=out['q'],interval=out['interval'],reused=True)
    tr=out['trace'][-1];calc=loss_gradient(B[tuple(ev['support'].T)],tr['eta_before'],tr['pairs'],tr['feedback'],1.)
    for k in calc:assert np.array_equal(calc[k],tr[k]),(a,s,k)
    controls+=1
   partial=dict(alias=a,q0=B,evidence=ev,outputs=outputs,GT_read=False)
   for name,c in COEFFICIENTS.items():
    for s in SEEDS:
     check();t=time.perf_counter();out=fit(B,ev,'Live',s,steps=40,lr=.1,check=check,kl_coefficient=c);costs[name]+=time.perf_counter()-t;out['interval']=final_mbr(old['frame_ids'],out['q'])['interval'];outputs[name+'Live'+str(s)]=out;fits+=1;updates+=out['actual_updates']
     ref=old['outputs']['Live'+str(s)]['trace'][0]
     for k in ['pairs','uniforms','gradient','eta_after']:assert np.array_equal(out['trace'][0][k],ref[k]),(a,c,s,k)
     first+=1
   assert set(outputs)==set(ARMS);files.append(save(D/'predictions'/f'{a}.pt',dict(alias=a,frame_ids=old['frame_ids'],trajectories=old['trajectories'],q0=B,evidence=ev,outputs=outputs,GT_read=False)))
   times.append(time.perf_counter()-tick);put(D/'STATUS.json',dict(stage='actual_Live_OPD',done=len(files),fits=fits,updates=updates,pid=os.getpid(),epoch=time.time()),True);print('KL_strength_OPD',a,round(time.perf_counter()-beg,2),flush=True)
   if len(files)==2:
    estimate=loading+128*max(times)+30;put(D/'PILOT2.json',dict(case_seconds=times,estimated_CPU_seconds=estimate,engineering_only=True,new_GT=False));assert estimate<900
   partial=None;del old,outputs,ev;gc.collect()
  check();pins();assert fits==768 and updates==30720 and first==768 and controls==384
  put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(count=128,files=files,all_before_GT=True,plan_sha256=sha(D/'PLAN_LOCK.json'),first_step_equals_saved_coefficient1=first,coefficient1_nonzero_state_objective_exact=controls));success=True
 finally:
  if not success and partial is not None:save(D/'partials'/f"{partial['alias']}_{time.time_ns()}.pt",partial)
  put(D/'COST.json',dict(success=success,CPU_prediction_and_updates_seconds=time.perf_counter()-beg,actual_fits=fits,actual_updates=updates,coefficient_fit_CPU_seconds=costs,
   scheduled_fits=768,scheduled_updates=30720,reused_coefficient1_fits=384,first_step_equals_saved_coefficient1=first,coefficient1_nonzero_state_objective_exact=controls,
   GPU_seconds=0,new_RGB=0,new_model_forward=0,new_DINO=0,new_ReferDINO=0,new_network_training=0,spatial_updates=0,upstream_cache_cost_not_free=True))
if __name__=='__main__':
 def timeout(signum,frame):raise TimeoutError('finite900s KL strength OPD stage')
 signal.signal(signal.SIGALRM,timeout);signal.signal(signal.SIGTERM,timeout);signal.alarm(900);D.mkdir(parents=True,exist_ok=True);lock=(D/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:globals()[sys.argv[1]]()
 except BaseException as e:put(D/f'FAILURE_{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e),partials_preserved=True));raise
 finally:lock.close()
