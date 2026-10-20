"""GT-before execution revision: floating tolerance for mathematically identical first steps.
No loss, coefficient, sampling, optimizer or final readout changes.
"""
from scripts.run_stvg_temporal_preference_kl_strength128_v1 import *
def measure():
 import numpy as np,torch
 from main_method.runtime import resource_guard
 from methods.stvg_referdino_opd_v1.runtime import guard
 from vg_tta.stvg_temporal_preference_kl_strength128_v1 import fit,loss_gradient
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
 sys.addaudithook(guard);sys.addaudithook(forbid_GT);torch.set_num_threads(4);p=read(D/'PLAN_LOCK.json');beg=time.perf_counter();files=[];fits=updates=first=controls=0;times=[];success=False;costs={m:0. for m in COEFFICIENTS};partial=None;first_gradient_max=first_state_max=0.
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
     for k in ['pairs','uniforms']:assert np.array_equal(out['trace'][0][k],ref[k]),(a,c,s,k)
     ge=float(abs(out['trace'][0]['gradient']-ref['gradient']).max());se=float(abs(out['trace'][0]['eta_after']-ref['eta_after']).max())
     assert ge<1e-14 and se<1e-8,(a,c,s,ge,se);first_gradient_max=max(first_gradient_max,ge);first_state_max=max(first_state_max,se)
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
  put(D/'COST_REVISION001.json',dict(success=success,CPU_prediction_and_updates_seconds=time.perf_counter()-beg,actual_fits=fits,actual_updates=updates,coefficient_fit_CPU_seconds=costs,
   scheduled_fits=768,scheduled_updates=30720,reused_coefficient1_fits=384,first_step_equals_saved_coefficient1=first,coefficient1_nonzero_state_objective_exact=controls,first_gradient_maxerror=first_gradient_max,first_state_maxerror=first_state_max,initial_failed_attempt_cost=read(D/'COST.json'),
   GPU_seconds=0,new_RGB=0,new_model_forward=0,new_DINO=0,new_ReferDINO=0,new_network_training=0,spatial_updates=0,upstream_cache_cost_not_free=True))

if __name__=='__main__':
 def timeout(signum,frame):raise TimeoutError('finite900s KL strength OPD revised stage')
 signal.signal(signal.SIGALRM,timeout);signal.signal(signal.SIGTERM,timeout);signal.alarm(900);lock=(D/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  assert all(sha(R/f)==h for f,h in read(D/'EXECUTION_REVISION001_LOCK.json')['pins'].items());measure()
 except BaseException as e:put(D/f'REVISION001_FAILURE_{time.time_ns()}.json',dict(type=type(e).__name__,message=str(e)));raise
 finally:lock.close()
