"""Native fixed-teacher Direct/Fresh-Fixed/Live HC128, three locked seeds."""
from pathlib import Path
import os,sys,time,gc
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=ROOT/'artifacts/stvg_referdino_dualscale_hc128_v1';P=ROOT/'results/stvg_error_state/2026-10-19/referdino_dualscale_hc128';PRE=ROOT/'artifacts/stvg_referdino_condition_hc128_v1';VER=None
SEEDS=[41,42,43]
def verify(full=False):
 global VER
 from main_method.runtime import PinVerifier,resource_guard
 p=read(D/'PLAN_LOCK.json');l=read(D/'RUNTIME_LOCK.json');assert sha(D/'PLAN_LOCK.json')==l['plan_sha256'];pins=dict(l['pins'])
 for f in sorted((D/'amendments').glob('*.json')):pins.update(read(f)['replacement_pins'])
 if VER is None:VER=PinVerifier(pins)
 VER.verify(full=full);resource_guard();return p

def prepare():
 import torch
 from vg_tta.stvg_referdino_dualscale_hc128_v1 import CONFIGS,properties
 assert not D.exists();D.mkdir();P.mkdir(parents=True)
 old=ROOT/'artifacts/stvg_referdino_native_opd_hc128_v1';assert read(old/'FINAL_COMPLETION.json')['status']=='complete';oldplan=read(old/'PLAN_LOCK.json');rs=oldplan['records'];assert len(rs)==128
 cfg=dict(lr=.01,steps=40,samples=32,tau=.05);l=read(old/'RUNTIME_LOCK.json');pins=dict(l['pins'])
 for f in sorted((old/'amendments').glob('*.json')):pins.update(read(f)['replacement_pins'])
 files=['scripts/score_stvg_referdino_dualscale_hc128_v1.py','scripts/audit_stvg_referdino_dualscale_hc128_v1.py','scripts/report_stvg_referdino_dualscale_hc128_v1.py','vg_tta/stvg_referdino_dualscale_hc128_v1.py','scripts/run_stvg_referdino_dualscale_hc128_v1.py','scripts/continue_stvg_referdino_dualscale_hc128_v1.py','protocols/stvg_referdino_dualscale_hc128_v1.md']
 files +=['results/stvg_error_state/2026-10-19/referdino_native_opd_hc128/COST_BREAKDOWN.json']+[str(f.relative_to(ROOT))for f in old.joinpath('predictions').glob('*.pt')]+[str((ROOT/'results/stvg_error_state/2026-10-19/referdino_native_opd_hc128/RESULTS_PUBLIC.json').relative_to(ROOT))]
 pins.update({f:sha(ROOT/f)for f in files});plan=dict(status='locked_before_new_prediction_GT',records=rs,count=128,seeds=[41,42,43],configuration=cfg,policies={k:list(v)for k,v in CONFIGS.items()},old_results='results/stvg_error_state/2026-10-19/referdino_native_opd_hc128/RESULTS_PUBLIC.json',old_predictions=str(old.relative_to(ROOT)),teacher='same cached native ReferDINO selected8, fixed',policy_coordinates='actual corrected student cx,cy,logw,logh',objective='exact equally weighted mixture NLL, target and actions and covariance stop gradient; NO old Gaussian KL baseline or control variate',Fixed='fresh initial mean AND initial size covariance each round; current student mean receives likelihood gradient',Live='fresh current decoded mean AND current size covariance, detached within each round',selection='rank old Live +3 new Live by mean full-vIoU, tie1e-9 then outside8 s, then measured fit time then name; all seeds pooled as within-parent metric means; retain all tradeoffs/tails',confirmation_reserved=read(PRE/'PLAN_LOCK.json')['confirmation_reserved'],confirmation='only after selection lock, same reserved32; no tuning on these outcomes; retain oldLive Direct E and selected Fixed/Live',total_new_fits=2304,total_configured_updates=92160,GPU_seconds_cap=4800,chunk_GPU_cap=900,chunks=8,chunk_size=16,deadline_epoch=time.time()+18000,created_epoch=time.time(),new_development_Source_or_teacher_calls=0,no_new_experts_or_trained_models=True,no_additional_structure_iteration=True)
 put(D/'PLAN_LOCK.json',plan);put(D/'RUNTIME_LOCK.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=pins));put(D/'INPUT_MANIFEST.json',read(old/'INPUT_MANIFEST.json'));put(P/'CPU_PROPERTIES.json',properties());put(P/'CONFIG_PUBLIC.json',{k:v for k,v in plan.items()if k not in ['records','confirmation_reserved']});put(D/'STATUS.json',dict(status='prepared',done=0,total=128,GT_read=False),True);verify(True);print('LOCKED 3policies 2arms 3seeds 128parents, cache only',flush=True)

def measure(chunk):
 import torch,numpy as np
 from scripts.run_decota_paper_main_v1 import gpu,model_for
 from methods.decota_final_simplified_v1.tensors import state_hash
 from vg_tta.stvg_referdino_dualscale_hc128_v1 import Replay,fit
 p=verify(True);assert 0<=chunk<8;used=sum(read(f)['GPU_seconds']for f in(D/'cost').glob('*.json'));lease=gpu();start=time.perf_counter();files=[];done=False
 def check():
  verify();assert time.time()<p['deadline_epoch'];assert used+time.perf_counter()-start+3<p['GPU_seconds_cap'];assert time.perf_counter()-start+3<p['chunk_GPU_cap'];assert torch.cuda.mem_get_info()[0]>4*2**30
 try:
  model=model_for('vidstg');mh=state_hash(model.state_dict());manifest={r['alias']:r for r in read(D/'INPUT_MANIFEST.json')['rows']}
  for i,r in enumerate(p['records'][chunk*16:(chunk+1)*16]):
   check();a=r['alias'];dest=D/'predictions'/(a+'.pt')
   assert not dest.exists(),'immutable output already exists'
   tick=time.perf_counter();z=load(PRE/'inputs'/(a+'.pt'));teacher=load(PRE/'teacher0'/(a+'.pt'));target=teacher['native']['boxes'];assert __import__('hashlib').sha256(target.numpy().tobytes()).hexdigest()==manifest[a]['target_boxes_sha256'];rp=Replay(model,z,z['Source'],z['E'],r['positions8'],check);fits={};outputs=dict(E_MBR=z['E']);contracts=[]
   for policy,scales in p['policies'].items():
    cfg=dict(p['configuration'],scales=scales)
    for seed in p['seeds']:
     fixed=fit(rp,target,a,'Fixed',seed,cfg);live=fit(rp,target,a,'Live',seed,cfg)
     for key in ['actions','boxes','weights','cost','eps','D']:assert torch.equal(fixed['rounds'][0]['rollout'][key],live['rounds'][0]['rollout'][key])
     assert torch.equal(fixed['path'][1]['state'],live['path'][1]['state'])
     for k,(f,l)in enumerate(zip(fixed['rounds'],live['rounds'])):
      assert torch.equal(f['rollout']['proposal_mean'],fixed['initial_policy_mean'])and torch.equal(f['rollout']['D'],fixed['initial_D']);assert f['rollout']['rng_seed']==l['rollout']['rng_seed']and torch.equal(f['rollout']['eps'],l['rollout']['eps'])
      if k:assert f['rollout']['rng_seed']!=fixed['rounds'][k-1]['rollout']['rng_seed']
     for arm,out in [('Fixed',fixed),('Live',live)]:name=f'{policy}_{arm}_{seed}';fits[name]=out;outputs[name]=out['final'].numpy()
     contracts.append(dict(policy=policy,seed=seed,first_samples_weights_update_exact=True,all40_paired_noises_exact=True,Fixed_initial_mean_covariance=True,Fixed40_fresh_batches=True,Live_current_mean_covariance=True))
   assert all(not q.requires_grad and q.grad is None for q in model.parameters());assert state_hash(rp.decoder.state_dict())==state_hash(model.ground_decoder.decoder.state_dict())
   cost=dict(case_seconds=time.perf_counter()-tick,decoder_seconds=rp.seconds,trajectory_decoder_calls=rp.calls,offset_decoder_calls=2*rp.calls,actual_updates=rp.backwards,new_Source=0,new_teacher=0)
   files.append(save(dest,dict(alias=a,dataset='hc2',query_ordinal=r['query_ordinal'],frame_ids=z['frame_ids'],interval=z['interval'],planned_positions=r['positions8'],outputs=outputs,fits=fits,contracts=contracts,target_sha256=manifest[a]['target_boxes_sha256'],GT_read=False,compute=cost)))
   put(D/'STATUS.json',dict(status='running',chunk=chunk,done=chunk*16+i+1,total=128,GT_read=False),True);print('sealed',a,chunk*16+i+1,'seconds',round(cost['case_seconds'],3),'updates',cost['actual_updates'],flush=True)
   # First two parents are the actual locked experiment, not an extra pilot.
   if chunk==0 and i==1:put(D/'FIRST_TWO_COST.json',dict(seconds_per_parent=cost['case_seconds'],rough128_parent_GPU_seconds=cost['case_seconds']*128,not_speed_promise=True))
   del rp,z,teacher,target,fixed,live,fits,outputs;gc.collect();torch.cuda.empty_cache()
  check();assert state_hash(model.state_dict())==mh;verify(True);put(D/f'CHUNK_{chunk}_BARRIER.json',dict(status='complete',count=16,files=files,GT_read=False,weights_unchanged=True));done=True
  if all((D/f'CHUNK_{c}_BARRIER.json').exists()for c in range(8)):
   allfiles=[f for c in range(8)for f in read(D/f'CHUNK_{c}_BARRIER.json')['files']];assert len(allfiles)==128;put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=128,logical_outputs=2432,adapted_fits=2304,files=allfiles,all_outputs_before_new_GT=True,GT_read=False))
 finally:put(D/'cost'/f'chunk{chunk}_{time.time_ns()}.json',dict(chunk=chunk,GPU_seconds=time.perf_counter()-start+3,files=len(files),complete=done));lease.close()

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','measure']);ap.add_argument('--chunk',type=int);a=ap.parse_args()
 try:prepare()if a.stage=='prepare'else measure(a.chunk)
 except BaseException as e:
  D.mkdir(exist_ok=True,parents=True);put(D/f'FAILURE_{time.time_ns()}.json',dict(stage=a.stage,chunk=a.chunk,type=type(e).__name__,error=str(e),partials_preserved=True));raise
