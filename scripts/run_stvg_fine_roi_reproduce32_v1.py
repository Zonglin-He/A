"""Deterministic cached rerun of completed Fine ROI scoped OPD; no new GT scoring."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=ROOT/'artifacts/stvg_fine_roi_reproduce32_v1';P=ROOT/'results/stvg_error_state/2026-10-19/fine_roi_reproduce32'
BASE=ROOT/'artifacts/stvg_fine_roi_scoped_opd32_v1';BP=ROOT/'results/stvg_error_state/2026-10-19/fine_roi_scoped_opd32'
CODE=['scripts/run_stvg_fine_roi_reproduce32_v1.py','protocols/stvg_fine_roi_reproduce32_v1.md']

def verify():
 from main_method.runtime import PinVerifier,resource_guard
 p=read(D/'PLAN_LOCK.json');r=read(D/'RUNTIME_LOCK.json');assert sha(D/'PLAN_LOCK.json')==r['plan_sha256'];PinVerifier(r['pins']).verify();resource_guard();return p

def main():
 import torch,numpy as np
 from scripts.run_stvg_fine_roi_scoped_opd32_v1 import verify as original_verify,FINE,PRE
 from vg_tta.stvg_fine_roi_scoped_opd32_v1 import build,scope,restore,scoped_direct,scoped_fit,interpolation_equivalence,MODES
 assert not D.exists();D.mkdir();P.mkdir();torch.set_num_threads(4);original_verify(full=True);oldplan=read(BASE/'PLAN_LOCK.json');oldbar=read(BASE/'GLOBAL_PREDICTION_BARRIER.json');assert oldbar['count']==32
 pins={f:sha(ROOT/f)for f in CODE+['artifacts/stvg_fine_roi_scoped_opd32_v1/PLAN_LOCK.json','artifacts/stvg_fine_roi_scoped_opd32_v1/RUNTIME_LOCK.json','vg_tta/stvg_fine_roi_scoped_opd32_v1.py','vg_tta/stvg_roi_layout_opd32_v1.py','vg_tta/stvg_discriminative_dense32_v1.py','methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json']}
 for f in oldbar['files']:assert sha(ROOT/f['path'])==f['sha256'];pins[f['path']]=f['sha256']
 for r in oldplan['records']:
  for f in [FINE/'features'/(r['alias']+'.pt'),PRE/'feedback'/(r['alias']+'.pt')]:pins[str(f.relative_to(ROOT))]=sha(f)
 plan=dict(status='locked',count=32,configuration=oldplan['configuration'],arms=['Fine_fixed','Fine_live','Fine_shuffle'],actual_fits=96,round_slots=192,original_commit='848b88d1f2246f55b581c403c94140de2cb88f1b',same_cached_inputs=True,new_experts_DINO_training_models_GPU=0,new_GT_scoring=False,CPU_cap_seconds=900,records=oldplan['records'],decision='Require all regenerated templates/probabilities/gradients/states/outputs exact against saved run. Reuse already sealed scores only after full global output equality; no new scientific confirmation.')
 put(D/'PLAN_LOCK.json',plan);put(D/'RUNTIME_LOCK.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=pins));start=time.perf_counter();files=[];rows=[];checks={};maxerr=0.
 def eq(a,b):
  nonlocal maxerr
  if torch.is_tensor(a):
   assert torch.is_tensor(b)and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b);checks['tensor_or_array_fields']=checks.get('tensor_or_array_fields',0)+1
   if a.is_floating_point()and a.numel():maxerr=max(maxerr,float((a-b).abs().max()))
  elif isinstance(a,np.ndarray):assert isinstance(b,np.ndarray)and np.array_equal(a,b);checks['tensor_or_array_fields']=checks.get('tensor_or_array_fields',0)+1
  elif isinstance(a,dict):
   assert set(a)==set(b)
   for k in a:eq(a[k],b[k])
  elif isinstance(a,(list,tuple)):
   assert len(a)==len(b)
   for x,y in zip(a,b):eq(x,y)
  else:assert a==b,(type(a),type(b))
 for ix,r in enumerate(plan['records']):
  verify();assert time.perf_counter()-start<900;a=r['alias'];x=load(FINE/'features'/(a+'.pt'));old=load(BASE/'predictions'/(a+'.pt'));fb=load(PRE/'feedback'/(a+'.pt'))['feedback'];E=x['E'].double();positions=x['positions'];boxes=[torch.as_tensor(b).tolist()for b in x['boxes']];ctx=build(x['maps'],x['valid'],positions,boxes);context=load(BASE/'context'/(a+'.pt'));eq(ctx['templates'],context['templates']);eq(ctx['weights'],context['weights']);relevant,allowed=scope(x['frame_ids'],x['planned_positions'],x['interval']);eq(allowed,old['allowed']);ep=fb['noise'];cfg=plan['configuration'];perm=fb['permutation'];fits={k:scoped_fit(E,ctx,ep,cfg,x['planned_positions'],m,perm,allowed)for k,m in MODES.items()}
  for k,v in fits.items():eq(v,old['fits'][k])
  eq(fits['Fine_fixed']['path'][1],fits['Fine_live']['path'][1]);eq(fits['Fine_live']['rounds'][1]['sampling_center'],fits['Fine_live']['path'][1]['eta'])
  direct=scoped_direct(E,ctx,ep,cfg,x['planned_positions'],allowed);eq(direct,old['direct']);outputs={k:torch.as_tensor(v).clone()for k,v in old['outputs'].items()};outputs.update(E_MBR=E,Fine_direct=direct['boxes'],**{k:f['final']for k,f in fits.items()})
  for k,b in outputs.items():eq(b,torch.as_tensor(old['outputs'][k]));assert torch.equal(b[~allowed],E[~allowed])and torch.equal(b[:,2:],E[:,2:])
  if not positions:assert all(torch.equal(b,E)for b in outputs.values())
  files.append(save(D/'predictions'/(a+'.pt'),dict(alias=a,dataset=r['dataset'],outputs=outputs,fits=fits,direct=direct,allowed=allowed,GT_read=False,exact_original_outputs=True)));rows.append(dict(alias=a,dataset=r['dataset'],active=bool(positions),allowed_nodes=int(allowed.sum()),retained_anchors=len(positions),all_three_fits_and_nine_outputs_exact=True,second_Live_center_actual=True));print(a,ix+1,round(time.perf_counter()-start,3),flush=True)
 verify();original_verify(full=True);assert time.perf_counter()-start<900 and not torch.cuda.is_initialized();put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=files,actual_fit_episodes=96,round_slots=192,CPU_prediction_and_equality_seconds=time.perf_counter()-start,no_GT_read=True))
 # No labels/annotation loader: score reuse is justified only by all32x9 exact outputs.
 prior=read(BP/'RESULTS_PUBLIC.json');assert len(prior['rows'])==32;put(P/'RESULTS_PUBLIC.json',prior);put(P/'REPRODUCTION.json',dict(status='pass',count=32,actual_fits=96,round_slots=192,checks=checks,max_tensor_difference=maxerr,CPU_seconds=time.perf_counter()-start,all_outputs_and_paths_exact=True,scores='exact inherited original sealed metrics, not new GT scoring',source_score_sha256=sha(BP/'RESULTS_PUBLIC.json'),rows=rows,new_GPU_models_experts_DINO_training_GT_scoring=0,independent_scientific_confirmation=False));put(D/'FINAL_COMPLETION.json',dict(status='complete',count=32,measurement='exact cached deterministic reproduction',public_sync='pending'));print('ALL32 EXACT ACTUAL RERUN',flush=True)
if __name__=='__main__':main()
