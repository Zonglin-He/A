"""Independent CPU readback: policy, reward, gradient, Adam, barriers, metrics."""
import os,sys,time,collections,math
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_e_retaining1792_route32_v1 import D,P,PRE,CLOSED,ARMS,STAGES,read,put,load,sha,verify

def main():
 import torch,numpy as np
 from methods.decota_final_simplified_v1.objectives import SpatialLoss
 from vg_tta.stvg_dino_query_closedloop32_v1 import corrected
 torch.set_num_threads(4);p=verify(True);start=time.perf_counter();counts=collections.Counter();maxerr=collections.defaultdict(float);score=read(P/'RESULTS_PUBLIC.json');previous={r['alias']:r for r in read(ROOT/'results/stvg_error_state/2026-10-19/reference_capacity32/RESULTS_PUBLIC.json')['rows']};sb=read(D/'ORACLE_INPUT_BARRIER.json');assert (D/'REAL_PREDICTION_BARRIER.json').stat().st_mtime_ns<(D/'ORACLE_INPUT_BARRIER.json').stat().st_mtime_ns
 for stage in STAGES:
  bar=read(D/(stage.upper()+'_PREDICTION_BARRIER.json'));assert bar['count']==32
  for rec in bar['files']:
   assert sha(ROOT/rec['path'])==rec['sha256'];z=load(ROOT/rec['path']);a=z['alias'];src=load(PRE/'source'/(a+'.pt'));e=load(PRE/'E'/(a+'.pt'));inp=load(D/('inputs_'+stage)/(a+'.pt'));assert z['GT_used']==(stage=='oracle')and z['positions']==[x['position']for x in inp['anchors']];assert z['interval']==e['interval']and np.array_equal(z['outputs']['E_MBR'],e['E']);pos=z['positions'];target=torch.tensor([t['box']for t in inp['anchors']],dtype=torch.float64).reshape(-1,4);fixed0=z['fits']['Fixed']['rounds'];live=z['fits']['Live']['rounds'];row=next(r for r in score['rows']if r['alias']==a and r['stage']==stage)
   for k in ['v','t','s','observed_actual_s','unobserved_actual_s']:
    u,w=row['values']['E_MBR'][k],previous[a]['values']['E_MBR'][k];assert u is None and w is None or u is not None and w is not None and abs(u-w)<1e-12;counts['baseline_metric_exact']+=1
   for arm,f in z['fits'].items():
    cfg=f['config'];assert sum(t.numel()for t in f['initial'].values())==1792;assert len(f['rounds'])==(cfg['steps']if pos else 0);assert torch.equal(f['path'][0]['boxes'],torch.as_tensor(e['E']))and torch.equal(f['path'][0]['native'],src['Source']);counts['initial1792_zero_E_exact']+=1
    names=list(f['initial']);param=torch.cat([f['initial'][n].flatten()for n in names]).clone().requires_grad_();opt=torch.optim.Adam([param],lr=cfg['lr'],betas=(.9,.999),eps=1e-8);sl=SpatialLoss(inp['anchors'],torch.as_tensor(e['E']));assert abs(float(sl(f['path'][-1]['boxes']))-f['path'][-1]['anchor_loss'])<1e-12;counts['anchor_endpoint_loss_exact']+=1
    for j,path in enumerate(f['path']):
     mapped=corrected(path['native'],src['Source'],torch.as_tensor(e['E']));map_error=float((mapped-path['boxes']).abs().max());maxerr['cross_device_E_map']=max(maxerr['cross_device_E_map'],map_error);assert map_error<5e-13;assert torch.equal(path['boxes'],torch.as_tensor(z['outputs'][arm]))if j==len(f['path'])-1 else True;counts['native_to_E_path_numerical']+=1
    for j,rr in enumerate(f['rounds']):
     path=f['path'][j];nextp=f['path'][j+1];mu=torch.logit(path['native'][pos]);roll=rr['rollout']
     if roll is not None:
      expected=fixed0[0]['rollout']if arm=='Fixed'else roll
      if arm=='Fixed':
       for k in ['mean','samples','rewards','weights']:assert torch.equal(roll[k],expected[k])
       counts['fixed_first_feedback_exact']+=1
      else:assert torch.allclose(roll['mean'],mu,rtol=0,atol=4e-6);counts['live_current_native_mean']+=1
      b=roll['candidate_boxes'];lo=b[...,:2]-b[...,2:]/2;hi=b[...,:2]+b[...,2:]/2;tlo=target[:,None,:2]-target[:,None,2:]/2;thi=target[:,None,:2]+target[:,None,2:]/2;inter=(torch.minimum(hi,thi)-torch.maximum(lo,tlo)).clamp_min(0).prod(-1);den=(hi-lo).prod(-1)+(thi-tlo).prod(-1)-inter;reward=inter/den.clamp_min(1e-7);err=float((reward-roll['rewards']).abs().max());maxerr['reward']=max(maxerr['reward'],err);assert err<1e-12
      weights=torch.softmax(reward/cfg['tau'],-1);assert torch.allclose(weights,roll['weights'],rtol=0,atol=2e-14);analytic=((mu-roll['mean'])/cfg['sigma']**2-((weights-1/32)[...,None]*(roll['samples']-mu[:,None,:])).sum(1)/cfg['sigma']**2)/len(pos);err=float((analytic-rr['autograd_mean_gradient']).abs().max());maxerr['policy_gradient']=max(maxerr['policy_gradient'],err);tolerance=4e-4+8*torch.finfo(mu.dtype).eps*max(1.,float(rr['autograd_mean_gradient'].abs().max()))+4e-6/(cfg['sigma']**2*len(pos));assert err<tolerance;counts['reward_weight_gradient_checks']+=1
      valid=(reward!=reward[:,:1]).any();assert bool(valid)==rr['updated']
     if rr['updated']:
      grad=path['update']['gradient'];opt.zero_grad(set_to_none=True);param.grad=grad.clone();opt.step();expectednext=torch.cat([nextp['state'][n].flatten()for n in names]);err=float((param.detach()-expectednext).abs().max());maxerr['Adam_state']=max(maxerr['Adam_state'],err);assert err<3e-6;counts['independent_Adam_update']+=1
     else:assert all(torch.equal(path['state'][n],nextp['state'][n])for n in names);counts['no_information_no_state_change']+=1
    assert len([t for t in f['rounds']if t['updated']])==f['gradient_calls'];assert f['gradient_calls']==row['fit_diagnostics'][arm]['gradient_calls'];counts['fit_count_readback']+=1
   if fixed0:
    for k in ['samples','rewards','weights']:assert torch.equal(fixed0[0]['rollout'][k],live[0]['rollout'][k]);counts['first_round_Fixed_Live_exact']+=1
   counts['full_query_stage_checks']+=1
 for stage,gg in score['groups'].items():
  for ds,g in gg.items():
   rr=[r for r in score['rows']if r['stage']==stage and r['dataset']==ds]
   for name,eps in g['paired'].items():
    a,b=name.split('_minus_')
    for k,t in eps.items():
     vv=np.array([100*(r['values'][a][k]-r['values'][b][k])for r in rr if r['values'][a][k]is not None and r['values'][b][k]is not None]);assert len(vv)==t['parents']and abs(vv.mean()-t['mean'])<1e-12;ix=np.random.default_rng(20261018).integers(len(vv),size=(10000,len(vv)));assert np.allclose(np.quantile(vv[ix].mean(1),[.025,.975]),t['CI95'],rtol=0,atol=1e-12);counts['independent_paired_mean_CI']+=1
 assert not torch.cuda.is_initialized();verify(True);put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',checks=dict(counts),maximum_errors=dict(maxerr),CPU_seconds=time.perf_counter()-start,real_seal_precedes_oracle_GT_inputs=True,registries_and_original_inputs_unchanged=True,no_extra_models_GPU_or_training=True));print('ROOT PASS',dict(counts),dict(maxerr),flush=True)
if __name__=='__main__':main()
