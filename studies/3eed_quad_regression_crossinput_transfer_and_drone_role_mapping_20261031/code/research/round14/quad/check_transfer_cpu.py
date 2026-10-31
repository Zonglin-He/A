"""Synthetic trainable last-head fixture; no actual source model/data/GT/GPU."""
from pathlib import Path
import sys,hashlib,json,random,tempfile
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'crossplatform/round14'),str(ROOT/'crossplatform/round7'),str(ROOT/'crossplatform/round9')]
from quad_transfer import run_transfer_episode,ARM_NAMES
from regression_adapter import SourceCache,raw_boxes,parameter_sha
from head_exact_fit import exact_fit_same_head

class MLP(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(*[nn.Identity() for _ in range(8)],nn.Conv1d(288,3,1))
class Head(nn.Module):
 def __init__(self):
  super().__init__();self.heading=False;self.center_residual_head=MLP();self.size_pred_head=MLP()
class FixtureModel(nn.Module):
 def __init__(self):
  super().__init__();self.frozen=nn.Linear(2,2);self.register_buffer('source_buffer',torch.tensor([3.]))
  self.prediction_heads=nn.ModuleList([nn.Identity() for _ in range(5)]+[Head()])
  with torch.no_grad():
   for layer in (self.prediction_heads[5].center_residual_head.net[8],self.prediction_heads[5].size_pred_head.net[8]):
    layer.weight.mul_(.01);layer.bias.fill_(2.)

def cache(model,seed,q):
 gen=torch.Generator().manual_seed(seed)
 hc=torch.rand((1,288,256),generator=gen);hs=torch.rand((1,288,256),generator=gen)
 base=torch.rand((1,256,3),generator=gen);scores=torch.zeros(1,256);scores[0,q]=1
 c=SourceCache(base,hc,hs,torch.zeros(1,256,6),scores,q,parameter_sha(model))
 with torch.no_grad():boxes=raw_boxes(model,c).detach().clone()
 return SourceCache(base,hc,hs,boxes,scores,q,parameter_sha(model))

def check():
 assert not torch.cuda.is_initialized();torch.manual_seed(1402);np.random.seed(1402);random.seed(1402)
 model=FixtureModel().eval().requires_grad_(False);a=cache(model,1,3);b=cache(model,2,7)
 target=a.boxes_raw6[0,3]+torch.tensor([.1,-.06,.05,.04,.03,.02])
 state={k:v.clone() for k,v in model.state_dict().items()};head=parameter_sha(model)
 py=random.getstate();npstate=np.random.get_state();trng=torch.get_rng_state().clone()
 seen=[]
 def hook(m,bc,expected):
  assert parameter_sha(m)!=head and torch.equal(raw_boxes(m,bc),expected)
  seen.append('post_real_step_before_restore');return {'passed':True,'synthetic_hook_order_only':True}
 result=run_transfer_episode(model,a,b,target,True,full_B_forward_hook=hook)
 assert result['engineering_checks_pass'] and seen==['post_real_step_before_restore']
 assert result['A_update_executed'] and tuple(result['arm_names'])==ARM_NAMES
 traces=result['fraction_traces']
 for alpha in (.5,1.):
  t=traces[str(alpha)]
  assert t['actual_optimizer_steps']==1 and t['nonzero_gradient_parameter_tensors']==4 and t['observed_parameter_change']
  assert t['source_A_head_replay_restored_bitexact'] and t['source_B_head_replay_restored_bitexact']
  assert t['B_parameter_kernel_all256_max_abs_error_m']<2e-4
  assert t['A_fit_max_coordinate_error_m']<=5e-5
 assert torch.equal(torch.get_rng_state(),trng) and random.getstate()==py
 now=np.random.get_state();assert now[0]==npstate[0] and np.array_equal(now[1],npstate[1]) and now[2:]==npstate[2:]
 assert all(torch.equal(v,state[k]) for k,v in model.state_dict().items())
 assert parameter_sha(model)==head and all(not p.requires_grad and p.grad is None for p in model.parameters())
 assert all(not m.training for m in model.modules())
 # Audited R9 true-step recipe yields the same selected A full-fit result.
 r9=exact_fit_same_head(model,a,target,[1]*6)
 assert np.max(np.abs(np.array(traces['1.0']['A_selected_after'])-r9['boxes_raw6'][0,3].numpy()))<2e-4
 # Changing unused B features changes transport, but never the A fitted update,
 # A gradient, loss or either learning rate. It may produce a ratio>1, retained.
 b2=SourceCache(b.base_xyz,b.center_features*3,b.size_features*3,
      b.boxes_raw6,b.text_scores,b.query_index,b.selected_parameter_sha256)
 with torch.no_grad():b2boxes=raw_boxes(model,b2).detach().clone()
 b2=SourceCache(b2.base_xyz,b2.center_features,b2.size_features,b2boxes,b2.text_scores,b2.query_index,b2.selected_parameter_sha256)
 changedB=run_transfer_episode(model,a,b2,target,True)
 for alpha in (.5,1.):
  x=traces[str(alpha)];y=changedB['fraction_traces'][str(alpha)]
  for key in ('loss_before','learning_rates_from_A_only','parameter_gradient_norm_by_name','parameter_delta_max_by_name','A_selected_after'):
   assert x[key]==y[key]
 assert changedB['feature_kernels']['B_selected_kernel_ratio_center']>1
 false=run_transfer_episode(model,a,b,target,False)
 zero=run_transfer_episode(model,a,b,a.boxes_raw6[0,3],True)
 for rr in (false,zero):
  assert not rr['A_update_executed'] and not rr['fraction_traces']
  assert torch.equal(rr['B_boxes_raw6'],b.boxes_raw6[0,7][None].repeat(7,1))
 sA=a.boxes_raw6[0,3];sB=b.boxes_raw6[0,7];delta=target-sA
 raw=result['B_boxes_raw6']
 assert torch.equal(raw[1],sB+delta)
 assert torch.equal(raw[2],sB+(sB[3:]/sA[3:]).repeat(2)*delta)
 assert not torch.cuda.is_initialized()
 from run_quad_transfer import completed_run_files
 with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent,prefix='zero_hook_synthetic_') as temp:
  tmp=Path(temp)
  required=('registered_plan.json','CACHE_BRIDGES.jsonl','TRACE_PRIVATE.jsonl','TRACE_PRIVATE.json')
  for name in required:(tmp/name).write_text('synthetic only')
  nohook=completed_run_files(tmp,False)
  assert set(nohook)==set(required)
  try:completed_run_files(tmp,True)
  except FileNotFoundError:pass
  else:raise AssertionError('Claimed actual hook missing file must fail')
  (tmp/'FIRST_NONZERO_A_FULLMODEL_B_CHECK.json').write_text('synthetic hook receipt')
  assert 'FIRST_NONZERO_A_FULLMODEL_B_CHECK.json' in completed_run_files(tmp,True)
 checks={'true_four_tensor_gradients_and_parameter_steps':True,'half_and_full_fresh_source_episodes':True,
  'B_readout_after_real_A_step_hook_order':True,'cross_input_all256_parameter_kernel_equivalence':True,
  'A_fit_matches_audited_R9_recipe':True,'B_features_not_in_A_loss_gradient_LR_gate':True,
  'kernel_ratio_gt1_retained_without_B_clip':True,'direct_additive_and_source_size_normalized_controls_exact':True,
  'A_gate_false_or_zero_residual_identity_no_step':True,'state_grad_modes_buffers_Python_NumPy_torchRNG_restored':True,
  'no_actual_model_sensor_GT_or_GPU_access':True,
  'zero_signal_prefix_optional_hook_file_not_required_and_real_hook_required':True}
 return {'state':'passed_R14_synthetic_actual_linear_head_transfer_checks','checks':checks,
   'max_parameter_kernel_error_m':max(t['B_parameter_kernel_all256_max_abs_error_m'] for t in traces.values()),
   'program_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
      [ROOT/'crossplatform/round14/quad_transfer.py',ROOT/'crossplatform/round14/run_quad_transfer.py',Path(__file__)]}}
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();r=check()
 if args.output.exists():raise FileExistsError(args.output)
 args.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps({'state':r['state'],'n_checks':len(r['checks']),'max_kernel_error':r['max_parameter_kernel_error_m']}))
