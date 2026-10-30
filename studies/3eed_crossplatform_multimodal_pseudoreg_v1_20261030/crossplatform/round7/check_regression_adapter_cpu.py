"""Synthetic CPU interface/mathematics fixture only; not a 3EED result."""
import hashlib,json,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from regression_adapter import (PARAMETER_NAMES,StepConfig,PseudoTarget,capture_source,
    adapt,first_noun_mask,parameter_sha,episode)

class MLP(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Conv1d(288,288,1,bias=False),nn.BatchNorm1d(288),nn.ReLU(),nn.Dropout(.3),nn.Conv1d(288,288,1,bias=False),nn.BatchNorm1d(288),nn.ReLU(),nn.Dropout(.3),nn.Conv1d(288,3,1))
    def forward(self,x):return self.net(x)
class Head(nn.Module):
    def __init__(self):
        super().__init__();self.heading=False;self.center_residual_head=MLP();self.size_pred_head=MLP()
        with torch.no_grad():self.size_pred_head.net[8].bias.copy_(torch.tensor([4.,2.,1.5]))
class Synthetic(nn.Module):
    def __init__(self):
        super().__init__();self.prediction_heads=nn.ModuleList([Head() for _ in range(6)])
        self.register_buffer('synthetic_features',torch.randn(1,288,7))
    def forward(self,inputs):
        f=self.synthetic_features;h=self.prediction_heads[5];base=inputs['point_clouds']
        return {'last_center':base+h.center_residual_head(f).transpose(1,2),'last_pred_size':h.size_pred_head(f).transpose(1,2),'last_base_xyz':base,'last_proj_queries':torch.nn.functional.normalize(f[:,:64].transpose(1,2),dim=-1),'proj_tokens':torch.nn.functional.normalize(torch.arange(4*64,dtype=torch.float32).reshape(1,4,64)+1,dim=-1)}

def snapshot(m):
    return {k:v.detach().clone() for k,v in m.state_dict().items()}
def assert_state(m,s):
    assert s.keys()==m.state_dict().keys()
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in s.items())
def main():
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(2);torch.manual_seed(7);random.seed(9);np.random.seed(11)
    m=Synthetic();m.train();m.prediction_heads[2].eval()
    # Existing nonzero gradients must survive every source/adaptation episode.
    for i,p in enumerate(m.parameters()):
        p.requires_grad_(i%3!=0)
        if i%5==0:p.grad=torch.ones_like(p)
    gradrefs={n:p.grad for n,p in m.named_parameters()};flags={n:p.requires_grad for n,p in m.named_parameters()};modes=[x.training for x in m.modules()]
    s=snapshot(m);rng=torch.get_rng_state().clone();pyrng=random.getstate();nprng=np.random.get_state()
    inputs={'point_clouds':torch.randn(1,7,3),'text':['the car. not mentioned']};rng=torch.get_rng_state().clone()
    mask=first_noun_mask(inputs['text'][0],[(0,0),(0,3),(4,7),(7,8)])
    c=capture_source(m,inputs,mask,query_index=2);repeat=capture_source(m,inputs,mask,query_index=2)
    assert torch.equal(c.boxes_raw6,repeat.boxes_raw6) and torch.equal(c.text_scores,repeat.text_scores)
    assert_state(m,s);assert torch.equal(torch.get_rng_state(),rng)
    target=c.boxes_raw6[0,2].clone()+torch.tensor([.2,-.1,.3,.4,-.2,.1]);target.requires_grad_()
    w=torch.tensor([1.,.5,0.,.75,0.,1.],requires_grad=True)
    r=adapt(m,c,PseudoTarget(2,target,w),StepConfig(steps=3,lr=.01,beta_metres=1.))
    atol=2e-6;rtol=2e-6
    assert torch.allclose(r['parameter_SGD'],r['exact_output_space_SGD'],atol=atol,rtol=rtol)
    assert not torch.equal(r['parameter_SGD'],r['source'])
    assert torch.equal(r['direct_G'][0,2][w==0],r['source'][0,2][w==0])
    assert torch.equal(r['parameter_SGD'][0,:,2],r['source'][0,:,2])
    assert torch.equal(r['parameter_SGD'][0,:,4],r['source'][0,:,4])
    assert target.grad is None and w.grad is None
    assert all(x['nonzero_gradient_parameter_tensors']==4 and x['parameter_gradient_norm']>0 for x in r['trace'])
    assert r['spillover']['other_query_count_changed']>0
    assert_state(m,s);assert torch.equal(torch.get_rng_state(),rng);assert random.getstate()==pyrng
    now=np.random.get_state();assert now[0]==nprng[0] and np.array_equal(now[1],nprng[1]) and now[2:]==nprng[2:]
    assert [x.training for x in m.modules()]==modes
    assert all(p.grad is gradrefs[n] and p.requires_grad==flags[n] for n,p in m.named_parameters())
    z=adapt(m,c,PseudoTarget(2,target,torch.zeros(6)),StepConfig(3,.01,1.))
    assert z['all_weights_zero'] and len(z['trace'])==0
    assert torch.equal(z['source'],z['direct_G']) and torch.equal(z['source'],z['parameter_SGD'])
    caught=False
    try:
        with episode(m):
            with torch.no_grad():m.prediction_heads[5].center_residual_head.net[8].weight.add_(1);m.synthetic_features.add_(3)
            torch.rand(1);random.random();np.random.rand();raise RuntimeError('synthetic forced failure')
    except RuntimeError:caught=True
    assert caught;assert_state(m,s);assert torch.equal(torch.get_rng_state(),rng)
    assert [x.training for x in m.modules()]==modes
    assert all(p.grad is gradrefs[n] and p.requires_grad==flags[n] for n,p in m.named_parameters())
    try:first_noun_mask('unrecognized object',[(0,0),(0,12),(13,19)])
    except ValueError:pass
    else:raise AssertionError('No-noun must fail closed')
    assert not torch.cuda.is_initialized()
    receipt={'state':'synthetic_CPU_fixture_pass_no_real_model_or_task_result','parameter_names':list(PARAMETER_NAMES),'updated_parameter_count':1734,'source_repeat_bitexact':True,'cached_Conv_bridge_bitexact':True,'SGD_vs_exact_output_space_allqueries_max_abs_error':float((r['parameter_SGD']-r['exact_output_space_SGD']).abs().max()),'predeclared_comparison_atol':atol,'predeclared_comparison_rtol':rtol,'shared_Conv_spillover_exposed':r['spillover'],'zero_weight_coordinates_unchanged':True,'target_and_reliability_stopgradient':True,'all_zero_weights_source_fallback':True,'normal_and_exception_restore_parameters_buffers_modes_grad_references_requiresgrad_RNG':True,'no_noun_GTfree_failclosed':True,'feature_norms':r['feature_norms'],'cuda_initialized':False,'no_actual_3EED_construction_checkpoint_data_GT_model_scoring':True,'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('regression_adapter.py')]}}
    Path(__file__).with_name('CPU_INTERFACE_FIXTURE.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
