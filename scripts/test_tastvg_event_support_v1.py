"""Analytical support, reduction, gradient/detachment and empty-evidence controls."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from vg_tta.tastvg_event_support_v1 import event_weights,event_rewards,event_geometry,event_target_loss
from vg_tta.tastvg_spatial_online_opd_s1_v1 import geometry
from scripts.tastvg_event_support_metrics_v1 import weights_from_candidates,weighted_rewards
def run():
 ids=[3,10,14,25,50]
 support=[dict(indices=[0,2],physical_interval=[3,15]),dict(indices=[2,3],physical_interval=[14,26])]
 w=event_weights(support,ids)
 np.testing.assert_array_equal(w,[.5,.5,1.,.5,0.])
 np.testing.assert_allclose(w,weights_from_candidates(support,ids),atol=0,rtol=0)
 boxes=np.array([[[.5,.5,.2,.2]]*5,[[.6,.5,.2,.2]]*5])
 expert=boxes[0];valid=np.array([True,False,True,False,True])
 np.testing.assert_allclose(event_rewards(boxes,expert,valid,w),[1.,1/3],atol=1e-12,rtol=0)
 np.testing.assert_allclose(event_rewards(boxes,expert,valid,w),weighted_rewards(boxes,expert,valid,w),atol=1e-12,rtol=0)
 assert event_rewards(boxes,expert,[False]*4+[True],w) is None
 assert event_rewards(boxes,expert,[False]*5,w) is None
 c=torch.tensor([[.4,.4,.2,.2],[.5,.5,.2,.2]],dtype=torch.float64,requires_grad=True)
 target=torch.tensor([[[.6,.4,.2,.2],[.8,.5,.2,.2]],[[.4,.5,.2,.2],[.5,.8,.2,.2]]],dtype=torch.float64,requires_grad=True)
 wt=torch.tensor([1.,0.],dtype=torch.float64,requires_grad=True)
 d=event_geometry(c,target,5.,2.,wt)
 torch.testing.assert_close(d,geometry(c[:1],target[:,:1],5.,2.),atol=0,rtol=0)
 gradients=torch.autograd.grad(d.sum(),[c,target,wt],allow_unused=True)
 assert gradients[1] is None and gradients[2] is None
 assert torch.equal(gradients[0][1],torch.zeros(4,dtype=torch.float64)) and gradients[0][0].abs().sum()>0
 # Uniform support recovers original geometry and rank-RKL values/gradients.
 from vg_tta.tastvg_selected_rollout_v1 import target_loss
 a=c.detach().clone().requires_grad_();b=c.detach().clone().requires_grad_();scores=torch.tensor([.3,.7],dtype=torch.float64)
 la,*_=target_loss(a,target,scores,[5.,2.],.34902548789596055,1.,'rank',1.)
 lb,*_=event_target_loss(b,target,scores,[5.,2.],.34902548789596055,1.,[1.,1.])
 torch.testing.assert_close(la,lb,atol=1e-12,rtol=0)
 torch.testing.assert_close(torch.autograd.grad(la,a)[0],torch.autograd.grad(lb,b)[0],atol=1e-12,rtol=0)
 # Flat nonempty rewards retain original loss rather than an invented no-op.
 flat=event_target_loss(b,target,torch.ones(2),[5.,2.],1.,1.,[1.,0.])[0]
 assert flat>0 and torch.autograd.grad(flat,b)[0].abs().sum()>0
 assert not torch.cuda.is_initialized()
 print('PASS: irregular-frame equal consensus/half-open intervals, weighted reward and zero mass, native geometry, detached targets/weights, off-support zero gradient, full-support RKL parity, unchanged flat semantics')
if __name__=='__main__':run()
