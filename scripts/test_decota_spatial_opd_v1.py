"""Meaningful CPU contracts for the stochastic interface and no-feedback Adam."""
import sys,unittest,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from vg_tta.decota_spatial_opd_v1 import *

class Contracts(unittest.TestCase):
 def test_frame_aligned_roundtrip(self):
  b=torch.tensor([[.1,.2,.3,.4],[.6,.7,.8,.9]],dtype=torch.float64)
  self.assertLess((action_boxes(mean_coordinates(b))-b).abs().max().item(),1e-15)
 def test_dino_isolated_and_analytic_gradient(self):
  mu=torch.tensor([[.1,-.2,.3,-.4]],dtype=torch.float64,requires_grad=True)
  e=torch.tensor([[.55,.45,.3,.2]],dtype=torch.float64,requires_grad=True)
  rr=rollout(mu.detach(),e,'contract',0)
  self.assertTrue(all(not rr[k].requires_grad for k in ['samples','mean','rewards','weights']))
  l=likelihood_loss(mu,rr['mean'],rr['samples'],rr['weights']);g=torch.autograd.grad(l,mu)[0]
  expected=-((rr['weights']-1/32)[:,:,None]*(rr['samples']-mu.detach()[:,None,:])).sum(1)/.25**2
  self.assertTrue(torch.allclose(g,expected,atol=1e-12,rtol=0));self.assertIsNone(e.grad)
 def test_no_feedback_skips_adam_with_nonzero_moments(self):
  p=torch.tensor([1.],requires_grad=True);o=torch.optim.Adam([p],lr=.03)
  optimizer_step(o,p.square().sum(),[p],True);before=p.detach().clone();old=copy.deepcopy(o.state_dict())
  self.assertIsNone(optimizer_step(o,p.sum()*0,[p],False));self.assertTrue(torch.equal(p,before))
  for k,v in old['state'][0].items():self.assertTrue(torch.equal(v,o.state_dict()['state'][0][k]))
 def test_equal_reward_zero_gradient(self):
  mu=torch.tensor([[0.,0.,0.,0.]],requires_grad=True)
  z=torch.ones(1,32,4);w=torch.full((1,32),1/32)
  g=torch.autograd.grad(likelihood_loss(mu,mu.detach(),z,w),mu)[0]
  self.assertEqual(g.count_nonzero().item(),0)
 def test_finite_difference(self):
  mu=torch.zeros(1,4,dtype=torch.float64,requires_grad=True);rr=rollout(mu.detach(),torch.tensor([[.5,.5,.2,.2]],dtype=torch.float64),'fd',0)
  fn=lambda m:likelihood_loss(m,rr['mean'],rr['samples'],rr['weights'])
  g=torch.autograd.grad(fn(mu),mu)[0];fd=[]
  for j in range(4):
   h=torch.zeros_like(mu);h[0,j]=1e-6;fd.append((fn(mu.detach()+h)-fn(mu.detach()-h)).item()/2e-6)
  self.assertTrue(torch.allclose(g,torch.tensor([fd],dtype=mu.dtype),atol=1e-9,rtol=0))
 def test_antithetic_refresh(self):
  m=torch.zeros(2,4);e=torch.tensor([[.4,.5,.3,.2],[.6,.6,.3,.3]])
  a=rollout(m,e,'a',0);b=rollout(m,e,'a',1)
  self.assertTrue(torch.equal(a['samples'][:,:16],-a['samples'][:,16:]));self.assertFalse(torch.equal(a['samples'],b['samples']))
 def test_feedback_shuffle_preserves_mass(self):
  m=torch.zeros(1,4);e=torch.tensor([[.6,.55,.3,.3]])
  a=rollout(m,e,'a',0);b=rollout(m,e,'a',0,True)
  self.assertTrue(torch.equal(a['samples'],b['samples']))
  self.assertTrue(torch.equal(a['rewards'].sort(1).values,b['used_rewards'].sort(1).values))
 def test_invalid_chart_fails(self):
  with self.assertRaises(AssertionError):mean_coordinates(torch.tensor([[0.,.5,.2,.2]]))

if __name__=='__main__':unittest.main()
