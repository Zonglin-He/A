"""Contracts for causal data separation, real visual dependence and joint choice."""
import unittest
import numpy as np
import torch
from vg_tta.stvg_joint_event_critic_v1 import EventCritic,collate,ranking_loss,temporal_candidates,candidate_tensor

class Contracts(unittest.TestCase):
 def test_visual_and_query_gradients(self):
  torch.manual_seed(3);m=EventCritic();x=torch.randn(1,9,1033,requires_grad=True);q=torch.randn(1,7,256,requires_grad=True)
  y=m(x,q);gx,gq=torch.autograd.grad(y.sum(),(x,q));self.assertGreater(float(gx[:,:,:1024].abs().sum()),0);self.assertGreater(float(gq.abs().sum()),0)
 def test_outside_context_not_removed(self):
  f=dict(times=torch.arange(9).float(),boxes=torch.ones(9,4),visual=torch.ones(9,1024),query=torch.ones(5,256));x,q=candidate_tensor(f,[2,5],1)
  self.assertEqual(len(x),9);self.assertEqual(int(x[:,-2].sum()),3);self.assertEqual(float(x[0,:1024].sum()),1024)
 def test_ties_not_negative_and_direction(self):
  s=torch.tensor([0.,0.,0.],requires_grad=True);y=torch.tensor([1.,0.,0.]);l,n=ranking_loss(s,y);self.assertEqual(n,2);g=torch.autograd.grad(l,s)[0];self.assertLess(float(g[0]),0);self.assertGreater(float(g[1]),0)
 def test_temporal_no_actions(self):
  z=[torch.zeros(1,6,2)]*2;r=[dict(frame_ids=list(range(o,12,2)))for o in [0,1]];c=temporal_candidates(z,r,list(range(12)),[0,12]);self.assertEqual(len(c['intervals']),3);self.assertFalse(c['Actionness_used']);self.assertEqual(len({tuple(x)for x in c['observation_fingerprints']}),3)
 def test_integer_boundary_membership(self):
  ids=torch.tensor([0,44,159,173,217,309,318,360,529])
  for fps in [23.976023976,29.97002997,20.,24.,25.]:
   f=dict(times=ids.float()/fps,boxes=torch.ones(len(ids),4),visual=torch.ones(len(ids),1024),query=torch.ones(5,256))
   x,q=candidate_tensor(f,[159,310],fps);self.assertTrue(torch.equal(x[:,-2].bool(),(ids>=159)&(ids<310)))
 def test_joint_not_marginal(self):
  p=np.array([[.26,.25,.01],[.02,.20,.26]]);j=np.unravel_index(p.argmax(),p.shape);s=(p.sum(1).argmax(),p.sum(0).argmax());self.assertNotEqual(j,s)

if __name__=='__main__':unittest.main()
