import unittest
import numpy as np
import torch
from vg_tta.tastvg_self_rank_v1 import rank_loss
from vg_tta.tastvg_spatial_rank_s11_v1 import average_ranks
from vg_tta.tastvg_spatial_online_opd_s1_v1 import geometry

class SelfRankTest(unittest.TestCase):
 def test_head_first_and_ties(self):
  np.testing.assert_array_equal(average_ranks([.1,.4,.4,.1]),[2.5,.5,.5,2.5])
 def test_reverse_kl_logits_gradient(self):
  z=torch.tensor([-.1,-.3,-.6],dtype=torch.float64,requires_grad=True)
  p=z.softmax(0);rank=average_ranks(p.detach().numpy());q=torch.tensor(-rank).softmax(0)
  kl=(p*(p.log()-q.log())).sum();g,=torch.autograd.grad(kl,z)
  torch.testing.assert_close(g,p*(p.log()-q.log()-kl));self.assertFalse(q.requires_grad)
 def test_frozen_self_target_not_zero_gradient(self):
  b=torch.tensor([[.5,.5,.2,.2]],requires_grad=True)
  target=torch.tensor([[[.5,.5,.2,.2]],[[.53,.5,.2,.2]],[[.5,.57,.2,.2]]])
  p=(-geometry(b,target,1.,1.)).softmax(0).detach();rank=average_ranks(p.numpy())
  loss,pi,q,d=rank_loss(b,target,rank,[1.,1.],'self_rank');g,=torch.autograd.grad(loss,b)
  self.assertEqual(rank[0],0);self.assertFalse(q.requires_grad);self.assertTrue(torch.isfinite(g).all());self.assertGreater(float(g.norm()),0)
if __name__=='__main__':unittest.main()
