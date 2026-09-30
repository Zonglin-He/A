import unittest
import numpy as np
import torch
from vg_tta.tastvg_noise_decomposition_n1_v1 import categories,eligible,hash_subset,directional_loss
class N1Test(unittest.TestCase):
 def test_four_types_and_ties(self):
  ye,yg=categories([.5,.7,.3,.7,.3,.5,.7,.3,.5],[.5,.7,.3,.3,.7,.7,.5,.5,.5])
  self.assertEqual(eligible('useful',ye,yg),[1,2]);self.assertEqual(eligible('noisy',ye,yg),[3,4])
  self.assertEqual(eligible('useful_positive',ye,yg),[1]);self.assertEqual(eligible('useful_negative',ye,yg),[2]);self.assertEqual(eligible('all',ye,yg),[1,2,3,4,6,7])
 def test_sign_and_mask_gradients(self):
  b=torch.tensor([[.5,.5,.2,.2]],requires_grad=True);t=torch.tensor([[[.5,.5,.2,.2]],[[.55,.5,.2,.2]]],requires_grad=True)
  lp,dp,_=directional_loss(b,t,[1],[1],[1,1]);ln,dn,_=directional_loss(b,t,[-1],[1],[1,1]);gp,=torch.autograd.grad(lp,b,retain_graph=True);gn,=torch.autograd.grad(ln,b)
  self.assertLess(float((gp*gn).sum()),0);self.assertIsNone(t.grad)
  self.assertGreater(float(lp),float(ln))
 def test_exact_count_matching_and_empty(self):
  u=[1,3,5,7];n=[2,4];m=min(len(u),len(n));self.assertEqual(len(hash_subset(u,m,'x')),len(hash_subset(n,m,'x')))
  self.assertEqual(hash_subset(list(reversed(u)),m,'x'),hash_subset(u,m,'x'));self.assertEqual(hash_subset([],0,'x'),[])
if __name__=='__main__':unittest.main()
