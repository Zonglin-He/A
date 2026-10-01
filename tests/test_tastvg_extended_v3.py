import unittest
from unittest.mock import patch
import torch
from vg_tta.tastvg_extended_method_v3 import reverse_kl,nested_directions,rollout_states,OnlineMethod,SingleStep
from vg_tta.tastvg_optuna_method_v1 import reverse_kl as prior_loss
from vg_tta.tastvg_native_spatial_rollout_s05_v1 import directions,rollout_states as prior_states

class ExtendedTest(unittest.TestCase):
 def test_exact_loss_gradient(self):
  torch.manual_seed(12)
  b=torch.rand(4,4,requires_grad=True);c=torch.rand(9,4,4);r=torch.arange(9,dtype=torch.float64)
  a=prior_loss(b,c,r,[5,2],.349);x=reverse_kl(b,c,r,[5,2],.349,1.)
  self.assertTrue(all(torch.equal(u,v) for u,v in zip(a,x)))
  self.assertTrue(torch.equal(torch.autograd.grad(a[0],b)[0],torch.autograd.grad(x[0],b)[0]))
  self.assertFalse(torch.equal(x[1],reverse_kl(b,c,r,[5,2],.349,.1)[1]))
 def test_nested_basis(self):
  base=nested_directions(1792,16)
  self.assertTrue(torch.equal(base[:4],directions()))
  self.assertTrue(torch.allclose(base@base.T,torch.eye(16,dtype=torch.float64),atol=1e-12))
  for d in [1,2,4,8,16]:self.assertTrue(torch.equal(base[:d],nested_directions(1792,d)))
 def test_default_candidate_parity(self):
  center={'a':torch.ones(1792)}
  old=prior_states(center,.05)[0];new=rollout_states(center,.05,4)[0]
  self.assertEqual(len(old),len(new))
  self.assertTrue(all(torch.equal(x['a'],y['a']) for x,y in zip(old,new)))
 def test_multistep_seal_and_cache(self):
  actor=OnlineMethod.__new__(OnlineMethod);actor.steps=3;actor.fast=True;actor.deltas=list(range(9))
  calls=[];seen=[]
  def step(self,data,scheduled,tp,sp):
   i=len(seen);seen.append(i)
   if self.fast:tp()
   sp()
   v=torch.tensor(float(i));w=v+1
   return dict(pre_state={'a':v},post_state={'a':w},pre_state_sha256=str(i),post_state_sha256=str(i+1),updated=True,update={'x':i},prediction=i,output_prediction=i,post_prediction=i+1,displacement_from_source=i+1),None
  with patch.object(SingleStep,'arrive',step):
   x,_=actor.arrive(None,True,lambda:calls.append('T'),lambda:calls.append('S'))
  self.assertEqual(calls,['T','S']);self.assertEqual(x['output_prediction'],0)
  self.assertEqual(x['post_state_sha256'],'3');self.assertEqual(len(x['update_steps']),3)
  self.assertEqual(x['compute']['backward_calls'],3);self.assertEqual(x['compute']['spatial_candidate_replays'],27)

if __name__=='__main__':unittest.main()
