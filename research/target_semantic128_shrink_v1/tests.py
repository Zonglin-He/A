import unittest,numpy as np
from research.target_semantic128_shrink_v1.core import shrunk_evidence,one_step
from research.native_event_semantic_prototype_v1.core import evidence,loop
class Test(unittest.TestCase):
 def setUp(self):
  self.p=np.random.default_rng(7).uniform(.1,.9,(14,5));q=np.triu(np.ones((14,14)),1);self.q=q/q.sum()
 def test_original_exact(self):
  a,b=evidence(self.p,self.q);c,d=shrunk_evidence(self.p,self.q,0);self.assertTrue(np.array_equal(a,c));self.assertTrue(np.array_equal(one_step(self.p,self.q)[0],loop(self.p,self.q)[1][0]['q']))
 def test_formula_train_only(self):
  L,s=shrunk_evidence(self.p,self.q,4)
  from research.native_event_semantic_prototype_v1.core import membership
  w=membership(self.q)
  for x in s:
   take=np.arange(14)%2!=x['heldout_parity'];v=self.p[take];mu=v.mean(0)
   plus=(w[take,None]*v).sum(0)+4*mu;plus/=w[take].sum()+4
   self.assertTrue(np.allclose(plus,x['positive'],atol=2e-16,rtol=0))
 def test_shrink_distance(self):
  a=shrunk_evidence(self.p,self.q,1)[1];b=shrunk_evidence(self.p,self.q,16)[1]
  for x,y in zip(a,b):
   for name in ['positive','negative']:self.assertLess(np.linalg.norm(y[name]-y['mu0']),np.linalg.norm(x[name]-x['mu0']))
 def test_constant_and_beta_zero(self):
  flat=np.tile(self.p[0],(14,1));self.assertTrue(np.array_equal(one_step(flat,self.q,4)[0],self.q));self.assertTrue(np.array_equal(one_step(self.p,self.q,beta=0)[0],self.q))
 def test_permutation_and_no_stack(self):
  self.assertTrue(np.allclose(one_step(self.p[:,::-1],self.q,4)[0],one_step(self.p,self.q,4)[0],atol=2e-17,rtol=1e-14))
  with self.assertRaises(AssertionError):one_step(self.p,self.q,4,.5)
if __name__=='__main__':unittest.main()
