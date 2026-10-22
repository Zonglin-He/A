import unittest,numpy as np
from research.native_event_semantic_prototype_v1.core import *
class CoreTests(unittest.TestCase):
 def setUp(self):
  self.q=np.triu(np.random.default_rng(2).random((9,9)),1);self.q/=self.q.sum()
 def test_membership(self):
  want=np.array([sum(self.q[s,e]for s in range(9)for e in range(9)if s<=t<=e)for t in range(9)])
  np.testing.assert_allclose(membership(self.q),want,atol=1e-15)
 def test_interval_endpoints(self):
  L=np.arange(9)-3;S=inclusive_scores(L)
  for s in range(9):
   for e in range(s,9):self.assertEqual(S[s,e],sum(L[s:e+1]))
 def test_constant_and_label_invariance(self):
  p=np.tile(np.array([.25,.5,.75]),(9,1));q,steps=loop(p,self.q);self.assertTrue(np.array_equal(q,self.q))
  p=np.random.default_rng(3).uniform(.1,.9,(9,4));a,_=loop(p,self.q);b,_=loop(p[:,[2,0,3,1]],self.q);np.testing.assert_allclose(a,b,atol=1e-15)
 def test_cross_fit_isolation(self):
  p=np.random.default_rng(4).uniform(.1,.9,(9,4));_,a=evidence(p,self.q);p[::2]=.9;_,b=evidence(p,self.q)
  for k in ['positive','negative']:self.assertTrue(np.array_equal(a[0][k],b[0][k]))
 def test_synthetic_positive_and_random_histogram(self):
  n=17;q=np.triu(np.ones((n,n)),1);q/=q.sum();q*=.5;q[5,11]+=.5;p=np.full((n,4),.1);p[5:12]=.9
  out,_=loop(p,q);self.assertGreater(out[5,11],q[5,11]);r=matched_random(p,np.random.default_rng(8).normal(size=(n,256)));self.assertTrue(np.array_equal(np.sort(p,axis=0),np.sort(r,axis=0)))
if __name__=='__main__':unittest.main()
