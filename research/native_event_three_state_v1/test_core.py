import unittest,numpy as np
from research.native_event_three_state_v1.core import *
from research.native_event_semantic_prototype_v1.core import membership
class Tests(unittest.TestCase):
 def setUp(self):
  self.q=np.triu(np.random.default_rng(1).random((11,11)),1);self.q/=self.q.sum();self.p=np.random.default_rng(5).uniform(.1,.9,(11,4))
 def test_membership(self):
  w=state_membership(self.q);np.testing.assert_allclose(w.sum(1),1,atol=1e-15);np.testing.assert_allclose(w[:,1],membership(self.q),atol=1e-15)
  for t in range(11):
   self.assertAlmostEqual(w[t,0],sum(self.q[s,e]for s in range(t+1,11)for e in range(11)));self.assertAlmostEqual(w[t,2],sum(self.q[s,e]for s in range(11)for e in range(t)))
 def test_full_video_likelihood(self):
  S,stats=evidence(self.p,self.q);LL=stats['loglikelihood']
  for s in range(11):
   for e in range(s+1,11):self.assertAlmostEqual(S[s,e],LL[:s,0].sum()+LL[s:e+1,1].sum()+LL[e+1:,2].sum()-LL[:,0].sum())
 def test_constant(self):
  p=np.tile([.123,.54,.765],(11,1));q,_=loop(p,self.q);self.assertTrue(np.array_equal(q,self.q))
 def test_parity_and_class_invariance(self):
  _,a=evidence(self.p,self.q);p=self.p.copy();p[::2]=.9;_,b=evidence(p,self.q);self.assertTrue(np.array_equal(a['prototypes'][0]['prototypes'],b['prototypes'][0]['prototypes']));q,_=loop(self.p,self.q);z,_=loop(self.p[:,[2,0,3,1]],self.q);np.testing.assert_allclose(q,z,atol=2e-15)
 def test_synthetic_order(self):
  p=np.repeat([[.9,.1,.1],[.1,.9,.1],[.1,.1,.9]],7,axis=0);n=len(p);q=np.triu(np.ones((n,n)),1);q/=q.sum();q*=.2;q[7,13]+=.8;z,_=loop(p,q);self.assertGreater(z[7,13],q[7,13])
if __name__=='__main__':unittest.main()
