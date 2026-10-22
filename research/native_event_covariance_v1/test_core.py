import unittest,numpy as np
from sklearn.covariance._shrunk_covariance import _oas
from research.native_event_covariance_v1.core import *
class Tests(unittest.TestCase):
 def setUp(self):
  self.rng=np.random.default_rng(42);self.p=self.rng.normal(size=(17,5));self.q=np.triu(self.rng.random((17,17)),1);self.q/=self.q.sum()
 def test_oas_reference(self):
  for n,C in [(32,34),(33,60),(17,5),(20,1)]:
   v=self.rng.normal(size=(n,C));a,r=oas(v);b,s=_oas(v);np.testing.assert_allclose(a,b,atol=1e-15);self.assertAlmostEqual(r,s,places=14)
 def test_gaussian_ratio(self):
  covs=covariances(self.p);L,stats=evidence(self.p,self.q,covs)
  for parity in (0,1):
   s=stats[parity];v=self.p[np.arange(17)%2==parity];inv=np.linalg.inv(s['covariance']);a=v-s['positive'];b=v-s['negative'];want=-.5*np.einsum('nc,cd,nd->n',a,inv,a)+.5*np.einsum('nc,cd,nd->n',b,inv,b);np.testing.assert_allclose(L[np.arange(17)%2==parity],want,atol=3e-15)
 def test_constant(self):
  p=np.tile([.123,.456,.789],(17,1));q,_=loop(p,self.q);self.assertTrue(np.array_equal(q,self.q))
 def test_axis_rotation_and_crossfit(self):
  q,_=loop(self.p,self.q);perm,_=loop(self.p[:,[4,2,0,1,3]],self.q);np.testing.assert_allclose(q,perm,atol=3e-15);U=np.linalg.qr(self.rng.normal(size=(5,5)))[0];rot,_=loop(self.p@U,self.q);np.testing.assert_allclose(q,rot,atol=3e-15)
  covs=covariances(self.p);_,a=evidence(self.p,self.q,covs);p=self.p.copy();p[::2]+=20;cc=covariances(p);_,b=evidence(p,self.q,cc)
  for k in ['positive','negative','covariance']:self.assertTrue(np.array_equal(a[0][k],b[0][k]))
 def test_synthetic_correlated_event(self):
  n=25;latent=self.rng.normal(0,.03,(n,2));latent[8:17,0]+=1.;p=latent@self.rng.normal(size=(2,12))+self.rng.normal(0,.01,(n,12));q=np.triu(np.ones((n,n)),1);q/=q.sum();q*=.5;q[8,16]+=.5;out,_=loop(p,q);self.assertGreater(out[8,16],q[8,16])
if __name__=='__main__':unittest.main()
