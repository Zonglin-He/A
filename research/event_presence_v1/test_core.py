import unittest,numpy as np
from research.event_presence_v1.core import conditional_event,full_reweight
class ConditionalTests(unittest.TestCase):
 def test_presence_only_exactly_removed(self):
  rng=np.random.default_rng(42);n=rng.normal(size=(8,15));z=np.zeros((8,15,3));z[:,:,0]=n;z[:,:,1]=2*n+3
  y,_=conditional_event(z,[[1]],[[0]],0);np.testing.assert_allclose(y,0,atol=1e-13)
 def test_target_specific_event_retained(self):
  rng=np.random.default_rng(42);n=rng.normal(size=(8,15));z=np.zeros((8,15,3));z[:,:,0]=n;z[:,:,1]=2*n+3;action=np.arange(8)/3;z[:,0,1]+=action
  y,_=conditional_event(z,[[1]],[[0]],0);np.testing.assert_allclose(y[:,0],action,atol=1e-13)
 def test_target_excluded_from_calibration(self):
  rng=np.random.default_rng(1);z=rng.normal(size=(8,15,4));a,da=conditional_event(z,[[1]],[[0]],2);z[:,2,1]+=7;b,db=conditional_event(z,[[1]],[[0]],2)
  np.testing.assert_allclose(b-a,7,atol=1e-13);self.assertEqual(da,db)
 def test_uniform_no_posterior_change(self):
  p=np.array([[0.,.2,.3],[0.,0.,.5],[0.,0.,0.]]);ij=np.stack(np.triu_indices(3,1),-1);np.testing.assert_allclose(full_reweight(p,ij,np.zeros(3)),p,atol=1e-15)
if __name__=='__main__':unittest.main()
