import unittest,numpy as np
from research.native_event_dense_projection_v1.core import project,loop
from research.joint_event_v1.core import ridge_predict
class Tests(unittest.TestCase):
 def test_old_ridge_exact(self):
  x=np.random.default_rng(4).normal(size=(65,1536));y=np.arange(65,dtype=float)/10;got,_=project(x,y)
  for parity in (0,1):
   indices=np.flatnonzero(np.arange(65)%2!=parity);pred,mu,sd=ridge_predict(x,y[indices,None],np.arange(len(indices)),indices);self.assertTrue(np.array_equal(got[np.arange(65)%2==parity],(pred*sd+mu)[np.arange(65)%2==parity,0]))
 def test_constant_exact_native(self):
  x=np.random.default_rng(4).normal(size=(65,10));q=np.triu(np.ones((65,65)),1);q/=q.sum();p=np.full((65,3),.4);z,steps=loop(p,q,x);self.assertTrue(np.array_equal(z,q));self.assertTrue(np.all(steps[0]['raw_L']==0));self.assertTrue(np.all(steps[0]['projected_L']==0))
 def test_heldout_label_isolation(self):
  x=np.random.default_rng(4).normal(size=(65,10));y=np.arange(65,dtype=float);a,_=project(x,y);y[::2]+=100;b,_=project(x,y);self.assertTrue(np.array_equal(a[::2],b[::2]))
 def test_rotation_scale(self):
  rng=np.random.default_rng(4);x=rng.normal(size=(65,12));u,_=np.linalg.qr(rng.normal(size=(12,12)));y=rng.normal(size=65);a,_=project(x,y);b,_=project(13*x@u,y);self.assertLess(abs(a-b).max(),2e-14)
 def test_synthetic_event_positive(self):
  n=65;x=np.c_[np.r_[np.zeros(15),np.ones(31),np.zeros(19)],np.linspace(-1,1,n)];L=6*(x[:,0]-.5);pred,_=project(x,L);self.assertGreater(np.corrcoef(pred,L)[0,1],.99)
if __name__=='__main__':unittest.main()
