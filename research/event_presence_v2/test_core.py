import unittest,numpy as np
from research.event_presence_v2.core import conditional_event
class TrackTests(unittest.TestCase):
 def test_pure_nuisance_removed(self):
  rng=np.random.default_rng(9);z=rng.normal(size=(8,15,3));z[:,0,1]=3*z[:,0,0]+2
  v,_=conditional_event(z,[[1]],[[0]],0);np.testing.assert_allclose(v,0,atol=1e-12)
 def test_other_slots_do_not_change_estimate(self):
  z=np.random.default_rng(9).normal(size=(8,15,3));v,a=conditional_event(z,[[1]],[[0]],0);z[:,1:]*=100;w,b=conditional_event(z,[[1]],[[0]],0);np.testing.assert_array_equal(v,w);self.assertEqual(a,b)
 def test_held_out_time_excluded(self):
  z=np.random.default_rng(9).normal(size=(8,15,3));v,a=conditional_event(z,[[1]],[[0]],0);z[2,0,1]+=4;w,b=conditional_event(z,[[1]],[[0]],0);self.assertAlmostEqual(w[2,0]-v[2,0],4);self.assertEqual(a[2],b[2])
if __name__=='__main__':unittest.main()
