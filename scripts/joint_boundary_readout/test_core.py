import unittest,numpy as np
from scripts.joint_boundary_readout.core import mean_curve,boundary_features,select,bounded_curve,iou
class CoreTests(unittest.TestCase):
 def test_physical_integral(self):self.assertAlmostEqual(mean_curve(np.array([0,1,3]),np.array([0,1,3]),.5,2.5),1.5);self.assertIsNone(mean_curve(np.array([0,1]),np.zeros(2),2,3))
 def test_constant_not_boundary(self):
  f=boundary_features(np.array([[2.,6.]]),np.arange(9.),np.ones(9),np.ones(9));self.assertEqual(f[0,1],0)
 def test_event_step_and_permutation(self):
  t=np.arange(0,10,.1);s=((t>=3)&(t<=7)).astype(float);h=np.exp(-((t-3)/.2)**2)+np.exp(-((t-7)/.2)**2)
  f=boundary_features(np.array([[3.,7.],[1.,9.]]),t,h,s);self.assertGreater(f[0,1],f[1,1]);bad=boundary_features(np.array([[3.,7.]]),t,h,-s);self.assertEqual(bad[0,1],0)
 def test_gate_and_zero(self):
  p=dict(ambiguous=False,utilities=np.array([.6,.58]));f=np.array([[0,0],[1,1]])
  self.assertEqual(select(p,f,.1),0);self.assertEqual(select(p,f,.1,gated=False),1);self.assertEqual(select(p,f,0,gated=False),0)
 def test_utility_bound(self):
  p=dict(ambiguous=True,utilities=np.array([.6,.3]));f=np.array([[0,0],[1,1]]);self.assertEqual(select(p,f,.1),0)
 def test_iou(self):self.assertEqual(float(iou([0,2],[1,3])),1/3);self.assertEqual(float(iou([0,1],[1,2])),0)
if __name__=='__main__':unittest.main()
