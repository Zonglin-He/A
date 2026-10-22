import unittest,numpy as np
from scripts.interval_state_readout.core import detrend,segment_gain,choose
class StateTests(unittest.TestCase):
 def test_constant(self):self.assertTrue(np.allclose(segment_gain(np.ones((8,3)),np.arange(8),[[1,6],[2,5]]),0))
 def test_piecewise(self):
  ids=np.arange(12);x=np.zeros((12,2));x[4:8,0]=1;g=segment_gain(x,ids,[[4,8],[1,3]]);self.assertGreater(g[0],g[1])
 def test_physical_weights(self):
  x=np.array([[0.],[1.],[0.]]);g=segment_gain(x,[0,10,30],[[5,20],[0,31]]);self.assertAlmostEqual(g[0],1);self.assertAlmostEqual(g[1],0)
 def test_no_tradeoff_weight(self):
  self.assertEqual(choose([.9,.2,.1],[.5,.6,.8],[.5,.8,.4]),1);self.assertEqual(choose([.9,.2],[.5,.8],[.5,.4]),0)
 def test_utility_tie(self):self.assertEqual(choose([.9,.7,.8],[0,1,1],[0,1,1]),2)
 def test_detrend(self):
  t=np.linspace(0,1,20);x=np.tile([1.,2.],(20,1));self.assertLess(abs(detrend(x,t)).max(),1e-12)
if __name__=='__main__':unittest.main()
