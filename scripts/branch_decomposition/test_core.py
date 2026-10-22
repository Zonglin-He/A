import unittest,numpy as np
from scripts.branch_decomposition.run import utility
from stvg.temporal import final_mbr
class BranchTests(unittest.TestCase):
 def test_inverse_blend(self):
  g=np.array([[0,.1,.8],[0,0,.1],[0,0,0]]);h=np.array([[0,.6,.2],[0,0,.2],[0,0,0]]);s=.5*g+.5*h;self.assertTrue(np.allclose(2*s-g,h,rtol=0,atol=1e-15))
 def test_same_decision_not_same_posterior(self):
  g=np.zeros((3,3));g[0,2]=1;h=.9*g;h[0,1]=.1;self.assertEqual(final_mbr([0,1,2],g)['interval'],final_mbr([0,1,2],h)['interval']);self.assertFalse(np.array_equal(g,h))
 def test_utility_linear(self):
  g=np.array([[0,.1,.8],[0,0,.1],[0,0,0]]);h=np.array([[0,.6,.2],[0,0,.2],[0,0,0]]);ids=[0,3,7]
  for span in [[0,8],[3,8],[0,4]]:self.assertAlmostEqual(utility(ids,.5*g+.5*h,span),.5*utility(ids,g,span)+.5*utility(ids,h,span))
if __name__=='__main__':unittest.main()
