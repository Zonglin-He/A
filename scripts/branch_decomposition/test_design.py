import unittest,numpy as np
from scripts.branch_decomposition.lightweight_design import physical_view,single_high_view_fusion
class GenericDesignTests(unittest.TestCase):
 def test_physical_grid(self):
  q=physical_view([0,4,8],[0,1],[1,2],[.3,.7],[0,2,4,6,8]);self.assertEqual(q[0,2],.3);self.assertEqual(q[2,4],.7);self.assertEqual(q.sum(),1)
 def test_low_identity(self):
  q=physical_view([0,4,8],[0,1],[1,2],[.3,.7],[0,2,4,6,8]);np.testing.assert_array_equal(single_high_view_fusion(q,q),q)
 def test_no_new_support(self):
  a=physical_view([0,4,8],[0],[2],[1.],[0,2,4,6,8]);b=physical_view([2,6],[0],[1],[1.],[0,2,4,6,8]);q=single_high_view_fusion(a,b);self.assertEqual(q[0,4],.5);self.assertEqual(q[1,3],.5);self.assertEqual(np.count_nonzero(q),2)
if __name__=='__main__':unittest.main()
