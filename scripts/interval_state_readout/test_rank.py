import unittest,numpy as np
from scripts.interval_state_readout.rank_core import likelihood
class RankTests(unittest.TestCase):
 def test_monotone_offset_invariant(self):
  ids=np.arange(8);x=np.array([-3,-2,1,2,1,-1,-2,-4.]);spans=[[0,8],[1.5,4.5],[3,7]];a=likelihood(ids,spans,x);self.assertTrue(np.allclose(a,likelihood(ids,spans,x+100)));self.assertTrue(np.allclose(a,likelihood(ids,spans,np.exp(x))))
 def test_event_and_ties(self):
  ids=np.arange(8);x=np.array([0,0,1,1,1,0,0,0]);s=likelihood(ids,[[0,8],[1.5,4.5]],x);self.assertEqual(s[0],.5);self.assertEqual(s[1],1)
 def test_no_weight(self):self.assertEqual(likelihood([0,1,2],[[0,3]],[1,2,3],[0,0,0])[0],.5)
if __name__=='__main__':unittest.main()
