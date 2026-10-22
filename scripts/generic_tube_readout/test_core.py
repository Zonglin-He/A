import unittest,numpy as np
from scripts.generic_tube_readout.core import consensus,expected_overlap,select
class TubeTests(unittest.TestCase):
 def test_constant_restores_temporal(self):
  ids=np.array([0,4,8]);q=np.zeros((3,3));q[0,1]=.4;q[1,2]=.6
  for a,b in [[0,5],[4,9],[0,9]]:
   expected=0
   for s,e,p in [(0,5,.4),(4,9,.6)]:
    inter=max(0,min(b,e)-max(a,s));expected+=p*inter/((b-a)+(e-s)-inter)
   self.assertAlmostEqual(expected_overlap(ids,q,[a,b],np.ones(3)),expected)
 def test_spatial_hypothesis_agreement(self):
  track=np.array([[.5,.5,.2,.2]]*3);hyp=np.repeat(track[:,None,:],3,axis=1);self.assertTrue(np.allclose(consensus(track,hyp,np.ones((3,3),bool)),1))
 def test_no_quality(self):
  q=np.zeros((3,3));q[0,2]=1;self.assertEqual(expected_overlap([0,1,2],q,[0,3],[0,0,0]),0)
 def test_tie(self):self.assertEqual(select([.4,.4],[.8,.7]),0)
if __name__=='__main__':unittest.main()
