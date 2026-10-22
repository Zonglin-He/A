import unittest,numpy as np
from scripts.interval_state_readout.action_core import likelihood,choose,posterior_sd
class ActionTests(unittest.TestCase):
 def test_loss_offset(self):
  eta=.7;c=.3;weighted=eta/(eta+c*(1-eta));z=np.log(weighted/(1-weighted));p=1/(1+np.exp(-(z+np.log(c))));self.assertAlmostEqual(p,eta)
 def test_whole_event(self):
  ids=np.arange(10);logits=np.r_[-np.ones(3)*5,np.ones(4)*5,-np.ones(3)*5];s=likelihood(ids,[[0,10],[2.5,6.5]],logits);self.assertGreater(s[1],s[0])
 def test_no_spatial_evidence(self):self.assertTrue(np.array_equal(likelihood([0,1,2],[[0,3],[0,1]],[-1,2,-1],[0,0,0]),[0,0]))
 def test_prior_risk_and_tie(self):
  self.assertEqual(choose([.8,.6],[0,1],.1),0);self.assertEqual(choose([.8,.6],[0,1],.1,False),1);self.assertEqual(choose([.8,.6],[1,1],.5,False),0)
 def test_certain_posterior_sd(self):
  q=np.zeros((3,3));q[0,2]=1;self.assertEqual(posterior_sd([0,1,2],q,[0,3]),0)
if __name__=='__main__':unittest.main()
