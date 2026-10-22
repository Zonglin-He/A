import unittest,numpy as np
from scripts.joint_boundary_readout.v2_core import select
class V2Tests(unittest.TestCase):
 def test_evidence_control_can_override_prior(self):
  p=dict(ambiguous=True,utilities=np.array([.8,.1]));f=np.array([[0.,0.],[1.,1.]])
  self.assertEqual(select(p,f,.25),0);self.assertEqual(select(p,f,0,'evidence_joint'),1);self.assertEqual(select(p,f,0,'evidence_hidden'),1)
 def test_ungated_and_bound(self):
  p=dict(ambiguous=False,utilities=np.array([.8,.6]));f=np.array([[0.,0.],[1.,1.]])
  self.assertEqual(select(p,f,.25),0);self.assertEqual(select(p,f,.25,gated=False),1);self.assertEqual(select(p,f,.10,gated=False),0)
if __name__=='__main__':unittest.main()
