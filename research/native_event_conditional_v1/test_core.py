import unittest,numpy as np
from research.native_event_conditional_v1.core import duration_conditional
class Tests(unittest.TestCase):
 def test_preserves_every_native_span_mass_and_shifts_locations(self):
  q=np.array([[0,.1,.2,.2],[0,0,.1,.2],[0,0,0,.2],[0,0,0,0.]])
  t=np.array([[0,.3,.01,.01],[0,0,.3,.01],[0,0,0,.37],[0,0,0,0.]])
  out,e=duration_conditional(q,t)
  for l in [1,2,3]:self.assertAlmostEqual(np.diag(out,l).sum(),np.diag(q,l).sum())
  self.assertNotAlmostEqual(out[0,1],q[0,1]);self.assertLess(e,1e-14)
 def test_native_exact_noop_and_zero_support(self):
  q=np.array([[0,.1,0,.4],[0,0,.2,0],[0,0,0,.3],[0,0,0,0.]])
  out,e=duration_conditional(q,q.copy());self.assertTrue(np.array_equal(out,q));self.assertEqual(e,0)
  t=np.triu(np.ones((4,4)),1)/6;out,e=duration_conditional(q,t);self.assertFalse(out[q==0].any())
 def test_teacher_span_mass_does_not_act_as_duration_evidence(self):
  q=np.triu(np.ones((4,4)),1)/6;t=q.copy();t[np.triu_indices(4,2)]*=4;t/=t.sum();out,_=duration_conditional(q,t);self.assertTrue(np.allclose(out,q))
if __name__=='__main__':unittest.main()
