import unittest,numpy as np
from research.native_event_coupling_v1.core import couple
from research.native_event_composition_v1.core import envelope
class Tests(unittest.TestCase):
 def test_two_occurrences(self):
  q=np.zeros((8,8));q[0,1]=q[5,6]=.5;r,info=couple(q,q,list(range(8)));self.assertTrue(np.array_equal(r,q));self.assertEqual(info['nonoverlapping_pair_mass'],0);self.assertEqual(envelope(q,q)[0,6],.5)
 def test_unequal_atoms(self):
  q=np.zeros((8,8));r=q.copy();q[0,1]=.3;q[5,6]=.7;r[1,2]=.6;r[6,7]=.4;out,info=couple(q,r,list(range(8)));expected=q*0;expected[0,2]=.3;expected[1,6]=.3;expected[5,7]=.4;self.assertTrue(np.allclose(out,expected,atol=1e-15));self.assertLess(info['marginal_error_last'],1e-12)
 def test_physical_grid_and_anti(self):
  rng=np.random.default_rng(4);q=np.triu(rng.random((9,9)),1);q/=q.sum();r=np.triu(rng.random((9,9)),1);r/=r.sum();ids=[0,2,7,9,11,40,55,67,100];out,info=couple(q,r,ids,True);self.assertAlmostEqual(out.sum(),1);same,_=couple(q,q,ids);self.assertTrue(np.allclose(same,q,atol=1e-14))
if __name__=='__main__':unittest.main()
