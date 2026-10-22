import unittest,itertools
import numpy as np
from research.native_event_all_phase_v1.core import all_ordered
from research.native_event_order_v1.core import ordered
class Tests(unittest.TestCase):
 def test_three_phase_bruteforce(self):
  rng=np.random.default_rng(11);n=4;qs=[np.triu(rng.random((n,n)),1)for _ in range(3)];qs=[q/q.sum()for q in qs];pairs=list(zip(*np.triu_indices(n,1)));raw=np.zeros((n,n))
  for choices in itertools.product(pairs,repeat=3):
   if all(choices[i][0]<=choices[i+1][0]and choices[i][1]<=choices[i+1][1]for i in range(2)):
    raw[choices[0][0],choices[-1][1]]+=np.prod([q[s,e]for q,(s,e)in zip(qs,choices)])
  q,mass=all_ordered(qs);self.assertAlmostEqual(mass,raw.sum());np.testing.assert_allclose(q,raw/raw.sum(),atol=1e-15)
 def test_overlapping_events_are_permitted(self):
  phases=[]
  for a,b in [(0,3),(1,4),(2,5)]:
   q=np.zeros((6,6));q[a,b]=1;phases.append(q)
  q,mass=all_ordered(phases);self.assertEqual(mass,1);self.assertEqual(q[0,5],1)
 def test_two_phase_is_exact_archived_operation(self):
  rng=np.random.default_rng(15);qs=[np.triu(rng.random((7,7)),1)for _ in range(2)];qs=[q/q.sum()for q in qs];a,m=all_ordered(qs);b,z=ordered(*qs);self.assertTrue(np.array_equal(a,b));self.assertEqual(m,z)
if __name__=='__main__':unittest.main()
