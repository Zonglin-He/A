import unittest
import numpy as np
from research.native_event_order_v1.core import ordered

class OrderedTests(unittest.TestCase):
    def test_matches_all_ordered_pairs(self):
        rng=np.random.default_rng(5);q=np.triu(rng.random((6,6)),1);r=np.triu(rng.random((6,6)),1);q/=q.sum();r/=r.sum();raw=np.zeros_like(q)
        for a,b in zip(*np.triu_indices(6,1)):
            for c,d in zip(*np.triu_indices(6,1)):
                if a<=c and b<=d:raw[a,d]+=q[a,b]*r[c,d]
        actual,mass=ordered(q,r);self.assertAlmostEqual(mass,raw.sum());np.testing.assert_allclose(actual,raw/raw.sum(),atol=1e-15)
    def test_overlapping_events_allowed(self):
        q=np.zeros((6,6));r=q.copy();q[0,4]=1;r[2,5]=1;out,mass=ordered(q,r);self.assertEqual(mass,1);self.assertEqual(out[0,5],1)

if __name__=='__main__':unittest.main()
