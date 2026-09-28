import unittest
import numpy as np
from vg_tta.desta3d_v3_direction_alignment import pair_stats, sign_gate


class DirectionAuditTest(unittest.TestCase):
    def test_sign_and_scale(self):
        x=np.array([1.,2.,-3.]); y=-7*x
        r=pair_stats(x,y,chunk=2)
        self.assertAlmostEqual(r['cosine'],-1.)
        self.assertAlmostEqual(r['dot'],-98.)
        self.assertAlmostEqual(pair_stats(x,2*y)['dot'],2*r['dot'])

    def test_zero_and_invalid(self):
        self.assertIsNone(pair_stats(np.zeros(3),np.ones(3))['cosine'])
        for x,y in [(np.zeros(3),np.zeros(4)),(np.array([]),np.array([])),
                    (np.array([np.nan]),np.ones(1))]:
            with self.assertRaises(ValueError): pair_stats(x,y)

    def test_qr_chain(self):
        rng=np.random.default_rng(20260928)
        q,_=np.linalg.qr(rng.normal(size=(17,4)))
        g=rng.normal(size=(5,17)); p=rng.normal(size=(5,4)); s=np.sqrt(17/4)
        self.assertAlmostEqual(pair_stats(g,(p@q.T)*s)['dot'],
                               pair_stats((g@q)*s,p)['dot'],places=12)

    def test_strict_conditional_gate(self):
        self.assertTrue(sign_gate(1,-2,-3))
        self.assertFalse(sign_gate(1,-2,3))
        self.assertFalse(sign_gate(-1,-2,-3))
        self.assertFalse(sign_gate(1,0,-3))


if __name__=='__main__': unittest.main()
