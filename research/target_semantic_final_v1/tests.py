import unittest
import numpy as np
from research.target_semantic_final_v1.core import *

class Tests(unittest.TestCase):
    def setUp(self):
        self.p = np.random.default_rng(8).uniform(.05,.8,(24,6))
        q = np.triu(np.ones((24,24)),1)
        self.q = q/q.sum()

    def test_disabled_native_parent(self):
        self.assertTrue(np.array_equal(stable_evidence(self.p,self.q,disabled=True)[0], evidence(self.p,self.q)[0]))

    def test_opposite_parity(self):
        s = stable_evidence(self.p,self.q)[1][0]
        p = self.p.copy(); p[::2] = .99
        t = stable_evidence(p,self.q)[1][0]
        for key in ['reliability','positive','variance']:
            self.assertTrue(np.array_equal(s[key],t[key]))

    def test_flat_noop(self):
        p = np.tile(self.p[0],(24,1))
        L,_ = stable_evidence(p,self.q)
        self.assertTrue(np.array_equal(L,np.zeros(24)))
        self.assertTrue(np.array_equal(update(self.q,inclusive_scores(L)),self.q))

    def test_coordinate_permutation(self):
        a = stable_evidence(self.p,self.q)[0]
        b = stable_evidence(self.p[:,::-1],self.q)[0]
        np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-14)

    def test_jackknife_formula(self):
        w = membership(self.q)[1::2]
        _,_,_,r,s = reliability(self.p[1::2],w)
        v = s['delete_group_eta']
        np.testing.assert_array_equal(s['variance'],.75*((v-v.mean(0))**2).sum(0))
        self.assertTrue(np.all((r>=0)&(r<=1)))

    def test_strength_matched(self):
        target=np.arange(24)/24
        x,_ = match_strength(np.cos(np.arange(24)),target)
        self.assertAlmostEqual(np.linalg.norm(x),np.linalg.norm(target),places=12)

    def test_synthetic_stable_vs_local_accident(self):
        # Alternating event seed repeats across blocks; second coordinate has
        # the same contrast only in one chronological group.
        w=np.tile([.1,.9],16); stable=.1+.6*w
        accident=np.full(32,.4); accident[:8]=stable[:8]
        v=np.stack([stable,accident],1)
        r=reliability(v,w)[3]
        self.assertGreater(r[0],r[1]); self.assertAlmostEqual(r[0],1.)

if __name__=='__main__': unittest.main()
