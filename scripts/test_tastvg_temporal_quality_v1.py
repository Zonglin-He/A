"""Mathematical contracts, temporal support, and absence of artificial preferences."""
import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from vg_tta.tastvg_temporal_quality_old8_v1 import activation,feature_edges,contrast,decide

class Contracts(unittest.TestCase):
    def test_pooled_query_not_token_average(self):
        v=np.zeros((1024,2)); v[0,0]=1; v[1,1]=1
        q=np.zeros((1024,4)); q[0,0]=2; q[1,1:]=100
        s,available=activation(v,q)
        np.testing.assert_array_equal(s,[1,0]); self.assertTrue(available)
    def test_scale_invariance(self):
        rng=np.random.default_rng(10); v=rng.normal(size=(1024,9)); q=rng.normal(size=(1024,3))
        np.testing.assert_allclose(activation(v,q)[0],activation(v*9,q*.02)[0],atol=1e-15)
    def test_fractional_bin_integral(self):
        z=contrast([0,1,0],[0,.25,.75,1],[[.375,.625]])[0]
        self.assertEqual(z['inner_mean'],1); self.assertEqual(z['outer_mean'],1)
        self.assertEqual(z['score'],0)
    def test_peak_boundaries(self):
        z=contrast([0,0,1,1,0,0],np.linspace(0,1,7),[[1/3,2/3],[1/6,5/6]])
        self.assertAlmostEqual(z[0]['score'],1)
        self.assertGreater(z[0]['score'],z[1]['score'])
    def test_full_window_is_neutral_and_retained(self):
        z=contrast([0,1], [0,.5,1], [[0,1]])[0]
        self.assertEqual(z['score'],0); self.assertTrue(z['no_outer_observation'])
    def test_flat_evidence_keeps_old_readout(self):
        z=decide([.4,.4],[0,.5,1],[[0,.5]]*8,6)
        self.assertEqual(z['selected'],6); self.assertEqual(z['fallback_reason'],'constant_semantic_evidence')
    def test_identical_scores_native_first(self):
        z=decide([0,1],[0,.5,1],[[.5,1]]*8,6)
        self.assertEqual(z['selected'],0); self.assertEqual(len(z['details']),8)
    def test_phase_zero_final_bin_and_unavailable(self):
        np.testing.assert_allclose(feature_edges(2,1.3),[0,.5/1.3,1])
        curve,avail=activation(np.zeros((1024,2)),np.zeros((1024,1)))
        z=decide(curve,[0,.5,1],[[0,.5]]*8,3,avail)
        self.assertEqual(z['selected'],3); self.assertEqual(z['fallback_reason'],'unavailable_embedding')

if __name__=='__main__':unittest.main()
