"""Analytic contracts for the proposed consensus and native-only readout."""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from vg_tta.tastvg_temporal_router_t0_v1 import route, proposal_quality, quantile_indices


class Contracts(unittest.TestCase):
    def test_isolated_and_cluster(self):
        a,q,s = proposal_quality([[0,2],[0,2],[3,4]], [.2,.2,1.])
        np.testing.assert_array_equal(a,[.5,.5,0])
        np.testing.assert_array_equal(q,[.1,.1,0])
        self.assertEqual(s.tolist(),[1.,1.,0.])

    def test_duplicate_preservation(self):
        c=[dict(physical_interval=[0,2]),dict(physical_interval=[3,4])]
        z=route(c,[0,1,2,3],[[0,2],[0,2],[3,4]],[.2,.2,1.])
        self.assertEqual(z['duplicate_proposals'],1)
        self.assertEqual(z['current_selected'],1)
        self.assertEqual(z['qc_selected'],0)
        np.testing.assert_array_equal(z['weights']['E'],[1,1,0,0])
        np.testing.assert_allclose(z['weights']['SE'],[.75,.75,0,.25])

    def test_missing_consensus(self):
        for p,c in [([],[]),([[0,2]],[1.]),([[0,1],[2,3]],[1.,1.])]:
            z=route([dict(physical_interval=[0,3])],[0,1,2],p,c)
            self.assertFalse(z['expert_available'])
            self.assertIsNone(z['quantiles']['E'])
            self.assertEqual(z['qc_selected'],0)
            np.testing.assert_array_equal(z['weights']['SE'],[.5,.5,.5])

    def test_quantiles_and_bounds(self):
        self.assertEqual(quantile_indices([0,1,0,1]),[1,1,1,3,3])
        self.assertEqual(quantile_indices([0,2,0,2]),[1,1,1,3,3])
        with self.assertRaises(AssertionError):
            proposal_quality([[1,0]],[1.])
        with self.assertRaises(AssertionError):
            proposal_quality([[0,1]],[-1.])


if __name__ == '__main__':unittest.main()
