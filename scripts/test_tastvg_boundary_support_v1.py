"""Meaningful contracts for support preservation and event boundary execution."""
import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from vg_tta.tastvg_temporal_boundary_support_v1 import expanded,boundary,decisions

class Contracts(unittest.TestCase):
    def test_preserves_actual_old_selection_and_deterministic_prefix(self):
        ids=[0,2,5,9,14,20,27,35,44,54]
        pairs=[[0,9],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8]]
        old=[dict(indices=p,physical_interval=[ids[p[0]],ids[p[1]]+1],origin='old') for p in pairs]
        a=expanded(old,ids);self.assertEqual(a[:8],old);self.assertEqual(a,expanded(old,ids))
        self.assertEqual(len({tuple(z['indices']) for z in a}),32)
        self.assertNotEqual(a[5],a[0]);self.assertEqual(a[5],old[5])
    def test_step_event_requires_two_edges(self):
        curve=[0,0,1,1,1,1,0,0];edges=np.arange(9)/8
        x=boundary(curve,edges,[[.25,.75],[.25,.5],[.5,.75]],4)
        self.assertEqual(x[0]['score'],1);self.assertEqual(x[1]['score'],0);self.assertEqual(x[2]['score'],0)
    def test_missing_context_neutral_and_retained(self):
        x=boundary([1,1],[0,.5,1],[[0,1],[0,.5]],2)
        self.assertEqual(x[0]['score'],0);self.assertTrue(x[0]['missing_start_context'])
        self.assertTrue(x[0]['missing_end_context']);self.assertEqual(len(x),2)
    def test_flat_and_unavailable_keep_A(self):
        intervals=[[0,1],[.25,.75]]
        for avail in [True,False]:
            d=decisions([.1,.1],[0,.5,1],intervals,2,1,avail)
            self.assertTrue(all(z['selected']==1 for z in d.values()))
    def test_fractional_bins_short_interval(self):
        z=boundary([0,1,0],[0,.3,.7,1],[[.4,.6]],3)[0]
        self.assertAlmostEqual(z['means'][0],1);self.assertAlmostEqual(z['means'][2],1)
        self.assertAlmostEqual(z['lengths'][0],.2)
    def test_only_one_strong_boundary_minimum(self):
        z=boundary([0,1,1,1],[0,.25,.5,.75,1],[[.25,.75]],4)[0]
        self.assertEqual(z['start_transition'],1);self.assertEqual(z['end_transition'],0)
        self.assertEqual(z['score'],0)

if __name__=='__main__':unittest.main()
