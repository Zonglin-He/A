"""Resource-free tests of scientific edge cases and aggregation units."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.tastvg_large_evidence_math_v1 import *

class Cases(unittest.TestCase):
    def test_extreme_logits_without_underflow(self):
        z=[np.array([[0.,-10000.],[-10000.,0.]]),np.zeros((2,2))]
        p=merged_logprior(z,[[0,2],[1,3]],4)
        self.assertTrue(np.isfinite(p).all());self.assertLess(p[2,0],-9999)
        np.testing.assert_allclose(np.exp(p).sum(0),1,atol=1e-12)
    def test_density_known_one_proposal_and_units(self):
        x=[[1.,3.],[2.,4.]];p=[[1.,3.]]
        np.testing.assert_allclose(boundary_density(x,p,1.),[[0.,0.],[-.5,-.5]])
        np.testing.assert_allclose(boundary_density(np.array(x)*30+7,np.array(p)*30+7,30),boundary_density(x,p,1.))
    def test_empty_evidence_and_ties_cannot_force_replacement(self):
        self.assertIsNone(boundary_density([[0,1]],[],.5))
        self.assertEqual(choose([2,5,5],0,3),0)
        self.assertEqual(choose([2,1,2],0,3),0)
        self.assertEqual(choose([2,1,4],0,3),2)
    def test_one_sided_trim_not_whole_interval_shift(self):
        g=geometry([.1,.9],[.101,.4]);self.assertEqual(g['kind'],'trim_end');self.assertTrue(g['large'])
        self.assertEqual(geometry([.1,.4],[.5,.8])['kind'],'shift')
        self.assertEqual(geometry([.1,.9],[.2,.8])['kind'],'trim_both')
    def test_undefined_auc_does_not_become_chance(self):
        z=binary_cell([0,1,2],[0,3,4],[1,2]);self.assertIsNone(z['auc'])
        z=binary_cell([0,1,-1],[0,1,1],[1,2]);self.assertEqual(z['auc'],.5)
    def test_no_candidate_pair_pseudoreplication(self):
        rr=[dict(source_id=0,order='o1',condition='a',x=0.),dict(source_id=0,order='o1',condition='b',x=0.),
            dict(source_id=1,order='o1',condition='a',x=1.)]
        z=aggregate(rr,['x']);self.assertEqual(z['metrics']['x']['mean'],.5)
        self.assertEqual(z['sources'],2)
    def test_zero_acceptance_precision_is_undefined(self):
        rr=[dict(source_id=0,order='o',condition='c',tp=0.,den=0.)]
        z=aggregate(rr,['tp','den'],{'precision':('tp','den')})['ratios']['precision']
        self.assertIsNone(z['mean']);self.assertEqual(z['bootstrap_zero_denominator_draws'],10000)

if __name__=='__main__':unittest.main()
