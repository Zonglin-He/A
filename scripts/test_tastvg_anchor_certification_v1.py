import unittest
import numpy as np
from sklearn.isotonic import IsotonicRegression
from scripts.tastvg_anchor_certification_math_v1 import isotonic,top1,evaluate,decide,summarize

class Tests(unittest.TestCase):
    def test_known_weighted_pool(self):
        k,y=isotonic([0,1,2],[3,1,4],[1,3,1]);np.testing.assert_array_equal(k,[0,1,2]);np.testing.assert_allclose(y,[1.5,1.5,4])
    def test_ties_are_pooled(self):
        k,y=isotonic([0,0,1],[0,1,2]);np.testing.assert_array_equal(k,[0,1]);np.testing.assert_allclose(y,[.5,2])
    def test_random_independent_solver(self):
        rng=np.random.default_rng(7)
        for _ in range(100):
            x=rng.integers(0,12,40);y=rng.normal(size=40);w=rng.integers(0,5,40);k,v=isotonic(x,y,w)
            s=IsotonicRegression(increasing=True,out_of_bounds='clip').fit(x,y,sample_weight=w)
            np.testing.assert_allclose(v,s.predict(k),atol=2e-14,rtol=0)
    def test_ordinary_top1_and_offset(self):
        z=np.zeros(32);z[5]=.3;self.assertEqual(top1(z,0),5);self.assertEqual(top1(z+32,0),5)
    def test_tie_keeps_anchor(self):
        z=np.zeros(32);z[5]=z[6]=.3;self.assertEqual(top1(z,0),0)
    def model(self):return dict(available=True,knots=[.1,.4],mean=[.02,.2],probability=[.6,.9],lower={str(q):[-.01,.1] for q in [.5,.2,.1,.05,.025,.01]},thresholds=[.1,.3])
    def test_no_extrapolation(self):
        self.assertFalse(evaluate(self.model(),.41)['in_domain']);self.assertIsNone(evaluate(self.model(),.09)['lower'])
    def test_gate_does_not_change_ranking(self):
        z=np.zeros(32);z[5]=.11;r=decide(z,0,self.model());self.assertEqual(r['choices']['L32'],5);self.assertEqual(r['choices']['Selective'],0)
        z[5]=.35;r=decide(z,0,self.model());self.assertEqual(r['choices']['Selective'],5)
    def test_no_calibrator_fallback(self):
        z=np.zeros(32);z[1]=1;r=decide(z,0,dict(available=False,thresholds=[]));self.assertEqual(r['choices']['Selective'],0)
    def test_zero_denominator_is_not_perfect_precision(self):
        rows=[dict(source_id=s,order='o',condition='c',accepted=0.,benefit=0.) for s in range(3)]
        z=summarize(rows,['accepted','benefit'],{'precision':('benefit','accepted')});self.assertIsNone(z['ratios']['precision']['mean']);self.assertEqual(z['ratios']['precision']['bootstrap_zero_denominator_draws'],10000)
    def test_source_balance_not_pooled_arrival_mean(self):
        rows=[dict(source_id=0,order='o',condition='c',v=1.)]*10+[dict(source_id=1,order='o',condition='c',v=0.)]
        self.assertEqual(summarize(rows,['v'])['metrics']['v']['mean'],.5)

if __name__=='__main__':unittest.main()
