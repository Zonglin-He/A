"""CPU controls for population membership, weight/penalty and readout semantics."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.linalg import solve
from scripts.tastvg_pr_role_math_v1 import *

class Tests(unittest.TestCase):
    def test_membership_not_score_sorted(self):
        np.testing.assert_array_equal(training_indices(7,2),[7,2])
    def test_noop_retains_two_roles(self):
        np.testing.assert_array_equal(training_indices(4,4),[4,4])
    def test_bad_indices_rejected(self):
        for a,w in [(-1,0),(0,32),(1.5,2)]:
            with self.assertRaises(AssertionError):training_indices(a,w)
    def test_equal_source_and_penalty_scale(self):
        for repeats in [2,32]:
            g=np.repeat([1,1,2],repeats);weights=source_weights(g)
            self.assertAlmostEqual(weights.sum(),1.)
            for s in [1,2]:self.assertAlmostEqual(weights[g==s].sum(),.5)
    def test_weighted_ridge_independent_solve(self):
        rng=np.random.default_rng(19);x=rng.normal(size=(24,8));x[:,3]=2
        y=rng.normal(size=24);g=np.repeat([1,1,2,3],6);m=fit_ridge(x,y,g,.1)
        w=source_weights(g);a=(x-m['mean'])/m['std']
        coef=solve(a.T@(w[:,None]*a)+.1*np.eye(8),a.T@(w*(y-w@y)),assume_a='pos')
        np.testing.assert_allclose(m['weight'],coef,atol=3e-12,rtol=0)
        self.assertEqual(m['std'][3],1.)
    def test_source_replication_invariance(self):
        x=np.array([[1.,2],[2,3],[3,5],[4,6]]);y=np.array([.1,.4,.6,.8]);g=np.array([1,1,2,2])
        a=fit_ridge(x,y,g,1.);b=fit_ridge(np.r_[x,x[:2]],np.r_[y,y[:2]],np.r_[g,g[:2]],1.)
        np.testing.assert_allclose(predict(a,x),predict(b,x),atol=3e-12,rtol=0)
    def test_true_interval_formula(self):
        p,r,t=interval_pr([[0,4],[2,8],[4,6],[8,10]],(3,7))
        np.testing.assert_allclose(analytic_t(p,r),t,atol=1e-15,rtol=0)
    def test_raw_predictions_and_clipped_decision(self):
        z=dict(P_A=-1.,R_A=.5,P_W=2.,R_W=1.5)
        self.assertEqual(analytic_readout(z)['delta_T'],1.)
        self.assertEqual(z['P_A'],-1.)
    def test_noop_never_accepted(self):
        r=dict(source_id=1,order='o',condition='c',eligible=False,delta_t=0.,delta_v=0.)
        z,_=decision([r],[3.],draws=20)
        self.assertEqual(z['counts']['accepted'],0);self.assertEqual(z['metrics'],{})
    def test_constant_truth_r2_undefined(self):
        r=dict(source_id=1,order='o',condition='c')
        z,_=regression([r],[[.5]],[[.4]],draws=20)
        self.assertIsNone(z['metrics']['r2']['mean'])
        self.assertEqual(z['metrics']['r2']['bootstrap_defined'],0)

if __name__=='__main__':unittest.main()
