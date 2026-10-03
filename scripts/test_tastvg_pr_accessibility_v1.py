"""Meaningful CPU controls for fitted readout and diagnostic decision."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.linalg import solve
from scripts.tastvg_pr_accessibility_math_v1 import *

class Tests(unittest.TestCase):
    def test_interval_identity(self):
        p,r,t=interval_pr([[0,4],[2,8],[4,6],[8,10]],(3,7))
        np.testing.assert_allclose(analytic_t(p,r),t,atol=1e-15,rtol=0)
    def test_empty_and_clipping(self):
        np.testing.assert_equal(analytic_t([0,-2,2,.5],[0,.7,3,1]),[0,0,1,.5])
    def test_monotonicity(self):
        a=np.linspace(0,1,50)
        self.assertTrue(np.all(np.diff(analytic_t(a,.6))>=0))
        self.assertTrue(np.all(np.diff(analytic_t(.6,a))>=0))
    def test_raw_not_clipped(self):
        v=regression_moments([.2,.8],[-.2,1.2])
        self.assertAlmostEqual(v[6],.4)
        self.assertAlmostEqual(v[7],.5)
        self.assertAlmostEqual(v[8],.5)
    def test_ridge_independent_system(self):
        rng=np.random.default_rng(9);x=rng.normal(size=(15,6));x[:,2]=4
        y=rng.normal(size=15);g=[1]*8+[2]*4+[3]*3;m=fit_ridge(x,y,g,.3)
        w=source_weights(g);a=(x-m['mean'])/m['std'];v=solve(a.T@(w[:,None]*a)+.3*np.eye(6),a.T@(w*(y-w@y)),assume_a='pos')
        np.testing.assert_allclose(m['weight'],v,atol=3e-12,rtol=0)
        self.assertAlmostEqual(m['std'][2],1.)
    def test_source_replication_invariance(self):
        x=np.array([[1.,2],[2,3],[3,5],[4,6]]);y=np.array([.1,.4,.6,.8]);g=np.array([1,1,2,2])
        a=fit_ridge(x,y,g,1.)
        b=fit_ridge(np.concatenate([x,x[:2]]),np.r_[y,y[:2]],np.r_[g,g[:2]],1.)
        np.testing.assert_allclose(predict(a,x),predict(b,x),atol=3e-12,rtol=0)
    def test_noop_and_undefined(self):
        r=dict(source_id=1,order='o',condition='c',eligible=False,delta_t=0.,delta_v=0.)
        s,bs=decision([r],[3.],draws=20)
        self.assertEqual(s['counts']['accepted'],0)
        self.assertEqual(s['metrics'],{})
        self.assertEqual(s['utility']['delta_t']['mean'],0.)
    def test_ladder_positive_control(self):
        truth=dict(P_A=.6,R_A=.5,P_W=.8,R_W=.7)
        m=ladder(dict.fromkeys(ROLES,-1.),truth,'GT_all')
        self.assertGreater(m['delta_T'],0.)
        self.assertEqual(ladder(truth,truth,'predicted'),m)

if __name__=='__main__':unittest.main()
