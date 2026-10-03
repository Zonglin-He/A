"""Meaningful source holdout, nesting, normalization and objective controls."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.linalg import solve
from scripts.tastvg_pr_loso_math_v1 import *

class Tests(unittest.TestCase):
    def test_guard_only_allows_extract_write_and_training_members(self):
        self.assertTrue(source_pack_allowed('extract','held','w',set()))
        self.assertFalse(source_pack_allowed('extract','held','r',set()))
        self.assertTrue(source_pack_allowed('extract','written_hash','r',{'written_hash'}))
        self.assertFalse(source_pack_allowed('readout','held','r',{'held'}))
        self.assertFalse(source_pack_allowed('fit','held','r',{'train'}))
        self.assertTrue(source_pack_allowed('fit','train','r',{'train'}))
    def test_every_source_held_once_per_level(self):
        ff=folds('vidstg',range(16))
        self.assertEqual(len(ff),64)
        for f in ff:self.assertNotIn(f['held_source'],f['train_sources'])
        for l in LEVELS:self.assertEqual(sorted(f['held_source'] for f in ff if f['level']==l),list(range(16)))
    def test_nested_and_max_is_N_minus_one(self):
        ff=folds('hc2',range(14))
        for held in range(14):
            f=[r for r in ff if r['held_source']==held]
            self.assertEqual([r['train_source_count'] for r in f],[4,8,12,13])
            for a,b in zip(f,f[1:]):self.assertEqual(a['train_sources'],b['train_sources'][:len(a['train_sources'])])
    def test_fold_identity_order_invariant(self):
        self.assertEqual(folds('v',range(16)),folds('v',reversed(range(16))))
    def test_no_label_or_condition_in_subset_key(self):
        self.assertEqual(folds('v',list(range(16))+[2]),folds('v',range(16)))
    def test_role_duplicate_kept(self):
        np.testing.assert_array_equal(population_indices('role',7,7),[7,7])
        self.assertEqual(len(population_indices('all',7,7)),32)
    def test_normalizer_training_only_extreme_holdout(self):
        x=np.array([[1.,3.],[2.,4.],[4.,7.],[5.,8.]])
        m=fit_ridge(x,np.array([.1,.2,.5,.7]),[1,1,2,2],1)
        self.assertAlmostEqual(m['mean'][0],3.)
        raw=np.array([[1e9,-1e9]])
        p=predict(m,raw);self.assertTrue(np.isfinite(p).all())
        self.assertAlmostEqual(m['mean'][0],3.)
    def test_training_source_equal_weight(self):
        g=[1]*2+[2]*32+[3]*6;w=source_weights(g)
        for s in [1,2,3]:self.assertAlmostEqual(w[np.array(g)==s].sum(),1/3)
        self.assertAlmostEqual(w.sum(),1.)
    def test_independent_ridge_solution_and_constant_column(self):
        rng=np.random.default_rng(71);x=rng.normal(size=(30,7));x[:,2]=4
        y=rng.normal(size=30);g=np.repeat([1,1,2,3,4],6);m=fit_ridge(x,y,g,.1)
        w=source_weights(g);a=(x-m['mean'])/m['std']
        q=solve(a.T@(w[:,None]*a)+.1*np.eye(7),a.T@(w*(y-w@y)),assume_a='pos')
        np.testing.assert_allclose(q,m['weight'],atol=2e-12,rtol=0)
        self.assertEqual(m['std'][2],1.)
    def test_physical_interval_F_identity(self):
        p,r,t=interval_pr([[0,4],[2,9],[7,10]],(3,8))
        from scripts.tastvg_pr_accessibility_math_v1 import analytic_t
        np.testing.assert_allclose(analytic_t(p,r),t,atol=1e-15)
    def test_splits_not_row_LOO(self):
        ff=folds('v',range(16));f=ff[0]
        rows=[(s,c,o) for s in range(16) for c in range(6) for o in range(2)]
        train=[r for r in rows if r[0] in f['train_sources']]
        self.assertFalse(any(r[0]==f['held_source'] for r in train))

if __name__=='__main__':unittest.main()
