"""CPU tests cover leakage and regression semantics, not fixed result values."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='2'
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.linalg import solve
from scripts.tastvg_pr_nested_alpha_math_v1 import *

class Tests(unittest.TestCase):
    def test_nested_source_exclusions(self):
        for f in nested_folds('test',range(6)):
            self.assertNotIn(f['held_source'],f['train_sources'])
            for inner in f['inner_sources']:
                ids=[s for s in f['train_sources'] if s!=inner]
                self.assertNotIn(inner,ids);self.assertNotIn(f['held_source'],ids)
                self.assertEqual(len(ids),4)
    def test_equal_source_weights(self):
        g=[0]*32+[1]*2;w=source_weights(g)
        self.assertAlmostEqual(sum(w[:32]),.5);self.assertAlmostEqual(sum(w[32:]),.5)
    def test_grid_matches_independent_solve(self):
        rng=np.random.default_rng(7);x=rng.normal(size=(50,9));x[:,-1]=3
        y=rng.normal(size=50);g=np.arange(50)%4;w=source_weights(g)
        for m in ridge_grid(x,y,g):
            z=(x-m['mean'])/m['std'];b=float(w@y)
            v=solve(z.T@(w[:,None]*z)+m['alpha']*np.eye(9),z.T@(w*(y-b)),assume_a='pos')
            np.testing.assert_allclose(m['weight'],v,atol=2e-11,rtol=2e-11)
            self.assertEqual(m['std'][-1],1.);self.assertAlmostEqual(m['bias'],b)
    def test_ties_are_smallest_grid(self):
        self.assertEqual(select_alpha([1]*7),.001)
        self.assertEqual(select_alpha([2,2,1,1,3,3,3]),.1)
    def test_nonfinite_selection_rejected(self):
        with self.assertRaises(AssertionError):select_alpha([np.nan]*7)
    def test_inner_objective_roles_conditions_orders(self):
        rows=[dict(source_id=1,order='x',condition='clean'),dict(source_id=1,order='x',condition='drop'),
            dict(source_id=1,order='y',condition='clean'),dict(source_id=1,order='y',condition='drop')]
        z=validation_errors(rows,np.zeros((4,2)),np.array([[0,0],[2,4],[0,0],[6,8.]]))
        self.assertEqual(z['mean'],2.5);self.assertEqual(z['clean'],0);self.assertEqual(z['corrupt'],5.)
    def test_outer_labels_cannot_change_models(self):
        rng=np.random.default_rng(6)
        packs={s:dict(x=rng.normal(size=(1,32,9)),P=rng.random((1,32)),R=rng.random((1,32)),anchors=[0],winners=[1]) for s in range(5)}
        def nested_selection(allpacks,outer):
            train={s:p for s,p in allpacks.items() if s!=outer};scores=[]
            for inner in train:
                x,y,g=subset_arrays({s:p for s,p in train.items() if s!=inner},'all')
                v=train[inner];mods=ridge_grid(x,y['P'],g)
                scores.append([np.mean(abs(predict(m,v['x'][0,[0,1]])-v['P'][0,[0,1]])) for m in mods])
            return select_alpha(np.mean(scores,0))
        before=nested_selection(packs,0)
        packs[0]['P'][:]=1e10;packs[0]['x'][:]=-1e10
        self.assertEqual(before,nested_selection(packs,0))
    def test_file_guard_rejects_outer_labels(self):
        import subprocess
        code='''from scripts.run_tastvg_pr_nested_alpha_v1 import guard
guard('fit', {'paths':set()})
try:
    open('/tmp/source_packs/outer.npz','rb')
except PermissionError:
    print('guard denied')
else:
    raise AssertionError('outer read permitted')
'''
        p=subprocess.run([sys.executable,'-B','-c',code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr);self.assertIn('guard denied',p.stdout)
    def test_noop_and_zero_denominator(self):
        z=analytic_readout(dict(P_A=-3,R_A=-4,P_W=-3,R_W=-4))
        self.assertEqual(z['T_A'],0);self.assertEqual(z['delta_T'],0)
    def test_role_duplicates_preserved(self):
        p=dict(x=np.arange(32*768).reshape(1,32,768),P=np.ones((1,32)),R=np.ones((1,32)),anchors=[3],winners=[3])
        x,y,g=subset_arrays({1:p},'role');self.assertEqual(x.shape,(2,768));np.testing.assert_array_equal(x[0],x[1])

if __name__=='__main__':unittest.main()
