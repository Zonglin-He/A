"""Mathematical contracts for causal prefix and zero/source handling."""
import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.decota_ln_spectrum_math_v1 import *

class Contracts(unittest.TestCase):
    def test_exact_rank_and_zero(self):
        x=np.array([[2.,0],[1,0],[0,0]]);m=[dict(source_id=i) for i in range(3)]
        q=spectrum(x@x.T,m,'raw');self.assertEqual(q['zero_cells'],1);self.assertEqual(q['energy']['1'],1)
    def test_norm_dominance_control(self):
        x=np.diag([100.,1]);m=[dict(source_id=i) for i in range(2)]
        self.assertGreater(spectrum(x@x.T,m,'raw')['energy']['1'],.99)
        self.assertEqual(spectrum(x@x.T,m,'unit')['energy']['1'],.5)
    def test_mean_can_create_low_rank(self):
        x=np.array([[10,1.],[10,-1]]);m=[dict(source_id=i) for i in range(2)]
        q=spectrum(x@x.T,m,'raw');self.assertGreater(q['mean_direction_fraction'],.99)
        self.assertEqual(spectrum(x@x.T,m,'centered')['energy']['1'],1)
    def test_prior_only_prefix(self):
        x=np.eye(3);s=dict(dataset='toy',stream='online100',split='search',condition='clean',order='order1',
            rows=[dict(arrival=i,source_id=i,commit_norm=1) for i in range(3)],gram=(x@x.T).tolist())
        p=prefix(s);self.assertIsNone(p[0]['energy']['1']);self.assertEqual(p[1]['energy']['1'],0);self.assertEqual(p[2]['energy']['2'],0)
    def test_rounded_zero_is_not_a_write(self):
        s=dict(dataset='toy',stream='online100',split='search',condition='clean',order='order1',
            rows=[dict(arrival=i,source_id=i,commit_norm=float(i)) for i in range(2)],gram=np.eye(2).tolist())
        q=prefix(s);self.assertEqual(len(q),1);self.assertEqual(q[0]['previous'],0)
    def test_projection_against_direct_svd(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(12,9));g=x@x.T;p=project_from_gram(g,np.arange(7),[8])[0]
        _,_,v=np.linalg.svd(x[:7],full_matrices=False)
        for r in RANKS:self.assertAlmostEqual(p['energy'][str(r)],np.linalg.norm(v[:r]@x[8])**2/np.linalg.norm(x[8])**2,places=10)
    def test_empty_and_zero_cosine(self):
        self.assertIsNone(cosine(np.zeros((2,2)),0,1));self.assertEqual(spectrum(np.zeros((2,2)),[dict(source_id=i) for i in range(2)],'raw')['total_energy'],0)
    def test_source_not_rows(self):
        q=[dict(source_id=1,v=1) for _ in range(100)]+[dict(source_id=2,v=0)]
        z=source_stats(q,['v']);self.assertEqual(z['metrics']['v']['mean'],.5);self.assertEqual(z['sources'],2)
    def test_undefined_cosine_retained(self):
        z=pair_stats([dict(donor=1,recipient=2,cosine=None,utility_v=.1)],draws=2)
        self.assertEqual(z['pairs'],1);self.assertEqual(z['defined_cosine_pairs'],0)
    def test_node_roles_and_determinism(self):
        q=[dict(donor=1,recipient=2,cosine=.5,utility_v=.1),dict(donor=2,recipient=3,cosine=-.5,utility_v=-.1),dict(donor=1,recipient=3,cosine=.1,utility_v=.01)]
        a=pair_stats(q,draws=100);b=pair_stats(q,draws=100);self.assertEqual(a,b);self.assertAlmostEqual(a['metrics']['positive_minus_negative']['mean'],.17)

if __name__=='__main__':unittest.main()
