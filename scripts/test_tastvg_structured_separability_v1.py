import unittest
import numpy as np
from scripts.tastvg_structured_separability_math_v1 import binary,extract,quadrant,label,source_mean

def row(s,x,y,eligible=True):
    return dict(source_id=s,order='o',condition='c',eligible=eligible,label_t=y,
                readouts={'r':dict(P_A=-x,R_A=-x,delta_P=x,delta_R=x)})

class Checks(unittest.TestCase):
    def test_perfect_and_reversed(self):
        rows=[row(0,3,'helpful'),row(0,1,'harmful'),row(1,4,'helpful'),row(1,2,'harmful')]
        self.assertEqual(binary(rows,'r','delta_P',draws=100)['auc']['mean'],1.)
        for r in rows:r['readouts']['r']['delta_P']*=-1
        self.assertEqual(binary(rows,'r','delta_P',draws=100)['auc']['mean'],0.)
    def test_ties_and_orientation(self):
        rows=[row(0,2,'helpful'),row(0,2,'harmful')]
        self.assertEqual(binary(rows,'r','P_A',draws=100)['auc']['mean'],.5)
    def test_replication_does_not_increase_source_weight(self):
        rows=[row(0,3,'helpful'),row(0,0,'harmful'),row(1,0,'helpful'),row(1,2,'harmful')]
        a=binary(rows,'r','delta_P',draws=100)
        b=binary(rows+[r.copy() for r in rows if r['source_id']==0]*9,'r','delta_P',draws=100)
        self.assertEqual(a['auc'],b['auc'])
    def test_undefined_and_nonmixed(self):
        a=binary([row(0,3,'helpful'),row(1,2,'harmful')],'r','delta_P',draws=100)
        self.assertIsNone(a['within_source_auc']['mean'])
        self.assertGreater(a['auc']['bootstrap_undefined'],0)
        self.assertIsNone(binary([row(0,1,'helpful')],'r','delta_P',draws=100)['auc']['mean'])
    def test_noops_neutrals(self):
        a=binary([row(0,100,'helpful',False),row(1,1,'neutral'),row(2,1,'helpful'),row(2,0,'harmful')],'r','delta_P',draws=100)
        self.assertEqual(a['counts']['noop'],1);self.assertEqual(a['counts']['neutral_replacements'],1)
        self.assertEqual(a['counts']['helpful'],1)
    def test_signs(self):
        self.assertEqual(quadrant(1.,-1.),'up/down');self.assertEqual(quadrant(0,0),'tie/tie')
        self.assertEqual(label(1e-13),'neutral')
    def test_source_order_condition_balance(self):
        rows=[dict(source_id=0,order='a',condition='x',v=0.),dict(source_id=0,order='a',condition='y',v=2.),
              dict(source_id=0,order='b',condition='x',v=5.),dict(source_id=1,order='a',condition='x',v=7.)]
        self.assertEqual(source_mean(rows,['v'],100)['metrics']['v']['mean'],5.)
    def test_noop_extraction(self):
        from scripts.tastvg_structured_separability_math_v1 import RECIPES
        z={}
        for pv,rv,c in RECIPES.values():
            for view,task in [(pv,'precision'),(rv,'recall')]:z[f'candidate/{view}/{task}/{c}']=np.linspace(-.5,1.5,32)
        for v in extract(z,5,5).values():
            self.assertEqual(v['delta_P'],0.);self.assertEqual(v['delta_R'],0.)
        self.assertLess(extract(z,0,31)['Full']['P_A'],0.)

if __name__=='__main__':unittest.main()
