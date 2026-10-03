import unittest
import numpy as np
from scripts.tastvg_pairwise_certification_math_v1 import pairs, spread, sorted_fit, decide, IQR_EPS
from scripts.tastvg_anchor_certification_math_v1 import isotonic

class PairwiseTests(unittest.TestCase):
    def row(self, source, scores, labels):
        return dict(source_id=source, scores=list(scores), candidate_t=list(labels))
    def test_source_balance_with_unequal_pair_counts(self):
        rr=[self.row(0,np.arange(32),np.arange(32)/31),self.row(1,[0]*16+[1]*16,[1]*16+[0]*16)]
        p,m=pairs(rr)
        self.assertEqual([x['strict_pairs'] for x in m],[496,256])
        for sid in [0,1]: self.assertAlmostEqual(sum(x['weight'] for x in p if x['source_id']==sid),1)
        self.assertTrue(all(x['delta_t']<0 for x in p if x['source_id']==1))
    def test_GT_ties_are_retained(self):
        p,m=pairs([self.row(0,np.arange(32),np.zeros(32))])
        self.assertEqual(len(p),496); self.assertTrue(all(x['delta_t']==0 for x in p))
    def test_empty_source_is_explicit(self):
        p,m=pairs([self.row(0,np.ones(32),np.zeros(32))])
        self.assertEqual(p,[]); self.assertEqual(m[0]['total_fit_weight'],0);self.assertTrue(m[0]['zero_IQR'])
    def test_within_cell_additive_invariance(self):
        s=np.linspace(-1,1,32); self.assertAlmostEqual(spread(s),spread(s+15))
        p,_=pairs([self.row(0,s,s)]);q,_=pairs([self.row(0,s+15,s)])
        np.testing.assert_allclose([r['norm_margin'] for r in p],[r['norm_margin'] for r in q],atol=1e-13)
    def test_zero_IQR_with_outlier(self):
        p,m=pairs([self.row(0,[0]*31+[1],np.zeros(32))])
        self.assertEqual(len(p),31);self.assertEqual(p[0]['norm_margin'],1/IQR_EPS)
    def test_solver_ties_weights_and_zero_bootstrap_weight(self):
        x=np.array([.1,.1,.2,.3,.4]);y=np.array([.8,.6,-.1,.4,.9]);w=np.array([.1,.4,0,.2,.3])
        a,b=sorted_fit(x,y,w);c,d=isotonic(x,y,w)
        np.testing.assert_array_equal(a,c);np.testing.assert_allclose(b,d,atol=1e-14)
    def test_frozen_ties_domain_and_no_positive_bound(self):
        model=dict(available=True,knots=[.001,2],mean=[.1,.1],lower=[.01,.01])
        models={k:model for k in ['Pair-Raw','Pair-Norm']}
        s=np.zeros(32);s[1]=1
        self.assertEqual(decide(s,0,models)['choices']['Pair-Raw'],1)
        self.assertEqual(decide(s,0,models)['evidence']['Pair-Norm']['reason'],'outside_source_pair_margin_support')
        s[2]=1;self.assertEqual(decide(s,0,models)['choices']['Pair-Raw'],0)
        models={k:dict(model,lower=[0,0]) for k in models};s[2]=0
        self.assertEqual(decide(s,0,models)['choices']['Pair-Raw'],0)

if __name__=='__main__': unittest.main()
