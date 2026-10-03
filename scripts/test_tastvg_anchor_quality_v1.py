import unittest
import numpy as np
from scripts.tastvg_anchor_quality_math_v1 import feature,pseudo_pairs,fit,predict,choose,quartiles

class AnchorQualityTests(unittest.TestCase):
    def row(self,s,scores):return dict(source_id=s,scores=list(scores),candidate_t=list(np.linspace(1,0,32)))
    def test_exact_feature_coupling(self):
        z=np.linspace(-3,4,32)
        for j in range(32):
            f=feature(z,j);self.assertAlmostEqual(f['margin']+f['anchor_score'],f['top_context'])
    def test_source_weights_self_and_negative_labels(self):
        p,m=pseudo_pairs([self.row(0,np.arange(32)),self.row(1,np.arange(32)**2)])
        for s in [0,1]:self.assertAlmostEqual(sum(r['weight'] for r in p if r['source_id']==s),1)
        self.assertEqual(len(p),64);self.assertEqual(sum(not r['eligible'] for r in p),2)
        self.assertTrue(all(r['y']<=0 for r in p))
    def test_ambiguous_top_excludes_whole_source(self):
        p,m=pseudo_pairs([self.row(0,[1]*32)])
        self.assertEqual(p,[]);self.assertEqual(m[0]['total_weight'],0)
    def test_minimum_norm_rank_deficiency(self):
        r=[dict(source_id=0,margin=2.,anchor_score=-1.,weight=.25,y=.3) for _ in range(4)]
        m=fit(r,'M1');self.assertEqual(m['rank'],1);self.assertAlmostEqual(predict(m,r[0]),.3)
    def test_source_total_weight_replication(self):
        a=[dict(source_id=s,margin=float(s),anchor_score=float(s*s),weight=1.,y=float(s)/10) for s in range(4)]
        b=[dict(r,weight=r['weight']/5) for r in a for _ in range(5)]
        np.testing.assert_allclose(fit(a,'M1')['coefficients'],fit(b,'M1')['coefficients'],atol=1e-14)
    def test_zero_delta_rule_no_GT_and_tie_fallback(self):
        models={a:dict(arm=a,coefficients=[.1,0]+([0] if a=='M1' else []),feature_ranges=dict(margin=[0,1],anchor_score=[-1,1])) for a in ['M0','M1']}
        s=np.arange(32);self.assertEqual(choose(s,0,models)['choices']['M1'],31)
        s[-2]=31;self.assertEqual(choose(s,0,models)['choices']['M1'],0)
        for a in models:models[a]['coefficients'][0]=0
        self.assertEqual(choose(np.arange(32),0,models)['choices']['M0'],0)
    def test_stable_quartile_ties(self):
        r=[dict(cell_key=str(j),A8_t=0.) for j in [7,2,1,6,4,3,0,5]]
        q=quartiles(r);self.assertEqual([x['cell_key'] for x in q[0]],['0','1'])

if __name__=='__main__':unittest.main()
