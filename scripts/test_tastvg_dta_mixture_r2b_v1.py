"""Distribution semantics, matched optimizer/reset and label embargo tests."""
import sys,unittest,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from vg_tta.tastvg_dta_oracle_v1 import gaussian,fit_query
from vg_tta.tastvg_dta_mixture_r2b_v1 import mixture,fit_mixture,validate_proposals,read_guard,teacher_decode

class Tests(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(7);h=torch.randn(8,256)
        w={'0.weight':torch.randn(256,256)*.03,'0.bias':torch.zeros(256),'1.weight':torch.randn(2,256)*.03,'1.bias':torch.zeros(2)}
        return h,list(range(0,16,2)),w
    def test_equal_component_arithmetic(self):
        ids=[0,2,5,9];props=[[0,4],[4,10],[.25,6.5]];q,s=mixture(ids,props)
        expected=torch.stack([gaussian(ids,p)[0].exp() for p in props]).mean(0)
        torch.testing.assert_close(q.exp(),expected,rtol=1e-13,atol=1e-15);self.assertAlmostEqual(float(q.exp().sum()),1,13)
    def test_single_component_full_fit_bitwise(self):
        h,ids,head=self.fixture();a=fit_query(h,ids,head,[2,12],.01);b=fit_mixture(h,ids,head,[[2,12]],.01)
        self.assertEqual(a['before'],b['before']);self.assertEqual(a['after'],b['after'])
        for x,y in zip(a['states'],b['states']):
            for k in x:self.assertTrue(torch.equal(x[k],y[k]))
        for x,y in zip(a['trace'],b['trace']):
            for k,v in x.items():self.assertEqual(v,y[{'GT_KL':'teacher_KL','after_GT_KL':'after_teacher_KL'}.get(k,k)])
    def test_duplicates_keep_empirical_weight(self):
        ids=[0,2,4,6,8];a,_=gaussian(ids,[0,3]);b,_=gaussian(ids,[5,9]);q,_=mixture(ids,[[0,3],[0,3],[5,9]])
        torch.testing.assert_close(q.exp(),(2*a.exp()+b.exp())/3,rtol=1e-13,atol=1e-15)
        equal,_=mixture(ids,[[0,3],[5,9]]);self.assertGreater(float((q.exp()-equal.exp()).abs().max()),.01)
    def test_proposal_order_invariant(self):
        ids=[0,2,4,6,8];x=[[0,3],[1.5,7],[5,9]];q,_=mixture(ids,x);r,_=mixture(ids,x[::-1]);torch.testing.assert_close(q,r,rtol=1e-13,atol=1e-14)
    def test_joint_not_product_of_mixed_marginals(self):
        ids=list(range(10));q,_=mixture(ids,[[0,2],[7,10]]);pairs=torch.triu_indices(10,10,1);full=torch.zeros(10,10,dtype=torch.float64);full[pairs[0],pairs[1]]=q.exp()
        product=full.sum(1)[:,None]*full.sum(0)[None,:];product=product.triu(1);product/=product.sum()
        self.assertGreater(float((full-product).abs().max()),.02)
    def test_invalid_support(self):
        for p in [[],[[1,1]],[[4,2]],[[float('nan'),2]]]:
            with self.assertRaises(ValueError):validate_proposals(p)
    def test_reset_and_gradient_channel(self):
        h,ids,head=self.fixture();x=fit_mixture(h,ids,head,[[2,12],[4,10]],.01);y=fit_mixture(h,ids,head,[[2,12],[4,10]],.01)
        self.assertEqual(x['head_final_sha256'],y['head_final_sha256']);self.assertTrue(x['episodic_reset']);self.assertFalse(x['GT_supervised'])
        self.assertEqual(len(x['trace']),3);self.assertGreater(x['trace'][0]['gradient_norm'],0);self.assertLess(x['trace'][-1]['after_loss'],x['trace'][0]['loss'])
    def test_label_and_result_guard(self):
        for f in ['a/GT_LABELS_search.json','a/oracle_runs/q.pt','a/ROWS.json','a/EOracle_TRACES_SCORED.json']:
            with self.assertRaises(PermissionError):read_guard('open',(f,'rb',0))
        read_guard('open',('a/EXPERT_SUPPORT.json','rb',0));read_guard('open',('a/HEAD.pt','rb',0))

if __name__=='__main__':torch.set_num_threads(2);unittest.main()
