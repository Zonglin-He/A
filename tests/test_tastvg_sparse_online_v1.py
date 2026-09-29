import unittest,itertools
import numpy as np
import torch
from vg_tta.tastvg_sparse_online_v1 import native_scores,source_state,arrive,score,update
from vg_tta.tastvg_corruption_c0c1_v1 import ranked_spans

class O1Tests(unittest.TestCase):
    def test_native_envelope_score_exhaustive(self):
        torch.manual_seed(17);zz=[torch.randn(1,4,2),torch.randn(1,4,2)];records=[{'frame_ids':[0,2,4,6]},{'frame_ids':[1,3,5,7]}];expected={}
        for a,b in itertools.product(ranked_spans(zz[0]),ranked_spans(zz[1])):
            interval=(min(records[0]['frame_ids'][a[1]],records[1]['frame_ids'][b[1]]),max(records[0]['frame_ids'][a[2]],records[1]['frame_ids'][b[2]])+1)
            expected[interval]=max(expected.get(interval,float('-inf')),(a[0]+b[0])/2)
        got=native_scores(zz,records,list(expected));np.testing.assert_allclose(got.numpy(),list(expected.values()),atol=0,rtol=0)
    def test_sgd_matches_independent_pair_gradient(self):
        torch.manual_seed(4);phi=torch.randn(4,768,dtype=torch.float64);base=torch.randn(4,dtype=torch.float64);state=source_state();teacher=[3.,1.,2.,1.]
        after,diag=update(base,phi,state,teacher,.001)
        grad=np.zeros(768);n=0
        for i,j in itertools.product(range(4),repeat=2):
            if teacher[i]>teacher[j]+1e-12:grad-=(phi[i]-phi[j]).numpy()/(1+np.exp(float(base[i]-base[j])));n+=1
        np.testing.assert_allclose(after['w'].numpy(),-.001*grad/n,atol=1e-17);self.assertEqual(float(after['b']),0.)
        self.assertTrue(torch.equal(score(base,phi,state),base));self.assertEqual(diag['steps'],1)
    def test_chain_clone_and_no_information_update(self):
        source=source_state();prev=arrive(source);prev['w'][0]=1
        self.assertEqual(float(source['w'][0]),0);next_state=arrive(prev);self.assertTrue(torch.equal(next_state['w'],prev['w']))
        new,d=update(torch.zeros(2,dtype=torch.float64),torch.ones(2,768,dtype=torch.float64),next_state,[1.,1.]);self.assertTrue(torch.equal(new['w'],prev['w']));self.assertEqual(d['pairs'],0)

if __name__=='__main__':unittest.main()
