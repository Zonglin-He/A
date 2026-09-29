"""Mathematical checks for the actual endpoint/ranking loss consumed by C3."""
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from vg_tta.tastvg_temporal_fourarm_v1 import endpoint_map,loss,candidate_logits

class FourarmTests(unittest.TestCase):
    def setUp(self):
        self.model=SimpleNamespace(cfg=SimpleNamespace(SOLVER=SimpleNamespace(SIGMA=2.,TEMP_COEF=5.)))
    def test_physical_halfopen_and_empty_mapping(self):
        records=[{'frame_ids':[0,4,8,12]},{'frame_ids':[2,6,10,14]}]
        self.assertEqual(endpoint_map(records,[[4,11],[15,16]]),[[[1,2],[3,3]],[[1,2],[3,3]]])
    def test_pairwise_gradient_finite_difference_and_direction(self):
        torch.manual_seed(3);z=torch.randn(1,4,2,dtype=torch.float64,requires_grad=True)
        bb=[[[0,1],[1,3],[0,3]]];scores=[.9,.1,.1]
        fn=lambda q:loss(self.model,[{'pred_sted':q}],bb,scores,0,'OPD')
        grad=torch.autograd.grad(fn(z),z)[0];direction=torch.randn_like(z);eps=1e-6
        fd=(fn(z+eps*direction)-fn(z-eps*direction))/(2*eps)
        self.assertAlmostEqual(float(fd),float((grad*direction).sum()),places=8)
        self.assertLess(float(fn(z-.01*grad)),float(fn(z)))
        self.assertEqual(float(loss(self.model,[{'pred_sted':z}],bb,[1,1,1],0,'OPD')),0.)
    def test_hard_matches_numpy_native_formula(self):
        z=torch.tensor([[[.2,.8],[1.2,-.5],[-.3,.6],[.7,.1]]],dtype=torch.float64)
        zz=z.numpy()[0];p=np.exp(zz-zz.max(0));p/=p.sum(0)
        target=np.exp(-(np.arange(4)[:,None]-np.array([1,3]))**2/8)+1e-6;target/=target.sum(0)
        expected=(p*np.log((p+1e-6)/target)).sum(1).mean()*5
        got=loss(self.model,[{'pred_sted':z}],[[[1,3]]],[1],0,'Hard')
        # Native target construction uses float32 arange/exponential.
        self.assertAlmostEqual(float(got),float(expected),places=6)

if __name__=='__main__':unittest.main()
