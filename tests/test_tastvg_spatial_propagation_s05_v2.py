import unittest
import numpy as np
import torch
from vg_tta.tastvg_spatial_propagation_s05_v2 import foreground_probability,moment_boxes,alignment,propagate
import test_tastvg_spatial_propagation_s05_v1 as fixtures

class SoftMoments(unittest.TestCase):
    def test_probability_independent_numpy_and_background_bank(self):
        rng=np.random.default_rng(19);h=rng.normal(size=(3,4,5));r=rng.normal(size=(8,5));fg=np.array([1,0,1,0,0,1,0,0],bool)
        def unit(x):return x/np.linalg.norm(x,axis=-1,keepdims=True)
        scores=[.5*(unit(h)@unit(bank.mean(0)))+.5*np.max(unit(h)@unit(bank).T,axis=-1) for bank in [r[fg],r[~fg]]]
        expected=1/(1+np.exp(scores[1]-scores[0]));got,logits=foreground_probability(torch.tensor(h),torch.tensor(r),torch.tensor(fg))
        np.testing.assert_allclose(got.numpy(),expected,atol=1e-12)
        swapped,_=foreground_probability(torch.tensor(h),torch.tensor(r),torch.tensor(~fg));np.testing.assert_allclose(swapped.numpy(),1-expected,atol=1e-12)
    def test_uniform_moments_exact_discrete_variance(self):
        b=moment_boxes(torch.full((2,70),.5,dtype=torch.double),7,10).numpy()
        expected=[.5,.5,np.sqrt(1-1/100),np.sqrt(1-1/49)]
        np.testing.assert_allclose(b,np.tile(expected,(2,1)),atol=1e-12)
    def test_corner_clipping(self):
        p=torch.tensor([[1.,0.,0.,1.,0.,0.,0.,0.]],dtype=torch.double);b=moment_boxes(p,2,4)
        lo=b[:,:2]-b[:,2:]/2;hi=b[:,:2]+b[:,2:]/2
        self.assertTrue(bool((lo>=0).all() and (hi<=1).all()));self.assertEqual(float(lo[0,0]),0.)
    def test_sparse_empty_and_reference_preservation(self):
        d,e=fixtures.Propagation().fixture();out=propagate(d,e)
        self.assertEqual(out['diagnostics']['propagated_frames'],2);self.assertTrue(out['valid'].all())
        np.testing.assert_allclose(out['boxes'][0],[.25,.25,.5,.5]);self.assertIn('probability',out);self.assertNotIn('propagated_masks',out)
        d,e=fixtures.Propagation().fixture(True);self.assertFalse(propagate(d,e)['valid'].any())
    def test_total_frame_denominator_and_gradient(self):
        p=torch.tensor([[.4,.4,.3,.2],[.6,.5,.2,.3]],dtype=torch.double,requires_grad=True);q=torch.tensor([[.5,.5,.2,.3],[.4,.4,.2,.3]],dtype=torch.double);v=torch.tensor([True,False])
        from vg_tta.tastvg_spatial_expansion_s0_v1 import alignment as old
        self.assertAlmostEqual(float(alignment(p,q,v,5,3)),float(old(p,q,v,5,3))/2)
        self.assertTrue(torch.autograd.gradcheck(lambda z:alignment(z,q,v,5,3),(p,)))

if __name__=='__main__':unittest.main()
