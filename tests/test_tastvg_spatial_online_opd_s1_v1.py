import unittest
import numpy as np
import torch
from vg_tta.tastvg_spatial_online_opd_s1_v1 import geometry,reverse_kl
from vg_tta.tastvg_spatial_expansion_s0_v1 import alignment

class OnlineSpatial(unittest.TestCase):
    def test_student_geometry_matches_existing_native_terms(self):
        p=torch.tensor([[.4,.5,.2,.3],[.6,.4,.3,.2]],dtype=torch.double,requires_grad=True);q=torch.stack([p.detach()+.01,p.detach()-.01]).requires_grad_(True)
        got=geometry(p,q,5,3)
        for k in range(2):self.assertAlmostEqual(float(got[k]),float(alignment(p,q[k],torch.ones(2,dtype=torch.bool),5,3)),places=12)
        g=torch.autograd.grad(got.sum(),[p,q],allow_unused=True);self.assertIsNotNone(g[0]);self.assertIsNone(g[1])
    def test_rkl_exact_and_teacher_stop_gradient(self):
        p=torch.tensor([[.4,.5,.2,.3],[.6,.4,.3,.2]],dtype=torch.double,requires_grad=True);q=torch.stack([p.detach()+.02,p.detach()-.01]);r=torch.tensor([.3,.8],dtype=torch.double,requires_grad=True)
        loss,pi,qi,d=reverse_kl(p,q,r,(5,3));expected=np.sum(pi.detach().numpy()*np.log(pi.detach().numpy()/qi.detach().numpy()));self.assertAlmostEqual(float(loss),expected,places=12)
        grad=torch.autograd.grad(loss,[p,r],allow_unused=True);self.assertIsNone(grad[1]);self.assertTrue(torch.isfinite(grad[0]).all());self.assertTrue(torch.autograd.gradcheck(lambda z:reverse_kl(z,q,r,(5,3))[0],(p,)))
    def test_identical_support_uniform_reward_zero(self):
        p=torch.tensor([[.5,.5,.2,.2]],requires_grad=True);q=p.detach()[None].repeat(9,1,1)
        loss,_,_,_=reverse_kl(p,q,torch.zeros(9),(5,3));self.assertEqual(float(loss),0);self.assertLess(float(torch.autograd.grad(loss,p)[0].abs().sum()),1e-10)

if __name__=='__main__':unittest.main()
