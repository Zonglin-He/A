import math,unittest
import torch
from vg_tta.desta3d_v3_oracle_mixer_gap import analytic_joint,direction_readout,mixer_readout
from vg_tta.desta3d_v3_decomposition import corrections,norm
from vg_tta.desta3d_v3_joint_mixer import JointCorrectionMixer,native_supervision

class Contracts(unittest.TestCase):
    def test_equal_old_budget_and_full_gradient_balancing(self):
        torch.manual_seed(3);q,_=torch.linalg.qr(torch.randn(8,4).double());q=q.float()
        gt=torch.randn(1,2,2,2,8);gs=torch.randn_like(gt);f=torch.ones_like(gt)
        r=math.sqrt((.087687**2+.170316**2)/2)
        actual=analytic_joint(gt,gs,q,f,r)
        old=corrections(gt,gs,q[:,:2],q[:,2:],q,f,{'event':.087687,'spatial':.170316})['J_pass']
        self.assertTrue(torch.allclose(actual,old,atol=1e-7,rtol=1e-6))
        self.assertAlmostEqual(norm(actual)/norm(f),r,places=7)
        self.assertGreater(direction_readout({'event':gt,'spatial':gs},{'oracle':actual},f,r)['cosine_to_oracle']['oracle'],.999999)
    def test_missing_zero_and_cancellation(self):
        f=torch.ones(1,1,1,1,4);q=torch.eye(4);g=torch.tensor([[[[[1.,0.,0.,0.]]]]]);z=torch.zeros_like(g)
        self.assertEqual(norm(analytic_joint(z,z,q,f,.1)),0)
        self.assertEqual(norm(analytic_joint(g,-g,q,f,.1)),0)
        d=analytic_joint(g,z,q,f,.1);self.assertAlmostEqual(norm(d),.2,places=7)
        self.assertLess(float((d*g).sum()),0)
        missing=native_supervision({'event_active':[False]}, {'branches':[]}, 'spatial')[1]
        self.assertEqual(missing,'missing_native_spatial_support')
    def test_mixer_identity_and_saturation_contract(self):
        torch.manual_seed(5);q=torch.eye(8)[:,:4];m=JointCorrectionMixer(q,hidden=2).requires_grad_(False)
        args=(torch.randn(1,2,2,2,2),torch.randn(1,2),torch.randn(1,2),torch.randn(1,2,2,2,8),torch.randn(1,2,2,2,8))
        baseline=m(*args);d,a,meta=mixer_readout(m,args)
        self.assertTrue(torch.equal(d,baseline));self.assertEqual(meta['coefficient_rms'],0)
        m.output.bias.fill_(10);baseline=m(*args);d,a,meta=mixer_readout(m,args)
        self.assertTrue(torch.equal(d,baseline));self.assertGreater(meta['norm_over_cap'],.999)
        self.assertEqual(len(m.output._forward_hooks),0)
    def test_full_support_and_cross_direction(self):
        f=torch.ones(1,1,1,1,4);g=f.clone();q=torch.eye(4)
        with self.assertRaises(ValueError):analytic_joint(g,g[..., :2],q,f,.1)
        r=direction_readout({'event':g,'spatial':-g},{'oracle':g,'learned':-g},f,.1)
        self.assertEqual(r['cosine_to_oracle']['learned'],-1.)
        self.assertEqual(r['descent_dot']['event']['learned'],4.)
        self.assertEqual(r['descent_dot']['spatial']['learned'],-4.)

if __name__=='__main__':unittest.main()
