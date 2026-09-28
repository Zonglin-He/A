import unittest
import torch
from vg_tta.desta3d_v3_decomposition import shared_fields, corrections, geometry, union_basis, project, norm
from vg_tta.desta3d_v3_free_actuation import native_ce

class Contracts(unittest.TestCase):
    def test_shared_F_chain_and_restore(self):
        class A(torch.nn.Module):
            def forward(self,x,q): return x + 2*x*q
        a=A();x=torch.tensor([2.,3.]);q=torch.tensor([1.,2.]);leaf=x.clone().requires_grad_()
        with shared_fields(a,{'event':leaf},['event']):
            y=a(x,q);self.assertTrue(torch.equal(y,a.forward(x,q)));y.sum().backward()
        self.assertTrue(torch.equal(leaf.grad,1+2*q));self.assertEqual(len(a._forward_pre_hooks),0)
    def test_pass_dispatch(self):
        a=torch.nn.Identity();x=torch.zeros(4);t=torch.ones(4);s=t*2
        with shared_fields(a,{'event':t,'spatial':s},['event','spatial']):
            self.assertTrue(torch.equal(a(x),t));self.assertTrue(torch.equal(a(x),s))
    def test_exception_restore(self):
        a=torch.nn.Identity()
        with self.assertRaises(ValueError):
            with shared_fields(a,{'event':torch.ones(3)},['event']):a(torch.zeros(4))
        self.assertEqual(len(a._forward_pre_hooks),0)
    def test_budget_and_cross_harm(self):
        q=torch.eye(4);gt=torch.tensor([[1.,0.,0.,0.]]);gs=torch.tensor([[-.5,1.,0.,0.]])
        qu,_=union_basis(q[:,:2],q[:,1:3]);d=corrections(gt,gs,q[:,:2],q[:,1:3],qu,torch.ones_like(gt),{'event':.1,'spatial':.2})
        self.assertAlmostEqual(norm(d['J'])**2,norm(d['T'])**2+norm(d['S'])**2,places=7)
        self.assertAlmostEqual(2*norm(d['J_pass'])**2,norm(d['T'])**2+norm(d['S'])**2,places=7)
        g=geometry(gt,gs,d);self.assertLess(g['descent_dot']['S']['T'],0);self.assertGreater(g['descent_dot']['T']['T'],0)
        self.assertLess(norm(d['S']-project(d['S'],q[:,1:3])),1e-7)
    def test_zero_and_support(self):
        z=torch.zeros(1,4);q=torch.eye(4);d=corrections(z,z,q,q,q,torch.ones_like(z),{'event':.1,'spatial':.2})
        self.assertTrue(all(norm(x)==0 for x in d.values()));self.assertIsNone(geometry(z,z,d)['gradient_cosine'])
        with self.assertRaises(ValueError):corrections(z,z[:,:2],q,q,q,z,{'event':.1,'spatial':.2})
    def test_objective_denominator(self):
        x=torch.randn(3,4,17,requires_grad=True);t=torch.arange(12).reshape(3,4);v=torch.ones_like(t,dtype=torch.bool);v[1]=False
        y=native_ce(x,t,v);ref=sum(-x[i,j].log_softmax(0)[t[i,j]] for i in [0,2] for j in range(4))/8
        self.assertTrue(torch.allclose(y,ref));y.backward();self.assertEqual(float(x.grad[1].abs().sum()),0.)

if __name__=='__main__':unittest.main()
