"""Numerical configuration checks independent of model/GT/development score."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from vg_tta.decota_spatial_opd_tunable_v1 import rollout,likelihood_loss,optimizer_step

class ConfigurationContracts(unittest.TestCase):
    def test_sigma_and_tau_apply_to_true_distribution(self):
        mu=torch.tensor([[.2,-.3,.4,-.5]])
        e=torch.tensor([[.5,.5,.3,.3]])
        cfg=dict(sigma=.1,tau=.5,lr=.01,steps=3,writeback=0.,samples=32)
        small=rollout(mu,e,'contract',0,config=cfg)
        large=rollout(mu,e,'contract',0,config={**cfg,'sigma':.5})
        torch.testing.assert_close(large['samples']-mu[:,None],5*(small['samples']-mu[:,None]),atol=2e-7,rtol=1e-6)
        narrow=rollout(mu,e,'contract',0,config={**cfg,'tau':.1})
        np.testing.assert_array_equal(small['samples'].numpy(),narrow['samples'].numpy())
        assert not torch.equal(small['weights'],narrow['weights'])
        expected=np.exp(small['rewards'].numpy()/.5);expected/=expected.sum(-1,keepdims=True)
        np.testing.assert_allclose(small['weights'].numpy(),expected,atol=1e-7)

    def test_likelihood_gradient_independent_across_scales(self):
        for sigma in [.1,.25,.5]:
            mu=torch.tensor([[.3,-.1,.4,.2]],requires_grad=True);roll=mu.detach()+.1
            rr=rollout(roll,torch.tensor([[.5,.5,.2,.3]]),'g',0,config=dict(sigma=sigma,tau=.25))
            g=torch.autograd.grad(likelihood_loss(mu,roll,rr['samples'],rr['weights'],sigma=sigma),mu)[0].numpy()
            a=rr['samples'].numpy().astype(float);w=rr['weights'].numpy().astype(float);m=mu.detach().numpy().astype(float)
            expected=(m-roll.numpy())/sigma**2-((w-1/32)[...,None]*(a-m[:,None])).sum(1)/sigma**2
            np.testing.assert_allclose(g,expected,rtol=2e-5,atol=2e-5)

    def test_uninformative_step_does_not_move_nonzero_adam_moments(self):
        p=torch.tensor([.2],requires_grad=True);opt=torch.optim.Adam([p],lr=.03)
        optimizer_step(opt,(p-1).square().sum(),[p],True)
        before=p.detach().clone();step=opt.state[p]['step'].clone()
        self.assertIsNone(optimizer_step(opt,p.square().sum(),[p],False))
        assert torch.equal(before,p) and torch.equal(step,opt.state[p]['step'])

if __name__=='__main__':unittest.main()
