import numpy as np
import torch
from vg_tta.tastvg_spatial_rank_s11_v1 import average_ranks,reverse_kl,update_scale

def test_rank_ties_and_monotone_invariance():
    x=np.array([.2,.8,.8,.1,.4]);want=[3, .5, .5,4,2]
    np.testing.assert_array_equal(average_ranks(x),want)
    np.testing.assert_array_equal(average_ranks(x*100+4),want)
    perm=[4,1,3,0,2];np.testing.assert_array_equal(average_ranks(x[perm]),np.array(want)[perm])
    np.testing.assert_array_equal(average_ranks(np.ones(9)),np.full(9,4.))

def test_rank_rkl_gradient_and_detached_support():
    b=torch.tensor([[.45,.5,.25,.3]],dtype=torch.double,requires_grad=True)
    c=torch.tensor([[[.5,.52,.2,.35]],[[.3,.42,.26,.32]],[[.6,.56,.21,.34]]],dtype=torch.double,requires_grad=True)
    r=torch.tensor([.1,.4,.3],dtype=torch.double,requires_grad=True)
    loss,p,q,d=reverse_kl(b,c,r,[5,2]);assert torch.autograd.gradcheck(lambda x:reverse_kl(x,c,r,[5,2])[0],(b,))
    loss.backward();assert b.grad is not None and c.grad is None and r.grad is None
    np.testing.assert_allclose(q.detach().numpy(),np.exp([-2,0,-1])/np.exp([-2,0,-1]).sum())
    np.testing.assert_allclose(loss.detach(),np.sum(p.detach().numpy()*np.log(p.detach().numpy()/q.detach().numpy())))

def test_global_normalization_and_zero():
    g=[torch.tensor([3.,0.]),torch.tensor([4.])];scale,n=update_scale(g,'norm',.005,.015)
    assert n==5.;assert abs(np.sqrt(sum(float((x*scale).square().sum()) for x in g))-.015)<1e-9
    assert update_scale([torch.zeros(9)],'norm',.005,.015)==(0.,0.)
    assert update_scale(g,'rank',.005,.015)==(.005,5.)
