import numpy as np
import torch
from vg_tta.tastvg_matched_ablation_a2_v1 import rank_loss
from vg_tta.tastvg_spatial_online_opd_s1_v1 import reverse_kl


def fixture():
    b=torch.tensor([[.4,.5,.2,.3],[.6,.4,.3,.2]],dtype=torch.float64,requires_grad=True)
    c=torch.stack([b.detach(),b.detach()+.01,b.detach()-.02])
    return b,c


def test_raw_matches_original_s1_value_and_gradient():
    b,c=fixture();r=torch.tensor([.43,.41,.4],dtype=b.dtype)
    a=rank_loss(b,c,-r.numpy(),[1.,1.],'raw_rkl')[0]
    z=reverse_kl(b,c,r,[1.,1.])[0]
    torch.testing.assert_close(a,z,rtol=0,atol=0)
    torch.testing.assert_close(torch.autograd.grad(a,b,retain_graph=True)[0],torch.autograd.grad(z,b)[0],rtol=0,atol=0)


def test_pairwise_independent_formula_and_gradient():
    b,c=fixture();rank=np.array([0.,1.,2.])
    loss,_,_,d=rank_loss(b,c,rank,[1.,1.],'pairwise_rank')
    expected=torch.stack([torch.logaddexp(d[i]-d[j],torch.zeros_like(d[i])) for i,j in [(0,1),(0,2),(1,2)]]).mean()
    torch.testing.assert_close(loss,expected)
    assert torch.isfinite(torch.autograd.grad(loss,b)[0]).all()


def test_tied_pairs_have_no_update():
    b,c=fixture();loss,*_=rank_loss(b,c,[1.,1.,1.],[1.,1.],'pairwise_rank')
    assert loss.item()==0
    assert torch.count_nonzero(torch.autograd.grad(loss,b)[0])==0
