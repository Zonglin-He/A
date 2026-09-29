import numpy as np
import torch
from vg_tta.tastvg_matched_ablation_a1_v1 import permutation,rank_loss
from vg_tta.tastvg_spatial_rank_s11_v1 import average_ranks,reverse_kl
from vg_tta.tastvg_spatial_expansion_s0_v1 import alignment

def test_rank_permutation_preserves_spectrum():
    rank=average_ranks([1,1,.5,.5,0,0,0,-1,-1]);p=permutation('fixture','order1');q=np.exp(-rank);q/=q.sum()
    assert sorted(p)==list(range(9)) and p==permutation('fixture','order1')
    np.testing.assert_array_equal(np.sort(q),np.sort(q[p]))

def test_identity_assignment_matches_frozen_loss_and_grad():
    g=torch.Generator().manual_seed(2);b=(torch.rand(4,4,generator=g)*.2+.3).requires_grad_();c=torch.rand(9,4,4,generator=g)*.2+.3;r=torch.tensor([1,.8,.7,.7,.4,.3,.2,.1,0.])
    a=reverse_kl(b,c,r,[5,2])[0];z=rank_loss(b,c,average_ranks(r),[5,2])[0]
    assert torch.equal(a,z);assert torch.equal(torch.autograd.grad(a,b)[0],torch.autograd.grad(z,b)[0])

def test_pl_only_uses_valid_frames():
    b=torch.tensor([[.4,.4,.2,.2],[.5,.5,.3,.3]],requires_grad=True);t=torch.tensor([[.3,.3,.2,.2],[.9,.9,.1,.1]])
    v=torch.tensor([True,False]);l=alignment(b,t,v,1,1);grad=torch.autograd.grad(l,b)[0]
    assert grad[0].abs().sum()>0 and grad[1].abs().sum()==0
    t[1]=0;assert torch.equal(l.detach(),alignment(b,t,v,1,1).detach())
