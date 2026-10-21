import pytest
import torch
from tools.cdf_extent_reward import CDFExtentReward


def test_uniform_valid_mass_has_exact_quantile_extent():
    r=CDFExtentReward(torch.ones(1,1,3,4,dtype=torch.double),torch.tensor([[.75,.6]]))
    b=torch.tensor([[[.5,.5,.95,.95]]],dtype=torch.double)
    assert r.scores(b).item()==pytest.approx(0,abs=1e-15)
    assert r.cdf(b)[0,0].tolist()==pytest.approx([.025,.975,.025,.975])


def test_padding_mass_cannot_change_valid_image_marginals():
    a=torch.tensor([[[[.2,.2,1.,1.],[1.,1.,1.,1.]]]],dtype=torch.double)
    r=CDFExtentReward(a,torch.tensor([[.5,.5]]));b=torch.tensor([[[.4,.6,.3,.4]]],dtype=torch.double)
    assert r.cdf(b)[0,0].tolist()==pytest.approx([.25,.55,.4,.8])


def test_fractional_canvas_integrals_are_exact():
    r=CDFExtentReward(torch.tensor([[[[.1,.2],[.3,.4]]]],dtype=torch.double),torch.tensor([[.75,.6]],dtype=torch.double))
    assert r.total.item()==pytest.approx(.3)
    assert r.cdf(torch.tensor([[[.75,.75,.5,.5]]],dtype=torch.double))[0,0].tolist()==pytest.approx([.4,1.,.4,1.])


def test_same_continuous_direct_gradient_matches_finite_difference():
    r=CDFExtentReward(torch.tensor([[[[.1,.7],[.8,.2]]]],dtype=torch.double),torch.ones(1,2))
    b=torch.tensor([[[.43,.39,.32,.24]]],dtype=torch.double,requires_grad=True)
    assert torch.autograd.gradcheck(r.scores,(b,),eps=1e-6,atol=1e-6)
    assert torch.autograd.grad(r.scores(b).sum(),b)[0].abs().sum()>0


def test_zero_mass_is_uninformative_and_off_image_saturates():
    b=torch.tensor([[[.5,.5,1.,1.],[.5,.5,3.,3.]]],dtype=torch.double,requires_grad=True)
    empty=CDFExtentReward(torch.zeros(1,1,2,2,dtype=torch.double),torch.ones(1,2))
    assert torch.equal(empty.scores(b),torch.zeros(1,2,dtype=torch.double))
    assert torch.equal(torch.autograd.grad(empty.scores(b).sum(),b)[0],torch.zeros_like(b))
    full=CDFExtentReward(torch.ones(1,1,2,2,dtype=torch.double),torch.ones(1,2))
    assert full.scores(b)[0].tolist()==pytest.approx([-.025,-.025])
