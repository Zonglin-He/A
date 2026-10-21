import pytest
import torch
from tools.universal_trial.core import VideoEvidence,evidence_subspace,initial_candidates,component_update


def fixture(features=None):
    f=torch.eye(4,dtype=torch.double)[None].repeat(10,1,1)if features is None else features
    cells=torch.tensor([[0.,0.,.5,.5],[.5,0.,1.,.5],[0.,.5,.5,1.],[.5,.5,1.,1.]],dtype=torch.double)
    seeds=torch.tensor([[.7,.1,.1,.1]]*8,dtype=torch.double)
    source=torch.tensor([[.43,.39,.32,.24]]*10,dtype=torch.double)
    return VideoEvidence(f,cells,torch.arange(10,dtype=torch.double),[0,1,2,3,6,7,8,9],seeds,source,[0,10])


def test_propagation_preserves_eight_separate_probability_hypotheses():
    e=fixture()
    assert e.density.shape==(8,10,4)
    assert torch.allclose(e.density.sum(-1),torch.ones(8,10,dtype=e.density.dtype),atol=1e-5)
    assert bool(((e.confidence>=0)&(e.confidence<=1)).all())
    assert (e.density[0,0]-e.density[0,-1]).abs().max()<1e-4


def test_uniform_visual_features_do_not_claim_supported_unobserved_identity():
    e=fixture(torch.ones(10,4,6,dtype=torch.double))
    assert e.evidence_strength[4].item()==pytest.approx(0,abs=1e-6)
    assert e.evidence_strength[5].item()==pytest.approx(0,abs=1e-6)


def test_same_global_trajectory_reward_has_direct_gradient():
    e=fixture();boxes=e.source.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(lambda b:e.score(b)['total'],(boxes,),eps=1e-4,atol=1e-4)
    g=torch.autograd.grad(e.score(boxes)['total'],boxes)[0]
    assert torch.isfinite(g).all()and g.abs().sum()>0


def test_gradient_subspace_and_candidates_respect_common_parameter_bounds():
    grads=[torch.arange(1.,21.)*(i+1)for i in range(8)]
    U,gram=evidence_subspace(grads);u=initial_candidates(8,U.device,U.dtype)
    assert U.shape==(20,8)and gram.shape==(8,8)and len(u)==21
    assert bool(((u@U.T).abs()<=.050001).all())
    assert bool((u.abs().sum(-1)<=1.000001).all())


def test_policy_components_remain_separate_parameter_states():
    u=initial_candidates(8,'cpu',torch.double);r=torch.arange(len(u),dtype=torch.double)/10
    means,w=component_update(u,r)
    assert means.shape==(8,8)and w.shape==(8,3)
    assert torch.equal(means-torch.diag(means.diag()),torch.zeros_like(means))
    assert bool((means.abs().sum(-1)<=1).all())
