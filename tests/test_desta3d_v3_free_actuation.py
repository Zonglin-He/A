import pytest,torch
from vg_tta.desta3d_v3_free_actuation import inject_delta,span_basis,token_delta,native_ce,native_targets
from vg_tta.desta3d_v2 import Desta3DAdapterV2

def test_zero_and_frozen_gradient_scope():
    torch.manual_seed(3)
    a=Desta3DAdapterV2(in_channels=16,query_dim=16,hidden_dim=8,architecture='dual3d',p1_enabled=False).eval().requires_grad_(False)
    x=torch.randn(1,3,2,2,16);q=torch.randn(1,4,16);kwargs={'query_mask':torch.ones(1,4,dtype=torch.bool),'frame_times':torch.arange(3)[None].float()}
    old=a(x,q,**kwargs);p=torch.nn.Parameter(torch.zeros_like(x))
    with inject_delta(a,'event',p):
        new=a(x,q,**kwargs)
        assert torch.equal(old['updated_tokens_event'],new['updated_tokens_event'])
        assert torch.equal(old['updated_tokens_spatial'],new['updated_tokens_spatial'])
        new['updated_tokens_event'].square().sum().backward()
    assert p.grad is not None and p.grad.norm()>0
    assert all(v.grad is None and not v.requires_grad for v in a.parameters())
    assert torch.equal(a(x,q,**kwargs)['updated_tokens_event'],old['updated_tokens_event'])

def test_span_reparameterization_and_restore():
    torch.manual_seed(4);w=torch.randn(16,8);q,r=span_basis(w);p=torch.randn(1,2,2,2,8);g=.0025
    d=token_delta(p,'span',q)
    dz=torch.linalg.solve_triangular(r,(p.double()*2**.5).reshape(-1,8).T,upper=True).T/g
    actual=(dz@w.double().T*g).reshape_as(d)
    assert torch.allclose(actual,d.double(),atol=2e-6,rtol=2e-6)
    assert torch.allclose(q.T@q,torch.eye(8),atol=2e-6)

def test_targets_missing_support_and_class_denominator():
    lab={'response':'<|time_start|><t2><t3>','frame_ids':[1,2,3], 'boxes_xyxy':[[0,0,.5,1],[.1,.2,.3,.4],[0,0,0,0]],'box_valid':[1,1,0]}
    t,v=native_targets(lab,{},'event');assert t.tolist()==[1,2]
    x=torch.randn(2,3,requires_grad=True);assert torch.equal(native_ce(x,t,v),torch.nn.functional.cross_entropy(x,t))
    t,v=native_targets(lab,{'positions':[1,2]},'spatial');assert v.sum()==4
    x=torch.randn(2,4,1001,requires_grad=True);loss=native_ce(x,t,v);loss.backward();assert x.grad[1].abs().sum()==0
    with pytest.raises(ValueError):native_targets(lab,{'positions':[2]},'spatial')
