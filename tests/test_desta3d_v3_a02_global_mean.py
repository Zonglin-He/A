import torch
from torch.nn import functional as F
from vg_tta.desta3d_v3_a02_global_mean import GlobalMeanDirectionMixer,coefficients,route
from vg_tta.desta3d_v3_a01_triage import FitabilityDirectionMixer
from vg_tta.desta3d_v3_a0_screen import coefficients as old_coefficients


def test_no_new_parameters_and_disabled_exact():
    torch.set_num_threads(2);basis=torch.eye(256)
    torch.manual_seed(20260928);a=FitabilityDirectionMixer(basis)
    torch.manual_seed(20260928);b=GlobalMeanDirectionMixer(basis)
    assert sum(p.numel() for p in b.parameters())==107648
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    c=dict(z=torch.randn(1,4,3,2,128,requires_grad=True),qT=torch.randn(1,128),qS=torch.randn(1,128),
           evidence8=torch.randn(1,4,3,2,8),state33=torch.randn(1,4,33))
    assert torch.equal(old_coefficients(a,c),coefficients(b,c,global_mean=False))
    # Independent explicit broadcast formula matches the enabled implementation.
    shape=c['z'].shape[:-1]
    q=[c[k][:,None,None,None].expand(*shape,-1) for k in ('qT','qS')]
    st=c['state33'][:,:,None,None].expand(*shape,-1)
    h0=F.silu(b.input(torch.cat((c['z'].detach(),*q,c['evidence8'],st),-1)))
    g=h0.flatten(1,3).mean(1)[:,None,None,None,:]
    h=h0+F.silu(b.local(h0.movedim(-1,1)).movedim(1,-1))+g
    expected=b.output(F.silu(b.mix(h)))
    out=coefficients(b,c);assert torch.equal(out,expected)
    out.square().mean().backward()
    assert c['z'].grad is None and b.basis.grad is None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in b.parameters())


def test_remote_context_changes_a_local_output():
    torch.manual_seed(8);m=GlobalMeanDirectionMixer(torch.eye(256));shape=(1,7,3,3)
    c=dict(z=torch.randn(*shape,128),qT=torch.randn(1,128),qS=torch.randn(1,128),
           evidence8=torch.randn(*shape,8),state33=torch.randn(1,7,33))
    other={k:v.clone() for k,v in c.items()};other['z'][:,6]+=3
    a=coefficients(m,c,global_mean=False);b=coefficients(m,other,global_mean=False)
    assert torch.equal(a[:,0],b[:,0])
    assert not torch.equal(coefficients(m,c)[:,0],coefficients(m,other)[:,0])


def test_fixed_route_priority():
    bad={'train':{'median':.01},'dev':{'median':.002}}
    good={'train':{'median':.3},'dev':{'median':.1}}
    assert route(good,good)=='native_dev64:GMean-S200'
    assert route(bad,good)=='native_dev64:GMean-S2000'
    assert route(bad,bad)=='stop_architecture_tuning_cached_structure_audit'
