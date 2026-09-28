import torch
from vg_tta.desta3d_v3_a0_screen import coefficients,fixed_field,select_rows,terminal_decision
from vg_tta.desta3d_v3_gap_candidates import StateAwareDirectionMixer,coefficient_direction_loss

def test_exact_cached_path_and_gradient_scope():
    torch.manual_seed(20260928);torch.set_num_threads(2)
    basis=torch.linalg.qr(torch.randn(300,256)).Q
    m=StateAwareDirectionMixer(basis)
    c=dict(z=torch.randn(1,3,2,2,128,requires_grad=True),qT=torch.randn(1,128,requires_grad=True),qS=torch.randn(1,128),evidence8=torch.randn(1,3,2,2,8),state33=torch.randn(1,3,33))
    f=torch.randn(1,3,2,2,300)
    field,a=m(c['z'],c['qT'],c['qS'],c['evidence8'],c['state33'],f)
    b=coefficients(m,c);assert torch.equal(a,b)
    assert torch.equal(field,fixed_field(m,b,f.flatten().norm()))
    target=torch.randn_like(a);loss=coefficient_direction_loss(b,target)['cosine'];loss.backward()
    assert c['z'].grad is None and c['qT'].grad is None and m.basis.grad is None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())

def test_roster_order_independent_and_coverage():
    tr=[dict(source=str(p),key=f't{p}:{q}') for p in range(95) for q in range(7)]
    de=[dict(source='d'+str(p),key=f'd{p}:{q}') for p in range(31) for q in range(8)]
    a,b=select_rows(tr,de);aa,bb=select_rows(tr[::-1],de[::-1]);assert a==aa and b==bb
    assert len(a)==128 and len({r['source'] for r in a})==95
    assert len(b)==64 and len({r['source'] for r in b})==16
    assert all(sum(x['source']==p for x in b)==4 for p in {r['source'] for r in b})

def test_locked_gates_and_missing():
    assert terminal_decision(.299,.099)=='capacity_or_optimization'
    assert terminal_decision(.3,.099)=='conditioning_or_generalization'
    assert terminal_decision(-.5,.1)=='native_dev64'
    x=torch.ones(1,2,256,requires_grad=True)
    r=coefficient_direction_loss(x,torch.zeros_like(x));assert not r['valid'].any() and r['cosine']==0
