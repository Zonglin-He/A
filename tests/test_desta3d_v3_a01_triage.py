import torch
from vg_tta.desta3d_v3_gap_candidates import StateAwareDirectionMixer, coefficient_direction_loss
from vg_tta.desta3d_v3_a0_screen import coefficients
from vg_tta.desta3d_v3_a01_triage import FitabilityDirectionMixer, q1_index, route


def fixture():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    return torch.randn(300,256), dict(z=torch.randn(1,3,2,2,128,requires_grad=True),
        qT=torch.randn(1,128,requires_grad=True), qS=torch.randn(1,128,requires_grad=True),
        evidence8=torch.randn(1,3,2,2,8), state33=torch.randn(1,3,33))


def test_h128_seed_and_three_adam_steps_exact():
    basis,c=fixture()
    torch.manual_seed(20260928);a=StateAwareDirectionMixer(basis)
    torch.manual_seed(20260928);b=FitabilityDirectionMixer(basis)
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    target=torch.randn(1,3,2,2,256)
    opts=[torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=0) for m in (a,b)]
    for _ in range(3):
        for m,o in zip((a,b),opts):
            o.zero_grad();coefficient_direction_loss(coefficients(m,c),target)['cosine'].backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1);o.step()
        assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    assert c['z'].grad is None and c['qT'].grad is None and a.basis.grad is None


def test_width256_preserves_feature_support_and_scope():
    basis,c=fixture();m=FitabilityDirectionMixer(basis,hidden_dim=256)
    assert m.input.in_features==425 and m.input.out_features==256
    a=coefficients(m,c);assert a.shape==(1,3,2,2,256)
    coefficient_direction_loss(a,torch.randn_like(a))['cosine'].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    assert c['z'].grad is None and c['qT'].grad is None and c['qS'].grad is None and m.basis.grad is None


def test_fixed_hash_and_routing():
    rows=[{'key':str(i)} for i in range(128)]
    assert rows[q1_index(rows)]==list(reversed(rows))[q1_index(list(reversed(rows)))]
    bad={'train':{'median':.02},'dev':{'median':.01}}
    good={'train':{'median':.3},'dev':{'median':.1}}
    assert route(good,good,.1)=='capacity_candidate_next_dev64_native'
    assert route(bad,good,.1)=='optimization_budget_candidate_next_dev64_native'
    assert route(bad,bad,.9)=='stop_width_steps_next_global_context_conditioning'
    assert route(bad,bad,.89)=='stop_mixer_capacity_tuning_next_cached_oracle_structure_audit'
