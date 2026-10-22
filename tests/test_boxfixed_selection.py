import torch
from test_boxfixed_tube_upstream import fixture
from tools.boxfixed_tube.spatial import build_geometry,evidence_dp
from tools.boxfixed_followup.selection_capture import capture_bank
from tools.boxfixed_followup.selection import signals,rescore


def test_bank_capture_matches_frozen_original_and_residual_execution():
    e,S,n=fixture();g=build_geometry(e,S,n)
    for mode in ['original','relative_residual']:
        p=dict(mode=mode,gamma=0.,admission=.1,guard=.4,link=1.,alpha=1.,beta=.5)
        bank,r=capture_bank(g,e,S.numpy(),S.numpy(),[0,31],p)
        assert torch.equal(evidence_dp(bank)['final'],r['final'])
        assert torch.equal(rescore(bank,dict(selection_mode='original'))['final'],r['final'])


def test_self_support_does_not_count_as_independent_evidence():
    e,S,n=fixture();g=build_geometry(e,S,n);bank=g.for_interval([0,31]);s=signals(bank)
    assert not s['independent'][bank.observed].any()
    # Native two identical references must not pass even with a zero margin.
    p=dict(selection_mode='cross_score',cycle_floor=0.,margin=0.,guard=.1)
    r=rescore(bank,p);assert r['actions'][bank.observed].eq(0).all()


def test_cross_advantage_ignores_own_reference_score_and_compares_same_reference():
    e,S,n=fixture();g=build_geometry(e,S,n);b=g.for_interval([0,31])
    before=signals(b)['advantage'].clone()
    b.extent_scores=b.extent_scores.clone();b.extent_scores[:,1,0]+=100
    assert torch.equal(signals(b)['advantage'],before)
    b.extent_scores[:,1,1]+=.2
    torch.testing.assert_close(signals(b)['advantage'][:,1],before[:,1]+.1)


def test_gates_keep_candidate_coordinates_and_input_bank_immutable():
    e,S,n=fixture();g=build_geometry(e,S,n);b=g.for_interval([0,31]);valid=b.valid.clone();unary=b.score_unary.clone();boxes=b.boxes.clone()
    p=dict(selection_mode='cross_agreement',cycle_floor=0.,margin=1.,agreement_floor=0.,guard=.1)
    r=rescore(b,p);assert r['actions'].eq(0).all()
    assert torch.equal(b.valid,valid) and torch.equal(b.score_unary,unary) and torch.equal(b.boxes,boxes)
    assert all(torch.equal(r['final'][t],b.boxes[t,r['actions'][t]])for t in range(b.frames))
