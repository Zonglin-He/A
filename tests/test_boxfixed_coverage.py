import torch
from test_boxfixed_tube_upstream import fixture
from tools.boxfixed_tube.spatial import build_geometry,evidence_dp
from tools.boxfixed_followup.coverage import appearance,geometric_valid,rescore_coverage


def test_geometric_admission_can_recover_preexisting_masked_candidates():
    e,S,n=fixture();b=build_geometry(e,S,n).for_interval([0,31]);original=b.valid.clone()
    b.valid[:,1:]=False;b.active.zero_();new=geometric_valid(b)
    assert new[~b.observed,1:].any() and not new[b.observed,1:].any()
    assert not b.valid[:,1:].any()


def test_zero_appearance_does_not_become_informative_by_renormalization():
    e,S,n=fixture();e.features.fill_(1.);b=build_geometry(e,S,n).for_interval([0,31]);a=appearance(b,e)
    assert a['cross'].abs().max()<1e-12 and a['own'].abs().max()<1e-12


def test_no_new_boxes_and_no_mutation_when_relaxing_old_mask():
    e,S,n=fixture();b=build_geometry(e,S,n).for_interval([0,31]);a=appearance(b,e);boxes=b.boxes.clone();valid=b.valid.clone();unary=b.score_unary.clone();active=b.active.clone()
    p=dict(policy='expanded',contrast=.5,cycle_floor=.1,unary_cycle_weight=False,guard=.4)
    r=rescore_coverage(b,a,p)
    assert torch.equal(r['final'],boxes[torch.arange(b.frames),r['actions']])
    assert torch.equal(b.boxes,boxes) and torch.equal(b.valid,valid) and torch.equal(b.score_unary,unary) and torch.equal(b.active,active)
    assert r['actions'][b.observed].eq(0).all()


def test_empty_background_is_unavailable_and_zero_signal_cannot_pass():
    e,S,n=fixture();b=build_geometry(e,S,n).for_interval([0,31]);b.native_boxes[:,2:]=3.
    a=appearance(b,e);assert not a['ring_available'].any()
    p=dict(policy='expanded',contrast=1.,cycle_floor=0.,unary_cycle_weight=False,guard=0.)
    assert rescore_coverage(b,a,p)['actions'].eq(0).all()
