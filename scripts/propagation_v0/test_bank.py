"""Synthetic CPU-only v0 support tests; no real inputs or labels."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from scripts.propagation_v0.bank import CandidateBank, component_boxes, occupancy, transport


def fixture():
    T, H, W = 11, 5, 6; positions = [0, 2, 3, 4, 6, 7, 8, 10]
    visual = torch.zeros(T, H * W, 256, dtype=torch.float32)
    visual[:, torch.arange(H * W), torch.arange(H * W)] = 1
    valid = torch.ones(T, H * W, dtype=torch.bool)
    seeds = torch.zeros(8, H, W, dtype=torch.float64); seeds[:, 2, 1] = 1; seeds[:, 2, 4] = .9
    observed = torch.zeros(T, dtype=torch.bool); observed[positions] = True
    E = torch.tensor([.5, .5, .7, .7], dtype=torch.float64).repeat(T, 1)
    teacher = torch.tensor([1.5/W, 2.5/H, 1/W, 1/H], dtype=torch.float64).repeat(8, 1)
    evidence = SimpleNamespace(height=H, width=W, features=visual, valid=valid, seeds=seeds,
        frame_ids=torch.arange(T), positions=positions, observed=observed, event=torch.ones(T, dtype=torch.bool),
        transport_adjacent=lambda x: x.clone(), metadata=dict(GT_read=False, selected_native_query=7))
    cache = dict(E=E, frame_ids=list(range(T)), planned_positions=positions, interval=[0, T],
                 teacher=dict(boxes=teacher, selected_query=7), GT_read=False)
    return evidence, cache


def test_relative_components_four_connectivity_bbox_and_mass():
    p = torch.zeros(4, 5, dtype=torch.float64)
    p[0, 0], p[1, 1], p[1, 2], p[3, 4] = 1, .8, .7, .5
    out = component_boxes(p, torch.ones_like(p, dtype=torch.bool))
    assert len(out) == 2  # Diagonal is separate; exactly .5 peak is excluded.
    assert sorted(x['cells'] for x in out) == [1, 2]
    assert sorted(x['mass'] for x in out) == [1, 1.5]
    torch.testing.assert_close(out[1]['box'], torch.tensor([.4, .375, .4, .25], dtype=torch.float64))
    assert component_boxes(torch.zeros_like(p), torch.ones_like(p, dtype=torch.bool)) == []


def test_affinity_retains_dense_fp32_top20_operation_order():
    g = torch.Generator().manual_seed(4)
    x = F.normalize(torch.randn(2, 25, 256, generator=g), dim=-1)
    valid = torch.ones(2, 25, dtype=torch.bool); seed = torch.rand(25, generator=g, dtype=torch.float64)
    values, indices = torch.topk(x[0] @ x[1].T, 20, dim=1, sorted=False)
    weights = (values/.1).softmax(1).double(); weights /= weights.sum(1, keepdim=True)
    matrix = torch.zeros(25, 25, dtype=torch.float64)
    matrix[torch.arange(25)[:, None].expand_as(indices), indices] = weights
    assert torch.equal(transport(x, valid, seed, 0, 1), matrix @ seed)


def test_shapes_observed_teacher_priority_prior_and_detached_outputs():
    e, c = fixture(); b = CandidateBank(e, c, {})
    assert b.boxes.shape == (11, 3, 4) and b.features.shape == (11, 3, 256)
    assert b.link.shape == (10, 3, 3) and b.valid.shape == b.pi0.shape == (11, 3)
    assert torch.equal(b.boxes[:, 0], c['E'])
    for j, t in enumerate(c['planned_positions']):
        assert torch.equal(b.boxes[t, 1], c['teacher']['boxes'][j])
        assert b.candidate_origin[t, 1] == -2
        assert b.confidence[t] == 1
        assert torch.equal(b.merged_targets[t], e.seeds[j])
    torch.testing.assert_close(b.pi0.sum(-1), torch.ones(11, dtype=torch.float64))
    assert (b.pi0[~b.valid] == 0).all()
    for x in (b.boxes, b.features, b.anchor_features, b.score_unary, b.score_pair, b.confidence):
        assert x.dtype == torch.float64 and x.device.type == 'cpu' and not x.requires_grad and torch.isfinite(x).all()
    assert b.metadata['model_calls'] == b.metadata['decoder_calls'] == 0


def test_cycle_merge_does_not_multiply_forward_backward_agreement():
    e, c = fixture(); e.seeds[1].zero_(); e.seeds[1, 0, 5] = 1
    b = CandidateBank(e, c, {})
    t = 1
    r = b.side_reliability[t]
    expected = (b.side_maps[t] * r[:, None, None]).sum(0) / r.sum().clamp_min(1e-15)
    assert torch.equal(b.merged_targets[t], expected)
    assert b.confidence[t] == r.mean() and b.confidence[t] > .1
    assert b.merged_targets[t, 2, 1] > .1 and b.merged_targets[t, 0, 5] > .1


def test_empty_or_eventoutside_falls_back_only_E_and_prior_one():
    e, c = fixture(); e.seeds.zero_()
    b = CandidateBank(e, c, {})
    assert not b.active.any() and not b.valid[:, 1:].any()
    assert (b.pi0[:, 0] == 1).all() and (b.pi0[:, 1:] == 0).all()
    e, c = fixture(); e.event[9:] = False; c['interval'] = [0, 9]
    b = CandidateBank(e, c, {})
    assert not b.active[9:].any() and not b.valid[9:, 1:].any()


def test_bounded_guard_and_exact_additive_score_without_confidence_normalization():
    e, c = fixture(); b = CandidateBank(e, c, {})
    assert (b.guard >= 0).all() and (b.guard <= 1).all() and (b.guard[:, 0] == 0).all()
    actions = torch.tensor([int(torch.nonzero(row).flatten()[-1]) for row in b.valid])
    nodes = torch.arange(11); edges = torch.arange(10)
    u = b.projection[nodes, actions]; G = b.guard[nodes, actions]
    L = b.link[edges, actions[:-1], actions[1:]]
    obs, unobs = b.event & b.observed, b.event & ~b.observed
    edge = b.event[:-1] & b.event[1:]
    expected = (.5*u[obs].mean() + .5*(b.confidence*u)[unobs].mean()
                + .25*(b.confidence[:-1]*b.confidence[1:]*L)[edge].mean() - .1*G.mean())
    torch.testing.assert_close(b.score_path(actions), expected, atol=1e-15, rtol=1e-15)
    state = b.private_state(); state['boxes'].zero_()
    assert not torch.equal(state['boxes'], b.boxes)


def test_selected_identity_config_and_invalid_action_are_rejected():
    e, c = fixture(); c['teacher']['selected_query'] = 0
    with pytest.raises(ValueError, match='identity'): CandidateBank(e, c, {})
    e, c = fixture()
    with pytest.raises(ValueError, match='Locked bank'): CandidateBank(e, c, {'nms_iou': .98})
    b = CandidateBank(e, c, {})
    with pytest.raises(ValueError): b.score_path(torch.full((11,), 3))


def test_sealed_state_reconstruction_never_rebuilds_candidate_or_transport(monkeypatch):
    e, c = fixture(); b = CandidateBank(e, c, {})
    state = b.private_state()
    def forbidden(*args, **kwargs):
        raise AssertionError('from_state must not construct propagation or candidates')
    monkeypatch.setattr('scripts.propagation_v0.bank.transport', forbidden)
    monkeypatch.setattr('scripts.propagation_v0.bank.component_boxes', forbidden)
    restored = CandidateBank.from_state(state)
    for name in ('boxes','valid','features','side_maps','side_reliability','merged_targets',
                 'score_unary','score_pair','pi0','logprior','initial_memory'):
        assert torch.equal(getattr(restored,name),getattr(b,name))
    actions = torch.zeros(11,dtype=torch.long)
    assert restored.score_path(actions) == b.score_path(actions)
    state['boxes'].zero_()
    assert not torch.equal(restored.boxes,state['boxes'])
    with pytest.raises(ValueError): CandidateBank.from_state(state)
    state = b.private_state(); state['GT_read'] = True
    with pytest.raises(ValueError): CandidateBank.from_state(state)
