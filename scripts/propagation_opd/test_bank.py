"""Synthetic CPU bank tests: no model, checkpoint, real query, or labels."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from scripts.propagation_opd.bank import (
    CandidateBank, anchor_transport, best_projection_interval, box_iou,
    fractional_occupancy, projection_box, secondary_peak,
)


def fixture():
    T, H, W = 11, 5, 6
    positions = [0, 2, 3, 4, 6, 7, 8, 10]
    features = torch.zeros(T, H * W, 256, dtype=torch.float32)
    features[:, torch.arange(H * W), torch.arange(H * W)] = 1
    valid = torch.ones(T, H * W, dtype=torch.bool)
    seeds = torch.zeros(8, H, W, dtype=torch.float64)
    seeds[:, 2, 1] = 1
    seeds[:, 2, 4] = .9
    targets = seeds[:1].expand(T, -1, -1).clone()
    observed = torch.zeros(T, dtype=torch.bool)
    observed[positions] = True
    event = torch.ones(T, dtype=torch.bool)
    confidence = torch.full((T,), .5, dtype=torch.float64)
    confidence[observed] = 1
    teacher = torch.tensor([1.5 / W, 2.5 / H, 1 / W, 1 / H], dtype=torch.float64).repeat(8, 1)
    E = torch.tensor([.5, .5, .7, .7], dtype=torch.float64).repeat(T, 1)

    def transport(previous):
        # A fixed identity graph has no trainable tensor or model call.
        return previous.clone()

    evidence = SimpleNamespace(
        height=H, width=W, features=features, valid=valid, seeds=seeds,
        targets=targets, confidence=confidence, event=event, observed=observed,
        frame_ids=torch.arange(T), positions=positions, transport_adjacent=transport,
        metadata=dict(GT_read=False, selected_native_query=5))
    cache = dict(E=E, frame_ids=list(range(T)), planned_positions=positions,
                 teacher=dict(boxes=teacher, selected_query=5), interval=[0, T], GT_read=False)
    return evidence, cache


def test_projection_interval_full_enumeration_has_correct_dice_denominator():
    p = torch.tensor([0., 1., 1., 0., 0.], dtype=torch.float64)
    assert best_projection_interval(p) == (.2, .6)
    assert best_projection_interval(torch.zeros(5)) is None
    # A high but broad plateau is retained; mean-intensity and total-coverage
    # heuristics are not substituted for Dice optimization.
    assert best_projection_interval(torch.ones(5)) == (0., 1.)
    b = projection_box(torch.tensor([[0., 0., 0.], [0., 1., 0.]], dtype=torch.float64))
    torch.testing.assert_close(b, torch.tensor([.5, .75, 1/3, .5], dtype=torch.float64))


def test_fractional_roi_pooling_uses_cell_fractions():
    occupancy, ox, oy = fractional_occupancy(torch.tensor([.25, .5, .25, 1.], dtype=torch.float64), 2, 2)
    torch.testing.assert_close(ox, torch.tensor([.5, 0.], dtype=torch.float64))
    torch.testing.assert_close(oy, torch.ones(2, dtype=torch.float64))
    torch.testing.assert_close(occupancy, torch.tensor([[.5, 0.], [.5, 0.]], dtype=torch.float64))


def test_secondary_peak_requires_locality_distance_ratio_and_valid_cell():
    p = torch.zeros(5, 6, dtype=torch.float64)
    p[2, 1], p[2, 4] = 1, .85
    valid = torch.ones_like(p, dtype=torch.bool)
    assert secondary_peak(p, valid) == (2, 4)
    p[2, 4] = .79
    assert secondary_peak(p, valid) is None
    p[2, 4] = .85
    valid[2, 4] = False
    assert secondary_peak(p, valid) is None
    p.zero_(); valid.fill_(True); p[2, 1], p[2, 2] = 1, .9
    assert secondary_peak(p, valid) is None


def test_original_affinity_fp32_and_weight_transport_match_dense_reference():
    generator = torch.Generator().manual_seed(11)
    x = F.normalize(torch.randn(2, 25, 256, generator=generator), dim=-1)
    valid = torch.ones(2, 25, dtype=torch.bool)
    valid[0, -2:] = False
    seed = torch.rand(25, generator=generator, dtype=torch.float64)
    target_ids = torch.nonzero(valid[0], as_tuple=False).flatten()
    source_ids = torch.nonzero(valid[1], as_tuple=False).flatten()
    cosine = x[0, target_ids] @ x[1, source_ids].T
    values, indices = torch.topk(cosine, 20, dim=1, sorted=False)
    weights = (values / .1).softmax(1).double()
    weights /= weights.sum(1, keepdim=True)
    matrix = torch.zeros(25, 25, dtype=torch.float64)
    matrix[target_ids[:, None].expand_as(indices), source_ids[indices]] = weights
    assert torch.equal(anchor_transport(x, valid, seed, 0, 1), matrix @ seed)


def test_bank_shapes_detached_features_native_observations_and_all_nodes():
    evidence, cache = fixture()
    bank = CandidateBank(evidence, cache, {})
    assert bank.boxes.shape == (11, 4, 4) and bank.valid.shape == (11, 4)
    assert bank.features.shape == (11, 4, 256) and bank.anchor_features.shape == (11, 256)
    assert bank.link.shape == (10, 4, 4) and bank.projection.shape == bank.guard.shape == (11, 4)
    assert bank.candidates is bank.boxes
    assert torch.equal(bank.boxes[:, 0], cache['E']) and bank.valid[:, 0].all()
    for j, t in enumerate(cache['planned_positions']):
        assert torch.equal(bank.boxes[t, 1], cache['teacher']['boxes'][j])
        assert torch.equal(bank.boxes[t, 2], cache['teacher']['boxes'][j])
        assert bank.valid[t, 1] and not bank.valid[t, 2]  # Native track duplicate.
    for value in (bank.boxes, bank.features, bank.anchor_features, bank.projection, bank.guard, bank.link):
        assert value.device.type == 'cpu' and value.dtype == torch.float64 and not value.requires_grad
        assert torch.isfinite(value).all()
    norms = bank.features.norm(dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms))
    assert bank.metadata['model_calls'] == bank.metadata['decoder_calls'] == 0


def test_inactive_lowconfidence_eventoutside_or_absent_evidence_have_only_E():
    evidence, cache = fixture()
    evidence.confidence[1] = .09
    evidence.event[9] = False
    cache['interval'] = [0, 9]
    evidence.event[10] = False
    evidence.targets[5].zero_()
    bank = CandidateBank(evidence, cache, {})
    for t in (1, 5, 9, 10):
        assert not bank.active[t] and bank.valid[t, 0] and not bank.valid[t, 1:].any()
    # An observed frame within the event remains eligible independently of its
    # stored confidence; downstream policy must gate memory using bank.active.
    evidence.confidence[0] = 0
    bank = CandidateBank(evidence, cache, {})
    assert bank.active[0]
    # The pre-run configuration threshold is read, not silently hardcoded.
    bank = CandidateBank(evidence, cache, {'confidence_threshold': .8})
    assert not bank.active[1] and not bank.active[5]


def test_no_observed_teacher_identity_reselection_or_gt_inputs():
    evidence, cache = fixture()
    cache['teacher']['selected_query'] = 4
    with pytest.raises(ValueError, match='identity changed'):
        CandidateBank(evidence, cache, {})
    evidence, cache = fixture()
    cache['GT_read'] = True
    with pytest.raises(ValueError, match='label-free'):
        CandidateBank(evidence, cache, {})
    evidence, cache = fixture()
    with pytest.raises(ValueError, match='Locked candidate-bank'):
        CandidateBank(evidence, cache, {'candidate_bank': {'duplicate_iou': .9}})
    with pytest.raises(ValueError, match='secondary_peak_distance_cells'):
        CandidateBank(evidence, cache, {'secondary_peak_distance_cells': 3.})
    bank = CandidateBank(evidence, cache, {'secondary_peak_distance_cells': 2.})
    assert bank.metadata['thresholds']['secondary_peak_min_distance'] == 2.


def test_link_is_all_previous_current_pairs_and_private_state_is_immutable_copy():
    evidence, cache = fixture()
    bank = CandidateBank(evidence, cache, {})
    occ, _, _ = fractional_occupancy(bank.boxes, 5, 6)
    masked = (occ * evidence.targets[:, None]).flatten(-2)
    for t in range(10):
        for previous in range(4):
            for current in range(4):
                a, b = masked[t, previous], masked[t + 1, current]
                expected = (2 * (a * b).sum() / (a.sum() + b.sum()).clamp_min(1e-15)).clamp(0, 1)
                assert bank.link[t, previous, current] == expected
    state = bank.private_state()
    state['boxes'].zero_()
    assert not torch.equal(state['boxes'], bank.boxes)
    assert state['GT_read'] is False
    assert abs(float(box_iou(bank.boxes[0, 1], bank.boxes[0, 2])) - 1.) < 1e-12
