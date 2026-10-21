"""Leave-self-out consistency on the existing eight cached observations.

No file, model, RGB or annotation is read. The existing full-box bank is never
changed. Cross-view consistency is not a verification of semantic identity.
"""
from __future__ import annotations
from collections import Counter
from collections.abc import Mapping

import torch
import torch.nn.functional as F

from scripts.box_propagation.bank import (
    affinity, fit_axis_affine, transform_native_box, extent_kernel,
    occupancy, soft_dice,
)

EPS = 1e-15


def _get(obj, name):
    return obj[name] if isinstance(obj, Mapping) else getattr(obj, name)


def _cpu(value, dtype):
    value = torch.as_tensor(value)
    assert value.device.type == 'cpu'
    return value.detach().to(dtype=dtype)


def _positions(bank):
    if isinstance(bank, Mapping):
        return [int(x) for x in bank['positions8']]
    return [int(x) for x in getattr(bank, 'positions8', getattr(bank, 'positions', []))]


def _roi_features(raw, boxes, valid, height, width):
    # Full continuous boxes, with fractional occupancy; no mask bbox or crop.
    occ, _, _ = occupancy(boxes, height, width)
    occ = occ * valid.reshape(1, height, width)
    mass = occ.flatten(-2)
    pooled = mass @ raw.double() / mass.sum(-1).clamp_min(EPS)[:, None]
    return F.normalize(pooled, dim=-1, eps=1e-12)


def _assemble(bank, cross):
    """Locked finite aggregation, separately testable without any propagation."""
    boxes = _cpu(_get(bank, 'boxes'), torch.float64)
    T = len(boxes)
    assert boxes.shape == (T, 3, 4)
    valid = _cpu(_get(bank, 'valid'), torch.bool)
    active = _cpu(_get(bank, 'active'), torch.bool)
    observed = _cpu(_get(bank, 'observed'), torch.bool)
    anchors = _cpu(_get(bank, 'source_anchors'), torch.long)
    reliability = _cpu(_get(bank, 'side_reliability'), torch.float64)
    extent = _cpu(_get(bank, 'extent_scores'), torch.float64)
    association = _cpu(_get(bank, 'association_scores'), torch.float64)
    cross = _cpu(cross, torch.float64)
    assert valid.shape == (T, 3) and active.shape == observed.shape == (T,)
    assert anchors.shape == reliability.shape == (T, 2)
    assert extent.shape == association.shape == (T, 3, 2)
    assert cross.shape == (8, 8) and torch.diagonal(cross).eq(0).all()
    assert torch.isfinite(cross).all() and ((cross >= 0) & (cross <= 1)).all()
    quality = (cross.sum(0) + cross.sum(1)) / 14.
    compatibility = torch.sqrt(cross * cross.T)
    compatibility = compatibility.clone()
    compatibility.fill_diagonal_(1.)
    witness_count = ((compatibility >= .1) & ~torch.eye(8, dtype=torch.bool)).sum(-1)
    source_verified = (quality >= .1) & (witness_count >= 2)
    routes = torch.zeros(T, 3, dtype=torch.long)
    assert ((anchors >= -1) & (anchors < 8)).all()
    routes[:, 1:] = torch.where(anchors >= 0, anchors + 1, torch.zeros_like(anchors))
    trust = torch.zeros(T, 3, dtype=torch.float64)
    trust[:, 1:] = torch.where(anchors >= 0, quality[anchors.clamp_min(0)],
                                torch.zeros_like(reliability))
    side_score = .5 * extent + .5 * association
    weighted = reliability * trust[:, 1:]
    # Keep the OLD reliability denominator: low external quality must not be
    # canceled by re-normalizing the new quality weights back to unit mass.
    external = (side_score * weighted[:, None]).sum(-1) / reliability.sum(-1).clamp_min(EPS)[:, None]
    delta = external - external[:, :1]
    verified_slots = torch.zeros(T, 3, dtype=torch.bool)
    verified_slots[:, 1:] = torch.where(anchors >= 0, source_verified[anchors.clamp_min(0)],
                                         torch.zeros_like(anchors, dtype=torch.bool))
    commit = active[:, None] & valid & verified_slots & (delta > 0)
    commit[:, 0] = False
    route_compat = torch.zeros(T, 9, 3, dtype=torch.float64)
    for action in (1, 2):
        target_route = routes[:, action]
        values = compatibility[:, (target_route - 1).clamp_min(0)].T
        route_compat[:, 1:, action] = torch.where(target_route[:, None] > 0,
                                                 values, torch.zeros_like(values))
    resets = torch.eye(9, dtype=torch.float64)[None].repeat(T, 1, 1)
    positions = _positions(bank)
    assert len(positions) == 8 and len(set(positions)) == 8
    for anchor, node in enumerate(positions):
        assert observed[node]
        if source_verified[anchor]:
            conflicting = torch.nonzero(compatibility[:, anchor] < .1, as_tuple=True)[0] + 1
            resets[node, conflicting] = 0.
            resets[node, conflicting, 0] = 1.
    assert torch.equal(resets[:, 0], torch.eye(9, dtype=torch.float64)[0].expand(T, -1))
    assert torch.equal(resets.sum(-1), torch.ones(T, 9, dtype=torch.float64))
    return dict(cross=cross.clone(), anchor_quality=quality,
        compatibility=compatibility, witness_count=witness_count, source_verified=source_verified,
        external_support=external, delta_support=delta,
        slot_routes=routes, candidate_trust=trust, commit=commit,
        route_compat=route_compat, reset_matrices=resets,
        anchor_incoming_support=cross.sum(0) / 7.,
        anchor_outgoing_support=cross.sum(1) / 7.)


@torch.no_grad()
def build(bank, evidence):
    """Return detached CPU evidence without changing any original bank field.

    ``cross[i,j]`` uses source mask i only for correspondences/WLS and verifies
    the propagated complete box against destination native extent and full-ROI
    features. Diagonal and invalid fits remain zero. All 56 directed checks are
    retained. reset_matrices[t,old_route,new_route] is row-stochastic; route0=E.
    """
    assert _get(bank, 'GT_read') is False if isinstance(bank, Mapping) else bank.metadata['GT_read'] is False
    assert _get(evidence, 'GT_read') is False if isinstance(evidence, Mapping) else evidence.metadata['GT_read'] is False
    positions = _positions(bank)
    raw = _cpu(_get(evidence, 'features'), torch.float32)
    valid = _cpu(_get(evidence, 'valid'), torch.bool)
    seeds = _cpu(_get(evidence, 'seeds'), torch.float64).clone()
    assert seeds.ndim == 3 and seeds.shape[0] == 8
    height, width = seeds.shape[-2:]
    T, N, D = raw.shape
    assert D == 256 and N == height * width and valid.shape == (T, N)
    assert len(positions) == 8 and all(0 <= p < T for p in positions)
    assert torch.equal(_cpu(_get(bank, 'frame_ids'), torch.long),
                       _cpu(_get(evidence, 'frame_ids'), torch.long))
    evidence_positions = _positions(evidence)
    assert positions == evidence_positions
    assert torch.isfinite(raw).all() and torch.isfinite(seeds).all()
    assert ((seeds >= 0) & (seeds <= 1)).all()
    seeds *= valid[positions].reshape_as(seeds)
    native = _cpu(_get(bank, 'native_boxes'), torch.float64)
    assert native.shape == (8, 4) and torch.isfinite(native).all() and (native[:, 2:] > 0).all()
    assert torch.equal(_cpu(_get(bank, 'boxes'), torch.float64)[positions, 1], native)
    normalized = F.normalize(raw, dim=-1, eps=1e-12)
    yy, xx = torch.meshgrid((torch.arange(height, dtype=torch.float64) + .5) / height,
                            (torch.arange(width, dtype=torch.float64) + .5) / width, indexing='ij')
    xy = torch.stack((xx, yy), -1).reshape(N, 2)
    cross = torch.zeros(8, 8, dtype=torch.float64)
    cycle = torch.zeros_like(cross)
    extents = torch.zeros_like(cross)
    cosines = torch.zeros_like(cross)
    fit_valid = torch.zeros(8, 8, dtype=torch.bool)
    transported_boxes = torch.zeros(8, 8, 4, dtype=torch.float64)
    fit_info = [[None for _ in range(8)] for _ in range(8)]
    native_features = torch.stack([_roi_features(raw[node], native[j:j+1], valid[node], height, width)[0]
                                   for j, node in enumerate(positions)])
    reasons = Counter()
    for i, source in enumerate(positions):
        mask = seeds[i].flatten()
        for j, target in enumerate(positions):
            if i == j:
                fit_info[i][j] = dict(valid=False, reason='excluded_self')
                continue
            target_source = affinity(normalized, valid, target, source)
            source_target = affinity(normalized, valid, source, target)
            propagated_mask = target_source @ mask
            returned = source_target @ propagated_mask
            reliability = soft_dice(returned, mask)
            joint = source_target * target_source.T * mask[:, None]
            fitted, info = fit_axis_affine(joint, xy, xy)
            fit_info[i][j] = info
            if fitted is None:
                reasons[info['reason']] += 1
                continue
            transformed = transform_native_box(native[i], *fitted)
            if transformed is None:
                info.update(valid=False, reason='invalid_or_fully_off_image_complete_box')
                reasons[info['reason']] += 1
                continue
            feature = _roi_features(raw[target], transformed[None], valid[target], height, width)[0]
            extent = extent_kernel(transformed, native[j])
            cosine = (feature * native_features[j]).sum().clamp(0, 1)
            value = reliability * (.5 * extent + .5 * cosine)
            assert torch.isfinite(value) and 0 <= value <= 1
            cross[i, j] = value
            cycle[i, j], extents[i, j], cosines[i, j] = reliability, extent, cosine
            transported_boxes[i, j], fit_valid[i, j] = transformed, True
            reasons['accepted'] += 1
    result = _assemble(bank, cross)
    result.update(pair_cycle=cycle, pair_extent=extents, pair_full_ROI_cosine=cosines,
        pair_valid=fit_valid, pair_transformed_boxes=transported_boxes,
        pair_fit_diagnostics=fit_info, native_ROI_features=native_features,
        positions8=list(positions), GT_read=False,
        metadata=dict(schema='committed_cross_anchor_evidence_v1', GT_read=False,
            anchor_checks=56, fit_reason_counts=dict(reasons),
            source_affinity='Original FP32 normalized dot/top20/softmax(.1), then FP64 weights',
            pair_score='fitvalid*cycle*(.5 fullbox extent RBF+.5 positive destination fullROI cosine)',
            self_checks='cross diagonal zero; excluded from fixed quality denominator14',
            quality='sum incoming and outgoing divided14; invalid pairs remain zero',
            compatibility='sqrt(cross*cross.T); diagonal1 for same-source routing only',
            external_support='sum(old_side_reliability*q_source*old_side_score)/sum(old_side_reliability)',
            commit='old active & old valid & q_source>=.1 & at_least2_other_C>=.1 & candidate_external_support-E_external_support>0',
            route_ids='0=E,1..8=original native observation index+1; missing side=0',
            reset_orientation='[physical_node,old_route,new_route]; conflicting old source goes to E',
            reset='At observation j, if source_verified_j and C_old,j<.1, old_route->E; otherwise identity',
            anchor_quality_mean=float(result['anchor_quality'].mean()),
            verified_anchors=int(result['source_verified'].sum()),
            valid_active_alternative_slots=int((_cpu(_get(bank,'valid'),torch.bool)[:,1:] & _cpu(_get(bank,'active'),torch.bool)[:,None]).sum()),
            committed_alternative_slots=int(result['commit'].sum()),
            quality_or_witness_suppressed_slots=int((_cpu(_get(bank,'valid'),torch.bool)[:,1:] & _cpu(_get(bank,'active'),torch.bool)[:,None] &
                ~result['source_verified'][_cpu(_get(bank,'source_anchors'),torch.long).clamp_min(0)]).sum()),
            old_bank_modified=False, selected_identity_verified=False,
            limitation='Correlated eight-frame teacher consistency, not independent teacher inference or semantic correctness',
            model_calls=0, decoder_calls=0, expert_calls=0, RGB_reads=0))
    for value in result.values():
        if torch.is_tensor(value):
            assert value.device.type == 'cpu' and not value.requires_grad
            if value.is_floating_point():
                assert value.dtype == torch.float64 and torch.isfinite(value).all()
    return result
