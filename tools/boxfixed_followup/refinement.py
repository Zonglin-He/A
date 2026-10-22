"""Two bounded, tensor-only candidate refinements on immutable Source evidence.

The frozen upstream geometry/scoring/DP remains untouched. These candidates are
experimental output-side corrections, not new neural-network predictions.
"""
import copy
import torch
import torch.nn.functional as F
from tools.boxfixed_tube.profiles import view_geometry, predict as original_predict
from tools.boxfixed_tube.spatial import BoxFixedBank, evidence_dp
from tools.boxfixed_tube._spatial_math import occupancy, extent_kernel, soft_dice, EPS, scoped_single_thread


def refined_box(source_now, source_anchor, teacher_anchor, propagated, alpha, beta, mode):
    if mode == 'center_preserve':
        center = source_now[:2] + alpha * (propagated[:2] - source_now[:2])
        size = source_now[2:] * (beta * (propagated[2:] / source_now[2:]).log()).exp()
    elif mode == 'relative_residual':
        center = source_now[:2] + alpha * source_now[2:] * (teacher_anchor[:2] - source_anchor[:2]) / source_anchor[2:]
        size = source_now[2:] * (beta * (teacher_anchor[2:] / source_anchor[2:]).log()).exp()
    else:
        raise ValueError(mode)
    return torch.cat((center, size))


@scoped_single_thread
def predict(base, evidence, Source, raw, interval, parameters):
    p = {k: parameters[k] for k in ['gamma', 'admission', 'guard', 'link']}
    mode = parameters['mode']
    if mode == 'original':
        return original_predict(base, evidence, Source, raw, interval, p)
    assert p['gamma'] == 0., 'Refinements isolate candidate transport from inherited ERef calibration'
    source = torch.as_tensor(Source, dtype=torch.float64)
    g = copy.copy(view_geometry(base, evidence, Source, raw, p))
    g.metadata = dict(g.metadata, candidate_refinement=mode)
    g.side_available = g.side_available.clone()
    g.side_reference_boxes = g.side_reference_boxes.clone()
    # Native observed boxes are retained unchanged. Only unobserved side proposals
    # are replaced, with no GT, query-type, identity or outcome-conditioned rule.
    for t in range(g.frames):
        if g.observed[t]:
            continue
        for side in (0, 1):
            if not g.side_available[t, side]:
                continue
            j = int(g.source_anchors[t, side]); a = g.positions[j]
            b = refined_box(source[t], source[a], g.native_boxes[j],
                            g.side_reference_boxes[t, side], parameters['alpha'], parameters['beta'], mode)
            lo, hi = b[:2] - b[2:] / 2, b[:2] + b[2:] / 2
            if not (torch.isfinite(b).all() and (b[2:] > 0).all() and ((hi > 0) & (lo < 1)).all()):
                g.side_available[t, side] = False
                continue
            g.side_reference_boxes[t, side] = b
    # Re-evaluate the exact same upstream evidence score on the changed boxes.
    # Reusing old pooled features/guard/link here would be an invalid comparison.
    T = g.frames
    g.boxes = source[:, None].expand(-1, 3, -1).clone()
    for side in (0, 1):
        g.boxes[g.side_available[:, side], side + 1] = g.side_reference_boxes[g.side_available[:, side], side]
    g.reliable = (g.confidence >= p['admission']) & g.side_available.any(-1)
    g.valid = torch.zeros(T, 3, dtype=torch.bool); g.valid[:, 0] = True
    g.candidate_origin = torch.full((T, 3), -1, dtype=torch.long)
    g.candidate_quality = torch.zeros(T, 3, dtype=torch.float64)
    for t in range(T):
        for side in (0, 1):
            if g.side_available[t, side] and (g.reliable[t] or g.observed[t]):
                g.valid[t, side + 1] = True
                g.candidate_origin[t, side + 1] = -2 if g.observed[t] else side
                g.candidate_quality[t, side + 1] = g.side_reliability[t, side]
        if g.valid[t, 1] and g.valid[t, 2] and torch.equal(g.boxes[t, 1], g.boxes[t, 2]):
            g.valid[t, 2] = False
    occ, _, _ = occupancy(g.boxes, g.height, g.width)
    occ *= evidence.valid.reshape(T, 1, g.height, g.width)
    pooled = torch.bmm(occ.flatten(-2), evidence.features.double()) / occ.flatten(-2).sum(-1).clamp_min(EPS)[..., None]
    g.features = F.normalize(pooled, dim=-1, eps=1e-12); g.E_features = g.features[:, 0]
    g.extent_scores = extent_kernel(g.boxes[:, :, None], g.side_reference_boxes[:, None])
    g.association_scores = torch.einsum('tkd,tsd->tks', g.features, g.side_prototypes).clamp(0, 1)
    side_score = .5 * g.extent_scores + .5 * g.association_scores
    g.projection = (side_score * g.side_reliability[:, None]).sum(-1) / g.side_reliability.sum(-1).clamp_min(EPS)[:, None]
    g.u = g.projection
    dz = torch.cat(((g.boxes[..., :2] - source[:, None, :2]) / source[:, None, 2:],
                    g.boxes[..., 2:].log() - source[:, None, 2:].log()), -1)
    g.guard = -torch.expm1(-(torch.linalg.vector_norm(dz, dim=-1) / 2).square())
    masked = (occ * g.merged_targets[:, None]).flatten(-2)
    previous = evidence.transport_adjacent(masked[:-1].transpose(0, 1)).transpose(0, 1)
    g.link = soft_dice(previous[:, :, None], masked[1:, None])
    bank = BoxFixedBank(g, interval=interval)
    obs = bank.event & bank.observed; unobs = bank.event & ~bank.observed
    w = .5 * obs.double() / max(1, int(obs.sum())) + .5 * bank.confidence * unobs / max(1, int(unobs.sum()))
    bank.score_unary = w[:, None] * bank.projection - p['guard'] * bank.guard / T
    ew = p['link'] * bank.event_edges * bank.confidence[:-1] * bank.confidence[1:] / max(1, int(bank.event_edges.sum()))
    bank.score_pair = ew[:, None, None] * bank.link
    r = evidence_dp(bank)
    r.update(candidate_admitted_frames=int(bank.active.sum()), changed_from_ERef_frames=int((r['final'] != source).any(-1).sum()))
    return r
