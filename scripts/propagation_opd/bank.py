"""Fixed four-slot trajectory support from cached single-instance evidence.

Construction is CPU-only and reads no files, RGB, labels or models. Affinity
selection/softmax retains the original DenseEvidence FP32 operation order;
mask transport, boxes, pooled features and feedback are detached CPU FP64.
Invalid/inactive slots must be masked by the downstream policy. Inactive nodes
must also leave that policy's recurrent identity memory unchanged.
"""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F


FIXED = dict(confidence_threshold=.1, affinity_topk=20, affinity_temperature=.1,
             duplicate_iou=.98, secondary_peak_min_distance=2., secondary_peak_ratio=.8)
EPS = 1e-15


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _cpu(value, dtype=None):
    if torch.is_tensor(value):
        _require(value.device.type == 'cpu', 'CandidateBank accepts CPU evidence only')
    return torch.as_tensor(value, dtype=dtype).detach().clone()


def fractional_occupancy(boxes, height, width):
    """Fraction of each grid cell covered by normalized cxcywh boxes."""
    _require(boxes.device.type == 'cpu' and boxes.dtype == torch.float64,
             'Candidate geometry must use CPU FP64')
    x0, y0 = (boxes[..., :2] - boxes[..., 2:] / 2).clamp(0, 1).unbind(-1)
    x1, y1 = (boxes[..., :2] + boxes[..., 2:] / 2).clamp(0, 1).unbind(-1)
    x = torch.arange(width + 1, dtype=torch.float64) / width
    y = torch.arange(height + 1, dtype=torch.float64) / height
    ox = (torch.minimum(x1[..., None], x[1:]) - torch.maximum(x0[..., None], x[:-1])).clamp_min(0) * width
    oy = (torch.minimum(y1[..., None], y[1:]) - torch.maximum(y0[..., None], y[:-1])).clamp_min(0) * height
    return oy[..., :, None] * ox[..., None, :], ox, oy


def best_projection_interval(projection):
    """Enumerate every nonempty grid-edge interval using projection Dice.

    The denominator includes predicted interval length and *all* map projection
    mass. This is neither mean mask intensity nor one-sided mask coverage.
    Exact ties use the first row-major (left edge, right edge) pair.
    """
    p = _cpu(projection, torch.float64)
    _require(p.ndim == 1 and bool(torch.isfinite(p).all()) and bool((p >= 0).all()),
             'A finite nonnegative one-dimensional projection is required')
    if not len(p) or float(p.sum()) <= EPS:
        return None
    cumulative = torch.cat((p.new_zeros(1), p.cumsum(0)))
    left = torch.arange(len(p) + 1)[:, None]
    right = torch.arange(len(p) + 1)[None, :]
    length = right - left
    mass = cumulative[None, :] - cumulative[:, None]
    score = 2 * mass / (length + p.sum()).clamp_min(EPS)
    score = torch.where(length > 0, score, score.new_full((), -torch.inf))
    index = int(score.flatten().argmax())
    start, end = divmod(index, len(p) + 1)
    return start / len(p), end / len(p)


def projection_box(mask):
    mask = _cpu(mask, torch.float64)
    _require(mask.ndim == 2, 'A two-dimensional propagated map is required')
    x = best_projection_interval(mask.amax(0))
    y = best_projection_interval(mask.amax(1))
    if x is None or y is None:
        return None
    return mask.new_tensor(((x[0] + x[1]) / 2, (y[0] + y[1]) / 2,
                            x[1] - x[0], y[1] - y[0]))


def box_iou(a, b):
    alo, ahi = a[:2] - a[2:] / 2, a[:2] + a[2:] / 2
    blo, bhi = b[:2] - b[2:] / 2, b[:2] + b[2:] / 2
    intersection = (torch.minimum(ahi, bhi) - torch.maximum(alo, blo)).clamp_min(0).prod()
    union = a[2:].prod() + b[2:].prod() - intersection
    return intersection / union.clamp_min(EPS)


def secondary_peak(mask, valid):
    """Return a distinct strong local maximum, not a GT-selected identity."""
    p = _cpu(mask, torch.float64)
    v = _cpu(valid, torch.bool)
    _require(p.shape == v.shape and p.ndim == 2, 'Peak mask/valid shape mismatch')
    p = p * v
    if not bool(v.any()) or float(p.max()) <= EPS:
        return None
    main = int(p.flatten().argmax())
    main_y, main_x = divmod(main, p.shape[1])
    yy, xx = torch.meshgrid(torch.arange(p.shape[0]), torch.arange(p.shape[1]), indexing='ij')
    distance2 = (yy - main_y).square() + (xx - main_x).square()
    local = p == F.max_pool2d(p[None, None], 3, stride=1, padding=1)[0, 0]
    eligible = (v & local & (distance2 >= FIXED['secondary_peak_min_distance'] ** 2)
                & (p >= FIXED['secondary_peak_ratio'] * p.max()))
    if not bool(eligible.any()):
        return None
    ranked = torch.where(eligible, p, p.new_full((), -torch.inf))
    y, x = divmod(int(ranked.flatten().argmax()), p.shape[1])
    return y, x


def anchor_transport(features, valid, seed, target, source):
    """Original DenseEvidence affinity arithmetic, with no new model evidence."""
    # Preserve FP32 normalization/matmul/top-k/softmax; only weights and mask
    # transport are FP64, exactly as in trajectory_opd.evidence.affinity.
    normalized = features
    target_ids = torch.nonzero(valid[target], as_tuple=False).flatten()
    source_ids = torch.nonzero(valid[source], as_tuple=False).flatten()
    matrix = torch.zeros(features.shape[1], features.shape[1], dtype=torch.float64)
    if len(target_ids) and len(source_ids):
        cosine = normalized[target, target_ids] @ normalized[source, source_ids].T
        count = min(FIXED['affinity_topk'], len(source_ids))
        values, indices = torch.topk(cosine, count, dim=1, sorted=False)
        weights = (values / FIXED['affinity_temperature']).softmax(1).double()
        weights /= weights.sum(1, keepdim=True)
        rows = target_ids[:, None].expand_as(indices)
        columns = source_ids[indices]
        matrix[rows, columns] = weights
    return matrix @ seed.flatten()


class CandidateBank:
    """E, nearest-past/future same-track proposals, and optional secondary peak."""
    def __init__(self, evidence, cache, cfg):
        _require(cache.get('GT_read') is False and evidence.metadata.get('GT_read') is False,
                 'Only label-free cached evidence is permitted')
        options = cfg.get('candidate_bank', cfg)
        for name, value in FIXED.items():
            if name != 'confidence_threshold' and name in options:
                _require(options[name] == value, 'Locked candidate-bank setting changed: ' + name)
        # Public configuration uses this descriptive spelling. Validate it
        # explicitly rather than silently ignoring an equivalent field name.
        for container in (cfg, options):
            if 'secondary_peak_distance_cells' in container:
                _require(container['secondary_peak_distance_cells'] == FIXED['secondary_peak_min_distance'],
                         'Locked candidate-bank setting changed: secondary_peak_distance_cells')
        self.confidence_threshold = float(options.get('confidence_threshold', cfg.get('confidence_threshold', FIXED['confidence_threshold'])))
        _require(0 <= self.confidence_threshold <= 1, 'Confidence threshold must be finite and in [0,1]')
        self.E = _cpu(cache['E'], torch.float64)
        _require(self.E.ndim == 2 and self.E.shape[1] == 4 and bool(torch.isfinite(self.E).all())
                 and bool((self.E[:, 2:] > 0).all()), 'Expected finite positive E[T,4]')
        self.frames = len(self.E)
        self.frame_ids = _cpu(cache['frame_ids'], torch.long)
        _require(torch.equal(self.frame_ids, _cpu(evidence.frame_ids, torch.long)) and
                 len(self.frame_ids) == self.frames and bool((self.frame_ids[1:] > self.frame_ids[:-1]).all()),
                 'Physical frame grid mismatch')
        self.positions = [int(p) for p in cache['planned_positions']]
        _require(self.positions == list(evidence.positions) and len(set(self.positions)) == 8
                 and all(0 <= p < self.frames for p in self.positions), 'Eight observation nodes must match evidence')
        selected = int(cache['teacher']['selected_query'])
        _require(selected == int(evidence.metadata['selected_native_query']), 'Selected teacher identity changed')
        teacher_boxes = _cpu(cache['teacher']['boxes'], torch.float64)
        _require(teacher_boxes.shape == (8, 4) and bool(torch.isfinite(teacher_boxes).all())
                 and bool((teacher_boxes[:, 2:] > 0).all()), 'Native teacher boxes must be normalized cxcywh[8,4]')
        self.height, self.width = int(evidence.height), int(evidence.width)
        cells = self.height * self.width
        visual = _cpu(evidence.features, torch.float32)
        visual_valid = _cpu(evidence.valid, torch.bool)
        _require(visual.shape == (self.frames, cells, 256) and visual_valid.shape == (self.frames, cells),
                 'Frozen physical visual grid dimensions changed')
        _require(bool(torch.isfinite(visual).all()), 'Nonfinite frozen Source features')
        normalized = F.normalize(visual, p=2, dim=-1, eps=1e-12)
        seeds = _cpu(evidence.seeds, torch.float64)
        targets = _cpu(evidence.targets, torch.float64)
        _require(seeds.shape == (8, self.height, self.width) and targets.shape == (self.frames, self.height, self.width),
                 'Selected same-instance mask support dimensions changed')
        _require(bool(torch.isfinite(seeds).all()) and bool((seeds >= 0).all()) and bool((seeds <= 1).all())
                 and bool(torch.isfinite(targets).all()), 'Invalid fixed mask evidence')
        seeds *= visual_valid[self.positions].reshape(8, self.height, self.width)
        self.confidence = _cpu(evidence.confidence, torch.float64)
        event = _cpu(evidence.event, torch.bool)
        self.observed = _cpu(evidence.observed, torch.bool)
        _require(self.confidence.shape == event.shape == self.observed.shape == (self.frames,),
                 'Per-node confidence/event/observation support mismatch')
        _require(bool(torch.isfinite(self.confidence).all()) and bool((self.confidence >= 0).all())
                 and bool((self.confidence <= 1).all()), 'Invalid fixed confidence')
        expected_observed = torch.zeros(self.frames, dtype=torch.bool)
        expected_observed[self.positions] = True
        _require(torch.equal(self.observed, expected_observed), 'Observation positions changed')
        interval = _cpu(cache['interval'], torch.float64)
        _require(interval.shape == (2,) and float(interval[1]) > float(interval[0]), 'Fixed Strong interval required')
        _require(torch.equal(event, (self.frame_ids >= interval[0]) & (self.frame_ids < interval[1])),
                 'Evidence event differs from fixed Strong prior')
        order = sorted(range(8), key=lambda j: self.positions[j])
        self.source_anchors = torch.full((self.frames, 2), -1, dtype=torch.long)
        self.nearest_anchor = torch.empty(self.frames, dtype=torch.long)
        self.propagated_maps = torch.zeros(self.frames, 2, self.height, self.width, dtype=torch.float64)
        self.anchor_features = torch.zeros(self.frames, 256, dtype=torch.float64)
        seed_features = torch.zeros(8, 256, dtype=torch.float64)
        for j, p in enumerate(self.positions):
            mass = seeds[j].flatten()
            pooled = (visual[p].double() * mass[:, None]).sum(0) / mass.sum().clamp_min(EPS)
            seed_features[j] = F.normalize(pooled, dim=0, eps=1e-12)
        with torch.no_grad():
            for t in range(self.frames):
                closest = min(order, key=lambda j: (abs(int(self.frame_ids[t]) - int(self.frame_ids[self.positions[j]])), self.positions[j]))
                self.nearest_anchor[t] = closest
                self.anchor_features[t] = seed_features[closest]
                if self.observed[t]:
                    j = self.positions.index(t)
                    self.source_anchors[t] = j
                    self.propagated_maps[t, 0] = self.propagated_maps[t, 1] = seeds[j]
                else:
                    past = [j for j in order if self.positions[j] < t]
                    future = [j for j in order if self.positions[j] > t]
                    for side, j in enumerate((past[-1] if past else None, future[0] if future else None)):
                        if j is not None:
                            self.source_anchors[t, side] = j
                            self.propagated_maps[t, side] = anchor_transport(
                                normalized, visual_valid, seeds[j], t, self.positions[j]).reshape(self.height, self.width)
            has_evidence = visual_valid.any(-1) & (targets.flatten(1).sum(-1) > EPS)
            has_evidence &= self.propagated_maps.flatten(2).sum((1, 2)) > EPS
            self.active = event & (self.observed | (self.confidence >= self.confidence_threshold)) & has_evidence
            self.boxes = self.E[:, None].expand(-1, 4, -1).clone()
            self.valid = torch.zeros(self.frames, 4, dtype=torch.bool)
            self.valid[:, 0] = True
            self.secondary_peak_cells = torch.full((self.frames, 2), -1, dtype=torch.long)
            for t in range(self.frames):
                if not self.active[t]:
                    continue
                for side in (0, 1):
                    anchor = int(self.source_anchors[t, side])
                    if anchor < 0:
                        continue
                    proposal = teacher_boxes[anchor] if self.observed[t] else projection_box(self.propagated_maps[t, side])
                    if proposal is not None:
                        self.boxes[t, side + 1] = proposal
                        self.valid[t, side + 1] = self._distinct(t, side + 1)
                available_maps = [self.propagated_maps[t, side] for side in (0, 1) if self.source_anchors[t, side] >= 0]
                combined = torch.stack(available_maps).mean(0)
                peak = secondary_peak(combined, visual_valid[t].reshape(self.height, self.width))
                if peak is not None:
                    y, x = peak
                    proposal = teacher_boxes[int(self.nearest_anchor[t])].clone()
                    proposal[:2] = proposal.new_tensor(((x + .5) / self.width, (y + .5) / self.height))
                    self.boxes[t, 3] = proposal
                    self.valid[t, 3] = self._distinct(t, 3)
                    self.secondary_peak_cells[t] = torch.tensor((y, x))
            occupancy, ox, oy = fractional_occupancy(self.boxes, self.height, self.width)
            occupancy *= visual_valid.reshape(self.frames, 1, self.height, self.width)
            px, py = targets.amax(1), targets.amax(2)
            sx = 2 * (ox * px[:, None]).sum(-1) / (ox.sum(-1) + px.sum(-1)[:, None]).clamp_min(EPS)
            sy = 2 * (oy * py[:, None]).sum(-1) / (oy.sum(-1) + py.sum(-1)[:, None]).clamp_min(EPS)
            self.projection = .5 * (sx + sy)
            dz = torch.cat(((self.boxes[..., :2] - self.E[:, None, :2]) / self.E[:, None, 2:],
                            self.boxes[..., 2:].log() - self.E[:, None, 2:].log()), -1)
            displacement = torch.linalg.vector_norm(dz, dim=-1) / 2
            self.guard = torch.relu(displacement - .25).square()
            masked = (occupancy * targets[:, None]).flatten(-2)
            previous = masked[:-1].transpose(0, 1)
            transported = evidence.transport_adjacent(previous).transpose(0, 1)
            current = masked[1:]
            a, b = transported[:, :, None, :], current[:, None, :, :]
            self.link = (2 * (a * b).sum(-1) / (a.sum(-1) + b.sum(-1)).clamp_min(EPS)).clamp(0, 1)
            pooled = torch.bmm(occupancy.flatten(-2), visual.double())
            pooled /= occupancy.flatten(-2).sum(-1).clamp_min(EPS)[..., None]
            self.features = F.normalize(pooled, p=2, dim=-1, eps=1e-12)
        self.candidates = self.boxes
        self.metadata = dict(
            schema='same_instance_four_slot_bank_v1', frames=self.frames, slots=4,
            grid_height=self.height, grid_width=self.width, feature_channels=256,
            slots_rule=['ERef', 'nearest past selected-track anchor', 'nearest future selected-track anchor',
                        'optional strong separated local peak, nearest-anchor size'],
            teacher_boxes='native ReferDINO normalized cxcywh; same selected query at all eight nodes',
            selected_native_query=selected, thresholds={**FIXED, 'confidence_threshold': self.confidence_threshold},
            affinity_arithmetic='original FP32 normalize/cosine/topk/softmax; FP64 normalized weights and map transport',
            affinity_teacher_identity_verified=False,
            projection_proposal='independent global xy projection-Dice maximum over all nonempty grid-edge intervals',
            projection='raw xy projection Dice to original fixed combined evidence targets',
            guard='raw relu(norm(relative center, log-size)/2-.25)^2; no mean/max weighting here',
            link='raw soft Dice, axes edge/previous slot/current slot; existing sparse forward transport',
            ROI_pool='fractional box-cell occupancy times valid grid; normalized weighted frozen Source feature',
            anchor_feature='normalized nearest physical-time selected-mask seed feature; ties earlier anchor',
            inactive_rule='only E is valid; downstream memory must retain its previous value',
            active_nodes=int(self.active.sum()), observed_nodes=int(self.observed.sum()),
            valid_slots=self.valid.sum(0).tolist(),
            optional_peak_nodes=int(self.valid[:, 3].sum()),
            model_calls=0, decoder_calls=0, expert_calls=0, RGB_reads=0,
            outputs_detached=True, downstream_dtype='CPU float64', GT_read=False)
        for value in (self.boxes, self.features, self.anchor_features, self.confidence,
                      self.projection, self.guard, self.link, self.E):
            _require(value.device.type == 'cpu' and value.dtype == torch.float64
                     and not value.requires_grad and bool(torch.isfinite(value).all()),
                     'Candidate bank output must be finite detached CPU FP64')
        _require(bool((self.boxes[..., 2:] > 0).all()) and bool(self.valid[:, 0].all()),
                 'E fallback must always be a valid positive trajectory')
        _require(not bool(self.valid[~self.active, 1:].any()), 'Inactive node has non-E valid support')

    def _distinct(self, t, slot):
        proposal = self.boxes[t, slot]
        if not bool(torch.isfinite(proposal).all()) or not bool((proposal[2:] > 0).all()):
            return False
        return all(float(box_iou(proposal, self.boxes[t, previous])) < FIXED['duplicate_iou']
                   for previous in range(slot) if self.valid[t, previous])

    def private_state(self):
        """Private support only; do not publish masks/features/boxes/row IDs."""
        names = ('boxes', 'valid', 'features', 'anchor_features', 'confidence', 'active',
                 'observed', 'E', 'projection', 'guard', 'link', 'frame_ids',
                 'source_anchors', 'nearest_anchor', 'propagated_maps', 'secondary_peak_cells')
        return dict(metadata=copy.deepcopy(self.metadata), positions8=list(self.positions),
                    **{name: getattr(self, name).detach().clone() for name in names}, GT_read=False)
