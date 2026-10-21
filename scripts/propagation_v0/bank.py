"""Label-free CPU support and additive fixed score for propagation v0.

No file, model, RGB or label reads. The Source grid and the selected ReferDINO
track are supplied by an already verified immutable evidence cache. Affinity
selection retains original FP32 arithmetic; all subsequent support is FP64.
"""
from __future__ import annotations

import copy
from collections import deque

import torch
import torch.nn.functional as F

EPS = 1e-15


def require(condition, message):
    if not condition: raise ValueError(message)


def cpu(value, dtype=None):
    if torch.is_tensor(value): require(value.device.type == 'cpu', 'CPU evidence required')
    return torch.as_tensor(value, dtype=dtype).detach().clone()


def occupancy(boxes, height, width):
    x0, y0 = (boxes[..., :2] - boxes[..., 2:] / 2).clamp(0, 1).unbind(-1)
    x1, y1 = (boxes[..., :2] + boxes[..., 2:] / 2).clamp(0, 1).unbind(-1)
    x = torch.arange(width + 1, dtype=torch.float64) / width
    y = torch.arange(height + 1, dtype=torch.float64) / height
    ox = (torch.minimum(x1[..., None], x[1:]) - torch.maximum(x0[..., None], x[:-1])).clamp_min(0) * width
    oy = (torch.minimum(y1[..., None], y[1:]) - torch.maximum(y0[..., None], y[:-1])).clamp_min(0) * height
    return oy[..., :, None] * ox[..., None, :], ox, oy


def iou(a, b):
    if torch.equal(a, b): return a.new_ones(())
    lo = torch.maximum(a[:2] - a[2:] / 2, b[:2] - b[2:] / 2)
    hi = torch.minimum(a[:2] + a[2:] / 2, b[:2] + b[2:] / 2)
    intersection = (hi - lo).clamp_min(0).prod()
    return intersection / (a[2:].prod() + b[2:].prod() - intersection).clamp_min(EPS)


def soft_dice(a, b):
    return (2 * (a * b).sum(-1) / (a.sum(-1) + b.sum(-1)).clamp_min(EPS)).clamp(0, 1)


def transport(normalized, valid, values, target, source):
    """Original DenseEvidence affinity, including original operation order."""
    target_ids = torch.nonzero(valid[target], as_tuple=False).flatten()
    source_ids = torch.nonzero(valid[source], as_tuple=False).flatten()
    matrix = torch.zeros(normalized.shape[1], normalized.shape[1], dtype=torch.float64)
    if len(target_ids) and len(source_ids):
        cosine = normalized[target, target_ids] @ normalized[source, source_ids].T
        count = min(20, len(source_ids))
        scores, indices = torch.topk(cosine, count, dim=1, sorted=False)
        weights = (scores / .1).softmax(1).double()
        weights /= weights.sum(1, keepdim=True)
        matrix[target_ids[:, None].expand_as(indices), source_ids[indices]] = weights
    return matrix @ values.flatten()


def component_boxes(mask, valid, relative_threshold=.5):
    """All strict relative-threshold 4-connected components and their map mass."""
    p, v = cpu(mask, torch.float64), cpu(valid, torch.bool)
    require(p.ndim == 2 and p.shape == v.shape and bool(torch.isfinite(p).all())
            and bool((p >= 0).all()), 'Invalid component support map')
    require(relative_threshold == .5, 'Locked relative component threshold is .5')
    p = p * v
    if float(p.max()) <= EPS: return []
    support = (p > relative_threshold * p.max()) & v
    seen = torch.zeros_like(support)
    height, width = p.shape
    out = []
    for y in range(height):
        for x in range(width):
            if not support[y, x] or seen[y, x]: continue
            pending = deque([(y, x)]); seen[y, x] = True; cells = []
            while pending:
                yy, xx = pending.popleft(); cells.append((yy, xx))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = yy + dy, xx + dx
                    if 0 <= ny < height and 0 <= nx < width and support[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True; pending.append((ny, nx))
            ys, xs = zip(*cells)
            x0, x1, y0, y1 = min(xs) / width, (max(xs) + 1) / width, min(ys) / height, (max(ys) + 1) / height
            box = p.new_tensor(((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0))
            mass = sum(float(p[yy, xx]) for yy, xx in cells)
            out.append(dict(box=box, mass=mass, cells=len(cells)))
    return out


class CandidateBank:
    def __init__(self, evidence, cache, cfg):
        require(cache.get('GT_read') is False and evidence.metadata.get('GT_read') is False, 'Label-free evidence required')
        options = cfg.get('candidate_bank', cfg)
        aliases = dict(component_relative_threshold=.5, relative_threshold=.5, component_threshold=.5,
                       nms_iou=.7, duplicate_iou=.7, component_nms_iou=.7,
                       confidence_threshold=.1, affinity_temperature=.1, affinity_topk=20,
                       max_candidates=3, original_prior=.7, component_connectivity=4)
        for name, expected in aliases.items():
            for source in (cfg, options):
                if name in source: require(source[name] == expected, 'Locked bank setting changed: ' + name)
        self.E = cpu(cache['E'], torch.float64); T = len(self.E)
        require(self.E.shape == (T, 4) and bool(torch.isfinite(self.E).all()) and bool((self.E[:, 2:] > 0).all()), 'Invalid E trajectory')
        self.frames = T; self.frame_ids = cpu(cache['frame_ids'], torch.long)
        require(torch.equal(self.frame_ids, cpu(evidence.frame_ids, torch.long)) and self.frame_ids.shape == (T,)
                and bool((self.frame_ids[1:] > self.frame_ids[:-1]).all()), 'Physical grid changed')
        self.positions = [int(p) for p in cache['planned_positions']]; self.positions8 = self.positions
        require(self.positions == list(evidence.positions) and len(set(self.positions)) == 8
                and all(0 <= p < T for p in self.positions), 'Observation geometry changed')
        selected = int(cache['teacher']['selected_query'])
        require(selected == int(evidence.metadata['selected_native_query']), 'Selected native identity changed')
        teacher = cpu(cache['teacher']['boxes'], torch.float64)
        require(teacher.shape == (8, 4) and bool(torch.isfinite(teacher).all()) and bool((teacher[:, 2:] > 0).all()), 'Native normalized cxcywh boxes required')
        self.height, self.width = int(evidence.height), int(evidence.width); N = self.height * self.width
        visual = cpu(evidence.features, torch.float32); valid_cells = cpu(evidence.valid, torch.bool)
        require(visual.shape == (T, N, 256) and valid_cells.shape == (T, N) and bool(torch.isfinite(visual).all()), 'Invalid frozen Source grid')
        normalized = F.normalize(visual, dim=-1, eps=1e-12)
        seeds = cpu(evidence.seeds, torch.float64)
        require(seeds.shape == (8, self.height, self.width) and bool(torch.isfinite(seeds).all())
                and bool(((seeds >= 0) & (seeds <= 1)).all()), 'Invalid selected-mask seeds')
        seeds *= valid_cells[self.positions].reshape_as(seeds)
        self.observed = torch.zeros(T, dtype=torch.bool); self.observed[self.positions] = True
        require(torch.equal(self.observed, cpu(evidence.observed, torch.bool)), 'Observed support changed')
        span = cpu(cache['interval'], torch.float64)
        require(span.shape == (2,) and span[1] > span[0], 'Fixed Strong interval required')
        self.event = (self.frame_ids >= span[0]) & (self.frame_ids < span[1])
        require(torch.equal(self.event, cpu(evidence.event, torch.bool)), 'Strong prior changed')
        self.source_anchors = torch.full((T, 2), -1, dtype=torch.long)
        self.anchor_distance = torch.full((T, 2), -1., dtype=torch.float64)
        self.nearest_anchor = torch.empty(T, dtype=torch.long)
        self.side_maps = torch.zeros(T, 2, self.height, self.width, dtype=torch.float64)
        self.side_reliability = torch.zeros(T, 2, dtype=torch.float64)
        self.side_available = torch.zeros(T, 2, dtype=torch.bool)
        self.anchor_features = torch.zeros(T, 256, dtype=torch.float64)
        seed_features = torch.zeros(8, 256, dtype=torch.float64)
        for j, node in enumerate(self.positions):
            mass = seeds[j].flatten()
            pooled = (visual[node].double() * mass[:, None]).sum(0) / mass.sum().clamp_min(EPS)
            seed_features[j] = F.normalize(pooled, dim=0, eps=1e-12)
        self.seed_features = seed_features
        self.seed_valid = (seeds.flatten(1).sum(-1) > EPS) & (seed_features.norm(dim=-1) > EPS)
        order = sorted(range(8), key=lambda j: self.positions[j])
        valid_order = [j for j in order if self.seed_valid[j]]
        initial_anchor = min(valid_order, key=lambda j: (abs(int(self.frame_ids[0]) - int(self.frame_ids[self.positions[j]])), self.positions[j])) if valid_order else None
        self.initial_memory = seed_features[initial_anchor].clone() if initial_anchor is not None else torch.zeros(256, dtype=torch.float64)
        with torch.no_grad():
            for t in range(T):
                nearest = min(order, key=lambda j: (abs(int(self.frame_ids[t]) - int(self.frame_ids[self.positions[j]])), self.positions[j]))
                self.nearest_anchor[t] = nearest; self.anchor_features[t] = seed_features[nearest]
                if self.observed[t]:
                    j = self.positions.index(t); neighbors = (j, j)
                else:
                    past = [j for j in order if self.positions[j] < t]
                    future = [j for j in order if self.positions[j] > t]
                    neighbors = (past[-1] if past else None, future[0] if future else None)
                for side, j in enumerate(neighbors):
                    if j is None: continue
                    a = self.positions[j]; self.source_anchors[t, side] = j; self.side_available[t, side] = True
                    self.anchor_distance[t, side] = abs(int(self.frame_ids[t]) - int(self.frame_ids[a]))
                    if a == t:
                        target, reliability = seeds[j].flatten(), 1.
                    else:
                        target = transport(normalized, valid_cells, seeds[j], t, a)
                        returned = transport(normalized, valid_cells, target, a, t)
                        reliability = float(soft_dice(returned, seeds[j].flatten()))
                    self.side_maps[t, side] = target.reshape(self.height, self.width)
                    self.side_reliability[t, side] = reliability
            rsum = self.side_reliability.sum(-1)
            self.merged_targets = (self.side_maps * self.side_reliability[..., None, None]).sum(1) / rsum.clamp_min(EPS)[:, None, None]
            self.merged_targets *= valid_cells.reshape(T, self.height, self.width)
            self.confidence = rsum / self.side_available.sum(-1).clamp_min(1)
            self.active = (self.event & (self.confidence >= .1) & valid_cells.any(-1)
                           & (self.merged_targets.flatten(1).sum(-1) > EPS))
            self.boxes = self.E[:, None].expand(-1, 3, -1).clone()
            self.valid = torch.zeros(T, 3, dtype=torch.bool); self.valid[:, 0] = True
            self.candidate_quality = torch.zeros(T, 3, dtype=torch.float64)
            self.candidate_origin = torch.full((T, 3), -1, dtype=torch.long)
            for t in range(T):
                if not self.active[t]: continue
                proposals = []
                if self.observed[t]:
                    j = self.positions.index(t)
                    proposals.append((1, float(seeds[j].sum()), -2, teacher[j]))
                for side in (0, 1):
                    if not self.side_available[t, side]: continue
                    for comp in component_boxes(self.side_maps[t, side], valid_cells[t].reshape(self.height, self.width)):
                        quality = float(self.side_reliability[t, side]) * comp['mass']
                        proposals.append((0, quality, side, comp['box']))
                proposals.sort(key=lambda p: (-p[0], -p[1], p[2], tuple(p[3].tolist())))
                slot = 1
                for _, quality, origin, box in proposals:
                    if any(float(iou(box, self.boxes[t, k])) >= .7 for k in range(slot)): continue
                    self.boxes[t, slot] = box; self.valid[t, slot] = True
                    self.candidate_quality[t, slot] = quality; self.candidate_origin[t, slot] = origin
                    slot += 1
                    if slot == 3: break
            occ, ox, oy = occupancy(self.boxes, self.height, self.width)
            occ *= valid_cells.reshape(T, 1, self.height, self.width)
            px, py = self.merged_targets.amax(1), self.merged_targets.amax(2)
            sx = 2 * (ox * px[:, None]).sum(-1) / (ox.sum(-1) + px.sum(-1)[:, None]).clamp_min(EPS)
            sy = 2 * (oy * py[:, None]).sum(-1) / (oy.sum(-1) + py.sum(-1)[:, None]).clamp_min(EPS)
            self.projection = .5 * (sx + sy); self.u = self.projection
            dz = torch.cat(((self.boxes[..., :2] - self.E[:, None, :2]) / self.E[:, None, 2:],
                            self.boxes[..., 2:].log() - self.E[:, None, 2:].log()), -1)
            d = torch.linalg.vector_norm(dz, dim=-1) / 2
            self.guard = -torch.expm1(-d.square())
            masked = (occ * self.merged_targets[:, None]).flatten(-2)
            previous = evidence.transport_adjacent(masked[:-1].transpose(0, 1)).transpose(0, 1)
            self.link = soft_dice(previous[:, :, None], masked[1:, None])
            pooled = torch.bmm(occ.flatten(-2), visual.double()) / occ.flatten(-2).sum(-1).clamp_min(EPS)[..., None]
            self.features = F.normalize(pooled, dim=-1, eps=1e-12)
            self.E_features = self.features[:, 0]
            self.pi0 = torch.zeros(T, 3, dtype=torch.float64)
            alternatives = self.valid[:, 1:].sum(-1)
            self.pi0[:, 0] = torch.where(alternatives > 0, torch.full((T,), .7, dtype=torch.float64), torch.ones(T, dtype=torch.float64))
            self.pi0[:, 1:] = self.valid[:, 1:] * (.3 / alternatives.clamp_min(1).double())[:, None]
            self.logprior = self.pi0.log()
            self._build_score()
        self.candidates = self.boxes
        self.metadata = dict(schema='propagation_v0_three_slot_bank', frames=T, slots=3,
            grid_height=self.height, grid_width=self.width,
            active_nodes=int(self.active.sum()), valid_slots=self.valid.sum(0).tolist(),
            thresholds=dict(relative_component=.5, confidence=.1, nms_iou=.7, affinity_topk=20, affinity_temperature=.1),
            side_confidence='mean available per-anchor return-cycle Dice; no F/B agreement multiplier',
            merged_support='sum(r_side*P_side)/sum(r_side); observed native seed and c=1',
            components='strict >.5 own-map peak, four-neighbor connectivity, grid-edge bbox',
            candidate_quality='r_side times sum of component map values; native observed box has priority',
            candidate_origin_codes={'-2':'native observed teacher','-1':'original E','0':'past map component','1':'future map component'},
            prior='E=.7 and valid alternatives share .3; without alternatives E=1',
            anchor_features='nearest physical-time same-instance seed features, initialization only',
            initial_memory='nearest nonempty valid selected-mask seed to first physical node, or zero if none',
            initial_memory_available=initial_anchor is not None,
            guard='bounded 1-exp(-d^2), d=norm(relative center/log-size)/2',
            score='sum(score_unary)+sum(score_pair); plain event group counts, no confidence-sum normalization',
            link_axes='edge,previous_slot,current_slot', inactive_rule='only E valid and no memory write',
            confidence_calibrated=False, selected_identity_verified=False,
            model_calls=0, decoder_calls=0, expert_calls=0, RGB_reads=0, GT_read=False)
        for x in (self.boxes, self.features, self.anchor_features, self.confidence, self.projection, self.guard,
                  self.link, self.pi0, self.score_unary, self.score_pair):
            require(x.device.type == 'cpu' and x.dtype == torch.float64 and not x.requires_grad
                    and bool(torch.isfinite(x).all()), 'Finite detached CPU FP64 bank required')
        require(bool(self.valid[:, 0].all()) and not bool(self.valid[~self.active, 1:].any()), 'Invalid fallback support')

    def _build_score(self):
        observed = self.event & self.observed; unobserved = self.event & ~self.observed
        self.event_edges = self.event[:-1] & self.event[1:]
        obs_w = .5 * observed.double() / max(1, int(observed.sum()))
        unobs_w = .5 * self.confidence * unobserved / max(1, int(unobserved.sum()))
        self.score_unary = (obs_w + unobs_w)[:, None] * self.projection - .1 * self.guard / self.frames
        edge_w = .25 * self.event_edges * self.confidence[:-1] * self.confidence[1:] / max(1, int(self.event_edges.sum()))
        self.score_pair = edge_w[:, None, None] * self.link
        self.score_counts = dict(observed=int(observed.sum()), unobserved=int(unobserved.sum()),
                                 event_edges=int(self.event_edges.sum()), all_nodes=self.frames)

    def score_path(self, actions):
        require(not torch.is_tensor(actions) or actions.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8),
                'Path action indices must be integers')
        actions = cpu(actions, torch.long)
        require(actions.shape == (self.frames,) and bool(((actions >= 0) & (actions < 3)).all()), 'One valid action per node required')
        nodes = torch.arange(self.frames)
        require(bool(self.valid[nodes, actions].all()), 'Path selected an invalid candidate')
        return (self.score_unary[nodes, actions].sum() +
                self.score_pair[torch.arange(self.frames - 1), actions[:-1], actions[1:]].sum())

    @classmethod
    def from_state(cls, state):
        """Rebuild only tensor containers from a sealed private bank snapshot.

        This does not execute feature transport, candidate construction or
        feedback evaluation. The runner must verify snapshot/seal/source hashes
        before calling it; this method additionally checks CPU shapes/support.
        """
        require(state.get('GT_read') is False, 'Label-free private bank required')
        metadata = copy.deepcopy(state['metadata'])
        require(metadata.get('schema') == 'propagation_v0_three_slot_bank' and metadata.get('GT_read') is False,
                'Private bank schema/GT boundary changed')
        obj = cls.__new__(cls)
        obj.metadata = metadata
        obj.frames = T = int(metadata['frames']); obj.height = H = int(metadata['grid_height']); obj.width = W = int(metadata['grid_width'])
        require(T >= 8 and H > 0 and W > 0 and metadata['slots'] == 3, 'Invalid saved bank dimensions')
        obj.positions = list(state['positions8']); obj.positions8 = obj.positions
        require(len(set(obj.positions)) == 8 and all(isinstance(p, int) and 0 <= p < T for p in obj.positions), 'Invalid saved observation positions')
        obj.score_counts = copy.deepcopy(state['score_counts'])
        shapes = dict(boxes=(T,3,4), valid=(T,3), features=(T,3,256), anchor_features=(T,256),
            E_features=(T,256), E=(T,4), confidence=(T,), active=(T,), observed=(T,), frame_ids=(T,),
            event=(T,), side_maps=(T,2,H,W), side_reliability=(T,2), side_available=(T,2),
            source_anchors=(T,2), anchor_distance=(T,2), nearest_anchor=(T,), merged_targets=(T,H,W),
            candidate_quality=(T,3), candidate_origin=(T,3), projection=(T,3), guard=(T,3),
            link=(T-1,3,3), pi0=(T,3), logprior=(T,3), initial_memory=(256,),
            seed_features=(8,256), seed_valid=(8,), score_unary=(T,3), score_pair=(T-1,3,3))
        booleans = {'valid','active','observed','event','side_available','seed_valid'}
        integers = {'frame_ids','source_anchors','nearest_anchor','candidate_origin'}
        for name, shape in shapes.items():
            dtype = torch.bool if name in booleans else torch.long if name in integers else torch.float64
            value = cpu(state[name], dtype)
            require(value.shape == shape, 'Saved bank tensor shape changed: ' + name)
            if dtype == torch.float64 and name != 'logprior': require(bool(torch.isfinite(value).all()), 'Nonfinite saved bank: ' + name)
            setattr(obj, name, value)
        require(bool((obj.frame_ids[1:] > obj.frame_ids[:-1]).all()) and bool((obj.boxes[...,2:] > 0).all()), 'Invalid saved physical grid/box sizes')
        require(torch.equal(obj.boxes[:,0], obj.E) and bool(obj.valid[:,0].all()) and not bool(obj.valid[~obj.active,1:].any()), 'Saved fallback changed')
        require(bool(((obj.confidence >= 0) & (obj.confidence <= 1)).all()) and bool(((obj.guard >= 0) & (obj.guard <= 1)).all()), 'Saved bounded support changed')
        require(bool((obj.pi0[~obj.valid] == 0).all()) and bool((obj.pi0[obj.valid] > 0).all())
                and torch.allclose(obj.pi0.sum(-1), torch.ones(T,dtype=torch.float64),atol=1e-12,rtol=0), 'Saved prior changed')
        require(torch.equal(obj.logprior, obj.pi0.log()), 'Saved log prior changed')
        require(torch.equal(obj.observed, torch.isin(torch.arange(T), torch.tensor(obj.positions))), 'Saved observed support changed')
        obj.event_edges = obj.event[:-1] & obj.event[1:]
        obj.candidates = obj.boxes; obj.u = obj.projection
        return obj

    def private_state(self):
        names = ('boxes', 'valid', 'features', 'anchor_features', 'E_features', 'E', 'confidence', 'active',
                 'observed', 'frame_ids', 'event', 'side_maps', 'side_reliability', 'side_available',
                 'source_anchors', 'anchor_distance', 'nearest_anchor', 'merged_targets', 'candidate_quality',
                 'candidate_origin', 'projection', 'guard', 'link', 'pi0', 'score_unary', 'score_pair')
        names += ('logprior', 'initial_memory', 'seed_features', 'seed_valid')
        return dict(metadata=copy.deepcopy(self.metadata), positions8=list(self.positions), score_counts=dict(self.score_counts),
                    **{name: getattr(self, name).detach().clone() for name in names}, GT_read=False)
