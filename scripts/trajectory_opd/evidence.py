"""Fixed instance evidence on the actual cached Source appearance grids.

No model, RGB, GT, learned parameters, or candidate predictions are read here.
The teacher supplies selected-track mask seeds; all affinities are computed in
the frozen Source feature space. Confidence is a cycle/agreement heuristic,
not an estimate of GT correctness. Native masks are explicitly coarsened.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


AFFINITY_TEMPERATURE = .1
AFFINITY_TOPK = 20
EPS = 1e-15


def _require(value, message):
    if not value:
        raise ValueError(message)


def _cpu_tensor(value, dtype=None):
    return torch.as_tensor(value).detach().to(device='cpu', dtype=dtype).clone()


def soft_dice(a, b, dim=-1):
    """Soft-mass Dice 2*sum(a*b)/(sum(a)+sum(b)); empty/empty is zero."""
    return (2 * (a * b).sum(dim) / (a.sum(dim) + b.sum(dim)).clamp_min(EPS)).clamp(0, 1)


def _valid_grid_shape(pos, mask, visual_count):
    """Infer row-major H,W from y/x position halves, ignoring padded cells.

    Every candidate factorization must have distinct row y vectors, distinct
    column x vectors, and constant valid y/x values within each row/column.
    Ambiguous or unsupported layouts fail instead of silently reshaping tokens.
    """
    candidates = []
    for height in range(1, visual_count + 1):
        if visual_count % height:
            continue
        width = visual_count // height
        okay = True; informative = False
        for view_pos, view_mask in zip(pos, mask):
            # A frame with the most valid cells is enough to constrain one view;
            # all remaining frames are checked again after joint inference.
            count = (~view_mask[:, :visual_count]).sum(1)
            frame = int(count.argmax())
            valid = ~view_mask[frame, :visual_count].reshape(height, width)
            if not bool(valid.any()):
                continue
            informative = True
            grid = view_pos[:visual_count, frame].reshape(height, width, 256)
            ys, xs = [], []
            for row in range(height):
                values = grid[row, valid[row], :128]
                if len(values):
                    if not torch.allclose(values, values[:1].expand_as(values), atol=1e-6, rtol=0):
                        okay = False; break
                    ys.append(values[0])
            if not okay:
                break
            for column in range(width):
                values = grid[valid[:, column], column, 128:]
                if len(values):
                    if not torch.allclose(values, values[:1].expand_as(values), atol=1e-6, rtol=0):
                        okay = False; break
                    xs.append(values[0])
            if not okay:
                break
            if (len(torch.unique(torch.stack(ys), dim=0)) != len(ys) or
                    len(torch.unique(torch.stack(xs), dim=0)) != len(xs)):
                okay = False; break
        if okay and informative:
            candidates.append((height, width))
    _require(len(candidates) == 1, f'Cannot uniquely validate visual grid H,W: {candidates}')
    return candidates[0]


def extract_visual_grids(cache):
    """Return physical-node [T,Nvisual,256] features and valid cells on CPU."""
    frame_ids = [int(x) for x in cache['frame_ids']]
    _require(len(frame_ids) >= 2 and all(b > a for a, b in zip(frame_ids, frame_ids[1:])),
             'Physical frame IDs must be strictly increasing')
    views = cache['spatial_inputs']
    _require(len(views) == 2, 'Exactly two native offsets are required')
    features, positions, masks, counts = [], [], [], []
    for offset, item in enumerate(views):
        feature = _cpu_tensor(item['encoded_feature'], torch.float32)
        pos = _cpu_tensor(item['encoded_pos'], torch.float32)
        mask = _cpu_tensor(item['encoded_mask'], torch.bool)
        expected_frames = len(frame_ids[offset::2])
        _require(feature.ndim == 3 and feature.shape[-1] == 256 and feature.shape[1] == expected_frames,
                 'Cached feature dimensions disagree with physical offset')
        _require(pos.shape == feature.shape and mask.shape == (expected_frames, feature.shape[0]),
                 'Position/mask dimensions disagree with cached feature')
        visual_flags = pos.abs().sum(-1).any(-1)
        count = int(visual_flags.sum())
        _require(count > 0 and bool(visual_flags[:count].all()) and not bool(visual_flags[count:].any()),
                 'Spatial visual positions must form one nonzero prefix; text positions must be zero')
        features.append(feature); positions.append(pos); masks.append(mask); counts.append(count)
    _require(counts[0] == counts[1], 'Native offsets have incompatible visual token counts')
    visual_count = counts[0]
    height, width = _valid_grid_shape(positions, masks, visual_count)
    for view_pos, view_mask in zip(positions, masks):
        for frame in range(view_pos.shape[1]):
            valid = ~view_mask[frame, :visual_count].reshape(height, width)
            grid = view_pos[:visual_count, frame].reshape(height, width, 256)
            for row in range(height):
                values = grid[row, valid[row], :128]
                _require(not len(values) or torch.allclose(values, values[:1].expand_as(values), atol=1e-6, rtol=0),
                         'A frame violates the verified row-major y grid')
            for column in range(width):
                values = grid[valid[:, column], column, 128:]
                _require(not len(values) or torch.allclose(values, values[:1].expand_as(values), atol=1e-6, rtol=0),
                         'A frame violates the verified row-major x grid')
    output, valid = [], []
    for node in range(len(frame_ids)):
        view, local = node % 2, node // 2
        v = ~masks[view][local, :visual_count]
        f = features[view][:visual_count, local]
        _require(bool(torch.isfinite(f[v]).all()), 'Nonfinite valid Source features')
        output.append(torch.where(v[:, None], f, torch.zeros_like(f)))
        valid.append(v)
    output, valid = torch.stack(output), torch.stack(valid)
    meta = dict(frames=len(frame_ids), visual_tokens=visual_count, grid_height=height, grid_width=width,
                channels=256, text_tokens_per_offset=[x.shape[0] - visual_count for x in features],
                offset_frames=[x.shape[1] for x in features], physical_interleave='view=t%2, local=t//2',
                valid_visual_cells_min=int(valid.sum(1).min()), valid_visual_cells_max=int(valid.sum(1).max()),
                visual_features='cached post-fusion appearance-position tokens; text/motion token slots excluded',
                pixel_correspondence_training_claim=False)
    return output, valid, frame_ids, (height, width), meta


class TrajectoryEvidence:
    """Immutable evidence shared by OPD and Direct; only boxes receive gradients."""

    def __init__(self, pack, cache, device='cpu'):
        _require(cache.get('GT_read') is False and pack.get('GT_read') is False, 'Only label-free cached evidence is accepted')
        features, valid, frame_ids, shape, grid_meta = extract_visual_grids(cache)
        self.height, self.width = shape
        self.features = features  # CPU private audit support; never used as new trainable parameters.
        self.valid = valid
        self.frame_ids = torch.tensor(frame_ids, dtype=torch.long)
        self.positions = [int(x) for x in cache['planned_positions']]
        _require(len(self.positions) == 8 and len(set(self.positions)) == 8 and
                 all(0 <= x < len(frame_ids) for x in self.positions), 'Eight distinct observation nodes are required')
        _require(list(pack['positions8']) == self.positions and list(pack['frame_ids']) == frame_ids,
                 'Native mask evidence is bound to different physical nodes')
        teacher = pack.get('teacher', pack.get('z', pack.get('native_result')))
        raw = pack.get('raw_out', pack.get('out'))
        _require(teacher is not None and raw is not None, 'Native selected-track mask pack is incomplete')
        selected = int(teacher.get('selected_query', pack.get('selected_query')))
        _require(selected == int(cache['teacher']['selected_query']), 'Native selected object query changed')
        _require(torch.equal(torch.as_tensor(teacher['boxes']), torch.as_tensor(cache['teacher']['boxes'])),
                 'Selected eight teacher boxes differ from cache')
        teacher_shape = teacher.get('input_shape', pack.get('input_shape'))
        _require(teacher_shape is not None and len(teacher_shape) == 4 and teacher_shape[0] == 8 and teacher_shape[1] == 3,
                 'Native teacher image shape must describe the same eight frames')
        _require(pack.get('nested_mask_all_false') is True, 'Teacher padding requires an explicit official crop mapping')
        _require(list(pack['nested_tensor_shape']) == [1, *teacher_shape] and
                 list(pack['nested_mask_shape']) == [1, 8, *teacher_shape[-2:]], 'Unexpected native padded geometry')
        masks = _cpu_tensor(raw['pred_masks'])
        if masks.ndim == 5:
            _require(masks.shape[0] == 1, 'Only one video is supported')
            masks = masks[0]
        _require(masks.ndim == 4 and masks.shape[0] == 8 and 0 <= selected < masks.shape[1], 'Native mask/query shape')
        with torch.no_grad():
            # Official ordering: resize logits to image size, then sigmoid.
            full = F.interpolate(masks[:, selected, None].float(), size=tuple(teacher_shape[-2:]),
                                 mode='bilinear', align_corners=False).sigmoid()
            self.seeds = F.interpolate(full, size=shape, mode='area').squeeze(1).double()
            self.seeds *= valid[self.positions].reshape(8, *shape)
            normalized = F.normalize(features, p=2, dim=-1, eps=1e-12)
            pair_cache = {}

            def affinity(target, source):
                key = (target, source)
                if key not in pair_cache:
                    target_ids = torch.nonzero(valid[target], as_tuple=False).flatten()
                    source_ids = torch.nonzero(valid[source], as_tuple=False).flatten()
                    matrix = torch.zeros(features.shape[1], features.shape[1], dtype=torch.float64)
                    if len(target_ids) and len(source_ids):
                        cosine = normalized[target, target_ids] @ normalized[source, source_ids].T
                        count = min(AFFINITY_TOPK, len(source_ids))
                        values, indices = torch.topk(cosine, count, dim=1, sorted=False)
                        weights = (values / AFFINITY_TEMPERATURE).softmax(1).double()
                        weights /= weights.sum(1, keepdim=True)
                        rows = target_ids[:, None].expand_as(indices)
                        columns = source_ids[indices]
                        matrix[rows, columns] = weights
                    pair_cache[key] = matrix
                return pair_cache[key]

            observed = torch.zeros(len(frame_ids), dtype=torch.bool)
            observed[self.positions] = True
            targets = torch.zeros(len(frame_ids), features.shape[1], dtype=torch.float64)
            confidence = torch.zeros(len(frame_ids), dtype=torch.float64)
            cycle_scores = torch.zeros(len(frame_ids), 2, dtype=torch.float64)
            agreement = torch.ones(len(frame_ids), dtype=torch.float64)
            source_anchors = torch.full((len(frame_ids), 2), -1, dtype=torch.long)
            order = sorted(range(8), key=lambda j: self.positions[j])
            self._neighbors = []
            for node in range(len(frame_ids)):
                if observed[node]:
                    anchor = self.positions.index(node)
                    targets[node] = self.seeds[anchor].flatten()
                    confidence[node] = 1
                    source_anchors[node, 0] = anchor
                    cycle_scores[node, 0] = 1
                    self._neighbors.append((anchor, None))
                    continue
                past = [j for j in order if self.positions[j] < node]
                future = [j for j in order if self.positions[j] > node]
                neighbors = (past[-1] if past else None, future[0] if future else None)
                self._neighbors.append(neighbors)
                propagated, cycles = [], []
                for slot, anchor in enumerate(neighbors):
                    if anchor is None:
                        continue
                    anchor_node = self.positions[anchor]
                    seed = self.seeds[anchor].flatten()
                    transported = affinity(node, anchor_node) @ seed
                    returned = affinity(anchor_node, node) @ transported
                    cycle = soft_dice(returned, seed)
                    propagated.append(transported); cycles.append(cycle)
                    source_anchors[node, slot] = anchor; cycle_scores[node, slot] = cycle
                _require(len(propagated) > 0, 'No observation can support this node')
                targets[node] = torch.stack(propagated).mean(0)
                if len(propagated) == 2:
                    agreement[node] = soft_dice(propagated[0], propagated[1])
                confidence[node] = torch.stack(cycles).mean() * agreement[node]
            targets *= valid
            self.targets = targets.reshape(len(frame_ids), *shape).clamp(0, 1)
            self.confidence = confidence.clamp(0, 1)
            self.observed = observed
            self.cycle_scores = cycle_scores
            self.agreement = agreement
            self.source_anchors = source_anchors
            self.adjacent_forward = torch.stack([affinity(t, t - 1) for t in range(1, len(frame_ids))])
            self.adjacent_reverse = torch.stack([affinity(t - 1, t) for t in range(1, len(frame_ids))])
            interval = cache.get('interval', cache.get('intervals', {}).get('Strong'))
            _require(interval is not None and len(interval) == 2 and float(interval[1]) > float(interval[0]),
                     'A fixed Strong physical interval is required as prior')
            self.event = (self.frame_ids >= float(interval[0])) & (self.frame_ids < float(interval[1]))
            self.event_adjacent = self.event[1:] & self.event[:-1]
            self.W = torch.zeros(len(frame_ids), 8, dtype=torch.float64)
            for node, neighbors in enumerate(self._neighbors):
                if observed[node]:
                    self.W[node, self.positions.index(node)] = 1
                elif self.event[node]:
                    past, future = neighbors
                    if past is None:
                        self.W[node, future] = confidence[node]
                    elif future is None:
                        self.W[node, past] = confidence[node]
                    else:
                        a, b, t = frame_ids[self.positions[past]], frame_ids[self.positions[future]], frame_ids[node]
                        self.W[node, past] = confidence[node] * (b - t) / (b - a)
                        self.W[node, future] = confidence[node] * (t - a) / (b - a)
            self._projection_x = self.targets.amax(1)
            self._projection_y = self.targets.amax(2)
        unobserved = ~observed
        self.metadata = dict(
            **grid_meta, selected_native_query=selected, teacher_input_shape=list(teacher_shape),
            native_mask_shape=list(masks.shape), mask_resize='bilinear logits to teacher image, align_corners=False; sigmoid; area to Source grid',
            native_mask_coarsened=True, source_padding_excluded=True,
            affinity=dict(feature_space='Source post-fusion visual slots only', L2_normalized=True,
                          cosine_temperature=AFFINITY_TEMPERATURE, topk=AFFINITY_TOPK, normalization='rows over valid source cells'),
            propagation='independent nearest past/future anchors, never recursive',
            soft_Dice='2 sum(a*b)/(sum(a)+sum(b)); empty/empty=0',
            confidence_rule='mean return-cycle Dice times past/future agreement; one-sided uses cycle only; observed=1',
            confidence_reliability_claim=False, confidence_normalized_across_nodes=False,
            confidence_unobserved_mean=float(confidence[unobserved].mean()) if bool(unobserved.any()) else 0.,
            confidence_unobserved_min=float(confidence[unobserved].min()) if bool(unobserved.any()) else 0.,
            confidence_unobserved_max=float(confidence[unobserved].max()) if bool(unobserved.any()) else 0.,
            observed_offset_counts=[sum(p % 2 == v for p in self.positions) for v in (0, 1)],
            event_observed_count=int((self.event & observed).sum()), event_unobserved_count=int((self.event & unobserved).sum()),
            event_adjacent_count=int(self.event_adjacent.sum()),
            graph_rule='physical-time linear nearest-anchor interpolation; unobserved multiplied by confidence and fixed Strong prior; observed onehot',
            group_denominators='plain event-node / adjacent-pair counts, never confidence sums',
            reward='U=.5*obs+.5*unobs+.1*link-.1*guard', GT_read=False)
        self.to(device)

    def to(self, device):
        """Move fixed reward tensors; private normalized Source features stay on CPU."""
        for name in ('targets', 'confidence', 'valid', 'frame_ids', 'observed', 'cycle_scores', 'agreement',
                     'source_anchors', 'adjacent_forward', 'adjacent_reverse', 'event', 'event_adjacent',
                     'W', '_projection_x', '_projection_y', 'seeds'):
            setattr(self, name, getattr(self, name).detach().to(device))
        return self

    @staticmethod
    def _on(value, boxes, *, floating=True):
        return value.to(device=boxes.device, dtype=boxes.dtype if floating else value.dtype)

    def box_occupancy(self, boxes):
        """Continuous fractional occupancy of each Source grid cell by each box."""
        x0, y0 = (boxes[:, :2] - boxes[:, 2:] / 2).clamp(0, 1).unbind(-1)
        x1, y1 = (boxes[:, :2] + boxes[:, 2:] / 2).clamp(0, 1).unbind(-1)
        x = torch.arange(self.width + 1, device=boxes.device, dtype=boxes.dtype) / self.width
        y = torch.arange(self.height + 1, device=boxes.device, dtype=boxes.dtype) / self.height
        ox = (torch.minimum(x1[:, None], x[None, 1:]) - torch.maximum(x0[:, None], x[None, :-1])).clamp_min(0) * self.width
        oy = (torch.minimum(y1[:, None], y[None, 1:]) - torch.maximum(y0[:, None], y[None, :-1])).clamp_min(0) * self.height
        return oy[:, :, None] * ox[:, None, :], ox, oy

    def projection_scores(self, boxes):
        """Grid-integral xy projection Dice, in the same [0,1] range as before."""
        _, ox, oy = self.box_occupancy(boxes)
        px, py = self._on(self._projection_x, boxes), self._on(self._projection_y, boxes)
        x = 2 * (ox * px).sum(-1) / (ox.sum(-1) + px.sum(-1)).clamp_min(EPS)
        y = 2 * (oy * py).sum(-1) / (oy.sum(-1) + py.sum(-1)).clamp_min(EPS)
        return .5 * (x + y)

    def utility(self, boxes, E):
        _require(boxes.ndim == 2 and boxes.shape == E.shape == (len(self.frame_ids), 4), 'Full physical trajectory boxes required')
        _require(bool(torch.isfinite(boxes).all()) and bool((boxes[:, 2:] > 0).all()), 'Nonfinite or nonpositive student boxes')
        E = torch.as_tensor(E, device=boxes.device, dtype=boxes.dtype).detach()
        _require(bool(torch.isfinite(E).all()) and bool((E[:, 2:] > 0).all()), 'Invalid fixed E trajectory')
        zero = boxes.sum() * 0
        event, observed = self._on(self.event, boxes, floating=False), self._on(self.observed, boxes, floating=False)
        confidence = self._on(self.confidence, boxes)
        target = self._on(self.targets, boxes)
        projection = self.projection_scores(boxes)
        obs_group, unobs_group = event & observed, event & ~observed
        obs = projection[obs_group].mean() if bool(obs_group.any()) else zero
        unobs = (confidence[unobs_group] * projection[unobs_group]).mean() if bool(unobs_group.any()) else zero
        occupancy, _, _ = self.box_occupancy(boxes)
        masked = (occupancy * target).flatten(1)
        propagated_previous = torch.bmm(self._on(self.adjacent_forward, boxes), masked[:-1, :, None]).squeeze(-1)
        link_scores = soft_dice(masked[1:], propagated_previous)
        adjacent = self._on(self.event_adjacent, boxes, floating=False)
        weighted_links = confidence[1:] * confidence[:-1] * link_scores
        link = weighted_links[adjacent].mean() if bool(adjacent.any()) else zero
        dz = torch.cat(((boxes[:, :2] - E[:, :2]) / E[:, 2:], torch.log(boxes[:, 2:]) - torch.log(E[:, 2:])), -1)
        d = torch.linalg.vector_norm(dz, dim=-1) / 2
        excess = torch.relu(d - .25).square()
        guard_mean, guard_max = excess.mean(), excess.max()
        guard = .5 * guard_mean + .5 * guard_max
        mask = .5 * obs + .5 * unobs
        return dict(U=mask + .1 * link - .1 * guard, mask=mask, link=link, guard=guard,
                    guard_mean=guard_mean, guard_max=guard_max, obs=obs, unobs=unobs,
                    d_mean=d.mean(), d_max=d.max(), projection_scores=projection, link_scores=link_scores,
                    confidence_mean=confidence.mean(),
                    confidence_event_unobserved_mean=confidence[unobs_group].mean() if bool(unobs_group.any()) else zero)

    def private_state(self) -> dict[str, Any]:
        """CPU-only support material; caller must keep raw features/maps private."""
        return dict(metadata=dict(self.metadata), features=self.features.detach().cpu().clone(),
                    **{name: getattr(self, name).detach().cpu().clone() for name in (
                        'valid', 'frame_ids', 'targets', 'confidence', 'seeds', 'cycle_scores', 'agreement',
                        'source_anchors', 'adjacent_forward', 'adjacent_reverse', 'W', 'event', 'observed')},
                    positions8=list(self.positions), GT_read=False)


Evidence = TrajectoryEvidence
