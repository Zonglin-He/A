"""Sparse, batched evaluation of the frozen trajectory evidence reward.

The dense implementation is used once while constructing fixed evidence, so
its top-k support, seed transport and confidence are unchanged.  No dense
affinity matrix survives construction or is rebuilt during reward evaluation.
Only engineering operations differ: fixed-value caching, sparse gather/reduce,
and one occupancy calculation shared by all reward terms and candidates.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch

from scripts.trajectory_opd.evidence import (
    AFFINITY_TOPK, EPS, TrajectoryEvidence as DenseTrajectoryEvidence,
    _require, soft_dice,
)


def _sparsify(matrix):
    """Keep every original nonzero weight, with zero-filled padding slots."""
    _require(matrix.ndim == 3 and matrix.shape[1] == matrix.shape[2], 'Expected edge-wise square affinities')
    edges, nodes, _ = matrix.shape
    count = min(AFFINITY_TOPK, nodes)
    rows = matrix.reshape(edges * nodes, nodes)
    locations = torch.nonzero(rows, as_tuple=False)
    counts = torch.count_nonzero(rows, dim=1)
    _require(bool((counts <= count).all()), 'Original affinity has more than top-k nonzeros')
    indices = torch.zeros(edges * nodes, count, dtype=torch.long)
    weights = torch.zeros(edges * nodes, count, dtype=matrix.dtype)
    if len(locations):
        offsets = counts.cumsum(0) - counts
        slots = torch.arange(len(locations)) - torch.repeat_interleave(offsets, counts)
        indices[locations[:, 0], slots] = locations[:, 1]
        weights[locations[:, 0], slots] = rows[locations[:, 0], locations[:, 1]]
    return indices.reshape(edges, nodes, count), weights.reshape(edges, nodes, count)


class TrajectoryEvidence:
    """Same fixed reward as the dense class, without persistent dense graphs."""

    _STATIC_NAMES = (
        'valid', 'frame_ids', 'targets', 'confidence', 'seeds', 'cycle_scores',
        'agreement', 'source_anchors', 'W', 'event', 'observed', 'event_adjacent',
        '_projection_x', '_projection_y',
        'adjacent_forward_indices', 'adjacent_forward_weights',
        'adjacent_reverse_indices', 'adjacent_reverse_weights',
        '_obs_indices', '_unobs_indices', '_active_edge_indices', '_edge_numbers',
        '_x_edges', '_y_edges', '_link_confidence',
    )

    def __init__(self, pack, cache, device='cpu'):
        dense = DenseTrajectoryEvidence(pack, cache, device='cpu')
        self.height, self.width = dense.height, dense.width
        self.features = dense.features
        self.positions = list(dense.positions)
        for name in self._STATIC_NAMES:
            if hasattr(dense, name):
                setattr(self, name, getattr(dense, name))
        self.adjacent_forward_indices, self.adjacent_forward_weights = _sparsify(dense.adjacent_forward)
        self.adjacent_reverse_indices, self.adjacent_reverse_weights = _sparsify(dense.adjacent_reverse)
        self.metadata = deepcopy(dense.metadata)
        frames, nodes = len(self.frame_ids), self.height * self.width
        self._obs_indices = torch.nonzero(self.event & self.observed, as_tuple=False).flatten()
        self._unobs_indices = torch.nonzero(self.event & ~self.observed, as_tuple=False).flatten()
        self._active_edge_indices = torch.nonzero(self.event_adjacent, as_tuple=False).flatten()
        self._edge_numbers = torch.arange(frames - 1)[:, None, None]
        self._x_edges = torch.arange(self.width + 1, dtype=torch.float64) / self.width
        self._y_edges = torch.arange(self.height + 1, dtype=torch.float64) / self.height
        self._link_confidence = self.confidence[1:] * self.confidence[:-1]
        graph_names = ('adjacent_forward_indices', 'adjacent_forward_weights',
                       'adjacent_reverse_indices', 'adjacent_reverse_weights')
        sparse_bytes = sum(getattr(self, name).numel() * getattr(self, name).element_size() for name in graph_names)
        dense_bytes = 2 * (frames - 1) * nodes * nodes * dense.adjacent_forward.element_size()
        self.metadata['engineering'] = dict(
            implementation='sparse fixed top-k gather and candidate-batched reward',
            graph_schema='edge_target_topk_indices_weights_v1',
            graph_index_dtype='int64', graph_weight_dtype='float64',
            topk_slots=min(AFFINITY_TOPK, nodes), persistent_dense_affinities=False,
            sparse_graph_bytes=sparse_bytes, equivalent_dense_graph_bytes=dense_bytes,
            graph_storage_ratio=sparse_bytes / dense_bytes if dense_bytes else 0.,
            reward_time_complexity='O(K*T*N*topk), K candidates, N visual cells',
            reward_workspace_complexity='O(K*T*N*topk)',
            fixed_cached=['event group indices', 'edge indices', 'confidence products',
                          'grid boundaries', 'target projections', 'fixed E center/size/log-size'],
            occupancy_computations_per_batch=1,
            reward_definition_changed=False,
        )
        # The original object, including all dense graph tensors, is temporary.
        del dense
        self._runtime_cache = {}
        self._reference = None
        self._reference_input_key = None
        self._reference_runtime = {}
        self.to(device)

    def to(self, device):
        """Move fixed reward support; private Source features stay on CPU."""
        for name in self._STATIC_NAMES:
            setattr(self, name, getattr(self, name).detach().to(device))
        self._runtime_cache = {}
        self._reference_runtime = {}
        return self

    @staticmethod
    def _tensor_key(value):
        if not torch.is_tensor(value):
            return None
        return (value.data_ptr(), value._version, tuple(value.shape), value.dtype, value.device)

    def set_reference(self, E):
        """Bind the fixed E trajectory once; a changed E is rejected explicitly."""
        key = self._tensor_key(E)
        if self._reference is not None and key is not None and key == self._reference_input_key:
            return self
        value = torch.as_tensor(E).detach()
        _require(value.shape == (len(self.frame_ids), 4), 'Fixed E must cover the full physical trajectory')
        _require(bool(torch.isfinite(value).all()) and bool((value[:, 2:] > 0).all()), 'Invalid fixed E trajectory')
        if self._reference is not None:
            _require(torch.equal(value.cpu(), self._reference), 'Evidence is already bound to a different fixed E trajectory')
        else:
            self._reference = value.cpu().clone()
        self._reference_input_key = key
        return self

    def _runtime(self, boxes):
        key = (boxes.device, boxes.dtype)
        if key not in self._runtime_cache:
            def cast(value):
                return value.to(device=boxes.device, dtype=boxes.dtype if value.is_floating_point() else value.dtype)
            names = ('targets', 'confidence', '_projection_x', '_projection_y',
                     '_obs_indices', '_unobs_indices', '_active_edge_indices',
                     '_edge_numbers', '_x_edges', '_y_edges', '_link_confidence',
                     'adjacent_forward_indices', 'adjacent_forward_weights')
            values = {name: cast(getattr(self, name)) for name in names}
            # Group sizes are CPU metadata established once, not GPU synchronizations.
            values['n_obs'] = len(self._obs_indices)
            values['n_unobs'] = len(self._unobs_indices)
            values['n_edges'] = len(self._active_edge_indices)
            values['confidence_mean'] = values['confidence'].mean()
            values['confidence_unobs_mean'] = (values['confidence'][values['_unobs_indices']].mean()
                                                  if values['n_unobs'] else boxes.new_zeros(()))
            values['projection_x_mass'] = values['_projection_x'].sum(-1)
            values['projection_y_mass'] = values['_projection_y'].sum(-1)
            values['_link_confidence'] = values['confidence'][1:] * values['confidence'][:-1]
            self._runtime_cache[key] = values
        return self._runtime_cache[key]

    def _fixed_reference(self, boxes, E):
        self.set_reference(E)
        key = (boxes.device, boxes.dtype)
        if key not in self._reference_runtime:
            value = self._reference.to(device=boxes.device, dtype=boxes.dtype)
            self._reference_runtime[key] = dict(
                center=value[:, :2], size=value[:, 2:],
                log_size=value[:, 2:].log(),
            )
        return self._reference_runtime[key]

    def _occupancy(self, boxes, support):
        x0, y0 = (boxes[..., :2] - boxes[..., 2:] / 2).clamp(0, 1).unbind(-1)
        x1, y1 = (boxes[..., :2] + boxes[..., 2:] / 2).clamp(0, 1).unbind(-1)
        x, y = support['_x_edges'], support['_y_edges']
        ox = (torch.minimum(x1[..., None], x[1:]) - torch.maximum(x0[..., None], x[:-1])).clamp_min(0) * self.width
        oy = (torch.minimum(y1[..., None], y[1:]) - torch.maximum(y0[..., None], y[:-1])).clamp_min(0) * self.height
        return oy[..., :, None] * ox[..., None, :], ox, oy

    def box_occupancy(self, boxes):
        """Preserve the original scalar helper and support candidate batches."""
        return self._occupancy(boxes, self._runtime(boxes))

    @staticmethod
    def _projection(ox, oy, support):
        px, py = support['_projection_x'], support['_projection_y']
        x = 2 * (ox * px).sum(-1) / (ox.sum(-1) + support['projection_x_mass']).clamp_min(EPS)
        y = 2 * (oy * py).sum(-1) / (oy.sum(-1) + support['projection_y_mass']).clamp_min(EPS)
        return .5 * (x + y)

    def projection_scores(self, boxes):
        support = self._runtime(boxes)
        _, ox, oy = self._occupancy(boxes, support)
        return self._projection(ox, oy, support)

    def transport_adjacent(self, values, reverse=False):
        """Apply fixed sparse edge graphs to [...,T-1,N] source masks."""
        expected = (len(self.frame_ids) - 1, self.height * self.width)
        _require(tuple(values.shape[-2:]) == expected, 'Adjacent source masks have incompatible dimensions')
        prefix = 'adjacent_reverse' if reverse else 'adjacent_forward'
        indices = getattr(self, prefix + '_indices').to(device=values.device)
        weights = getattr(self, prefix + '_weights').to(device=values.device, dtype=values.dtype)
        edges = self._edge_numbers.to(device=values.device)
        return (values[..., edges, indices] * weights).sum(-1)

    @staticmethod
    def _transport(values, support):
        sampled = values[:, support['_edge_numbers'], support['adjacent_forward_indices']]
        return (sampled * support['adjacent_forward_weights']).sum(-1)

    def utility_batch(self, boxes, E):
        """Return every original reward field with a leading candidate axis."""
        _require(boxes.ndim == 3 and tuple(boxes.shape[1:]) == (len(self.frame_ids), 4),
                 'Candidate full physical trajectories [K,T,4] required')
        _require(len(boxes) > 0 and bool(torch.isfinite(boxes).all()) and bool((boxes[..., 2:] > 0).all()),
                 'Nonfinite or nonpositive student boxes')
        support = self._runtime(boxes)
        reference = self._fixed_reference(boxes, E)
        zero = boxes.sum((-2, -1)) * 0
        occupancy, ox, oy = self._occupancy(boxes, support)
        projection = self._projection(ox, oy, support)
        obs = projection[:, support['_obs_indices']].mean(-1) if support['n_obs'] else zero
        unobs = ((support['confidence'][support['_unobs_indices']] * projection[:, support['_unobs_indices']]).mean(-1)
                 if support['n_unobs'] else zero)
        masked = (occupancy * support['targets']).flatten(-2)
        propagated_previous = self._transport(masked[:, :-1], support)
        link_scores = soft_dice(masked[:, 1:], propagated_previous)
        weighted_links = support['_link_confidence'] * link_scores
        link = weighted_links[:, support['_active_edge_indices']].mean(-1) if support['n_edges'] else zero
        dz = torch.cat(((boxes[..., :2] - reference['center']) / reference['size'],
                        boxes[..., 2:].log() - reference['log_size']), -1)
        d = torch.linalg.vector_norm(dz, dim=-1) / 2
        excess = torch.relu(d - .25).square()
        # Scalar Tensor.max() splits gradients across exact ties; amax preserves
        # that rule per candidate, whereas max(dim=...) picks one tied index.
        guard_mean, guard_max = excess.mean(-1), excess.amax(-1)
        guard = .5 * guard_mean + .5 * guard_max
        mask = .5 * obs + .5 * unobs
        return dict(U=mask + .1 * link - .1 * guard, mask=mask, link=link, guard=guard,
                    guard_mean=guard_mean, guard_max=guard_max, obs=obs, unobs=unobs,
                    d_mean=d.mean(-1), d_max=d.amax(-1),
                    projection_scores=projection, link_scores=link_scores,
                    confidence_mean=support['confidence_mean'].expand(len(boxes)),
                    confidence_event_unobserved_mean=support['confidence_unobs_mean'].expand(len(boxes)))

    def utility(self, boxes, E):
        _require(boxes.ndim == 2 and tuple(boxes.shape) == (len(self.frame_ids), 4),
                 'Full physical trajectory boxes required')
        return {name: value[0] for name, value in self.utility_batch(boxes[None], E).items()}

    def private_state(self) -> dict[str, Any]:
        """Sparse schema; no dense graph or raw support should be published."""
        names = ('valid', 'frame_ids', 'targets', 'confidence', 'seeds', 'cycle_scores', 'agreement',
                 'source_anchors', 'adjacent_forward_indices', 'adjacent_forward_weights',
                 'adjacent_reverse_indices', 'adjacent_reverse_weights', 'W', 'event', 'observed')
        return dict(schema='trajectory_sparse_evidence_v1', metadata=deepcopy(self.metadata),
                    features=self.features.detach().cpu().clone(),
                    **{name: getattr(self, name).detach().cpu().clone() for name in names},
                    positions8=list(self.positions), GT_read=False)


Evidence = TrajectoryEvidence
