"""Causal per-node factors of the existing frozen trajectory reward.

This module does not build evidence or change its utility.  It calls the shared
sparse evidence once per candidate batch, then distributes that same reward
across physical nodes.  Incoming links belong to their later node; the global
maximum guard belongs to the terminal node.  The resulting reward-to-go can be
detached by a score-function caller without discarding later-frame feedback.
Direct callers retain ordinary pathwise gradients through the same tensors.
"""
from __future__ import annotations

import torch

from scripts.trajectory_fast.evidence import TrajectoryEvidence
from scripts.trajectory_opd.evidence import _require


def reward_to_go(factors):
    """Inclusive suffix sums over physical time, without normalization."""
    _require(factors.ndim >= 1, 'A physical-time axis is required')
    return factors.flip(-1).cumsum(-1).flip(-1)


class TrackedFeedback:
    """Factor one already-built immutable sparse evidence object."""

    def __init__(self, evidence: TrajectoryEvidence):
        self.evidence = evidence
        self.frames = len(evidence.frame_ids)
        event = evidence.event.detach().cpu()
        observed = evidence.observed.detach().cpu()
        confidence = evidence.confidence.detach().cpu().double()
        obs_group, unobs_group = event & observed, event & ~observed
        edges = event[1:] & event[:-1]
        self.counts = dict(observed=int(obs_group.sum()), unobserved=int(unobs_group.sum()),
                           adjacent=int(edges.sum()), trajectory=self.frames)
        self._obs_weight = torch.zeros(self.frames, dtype=torch.float64)
        self._unobs_weight = torch.zeros(self.frames, dtype=torch.float64)
        self._edge_weight = torch.zeros(self.frames - 1, dtype=torch.float64)
        if self.counts['observed']:
            self._obs_weight[obs_group] = .5 / self.counts['observed']
        if self.counts['unobserved']:
            self._unobs_weight[unobs_group] = .5 * confidence[unobs_group] / self.counts['unobserved']
        if self.counts['adjacent']:
            product = confidence[1:] * confidence[:-1]
            self._edge_weight[edges] = .1 * product[edges] / self.counts['adjacent']
        self._cache = {}
        self.metadata = dict(
            definition='Exact factorization of frozen sparse trajectory utility',
            factor_time='physical node order, all trajectory nodes retained',
            observation_unary='.5*projection/count_event_observed',
            unobserved_unary='.5*confidence*projection/count_event_unobserved',
            pair_assignment='incoming edge (t-1,t) at physical node t',
            guard_mean_assignment='-.05*relu(d_t-.25)^2/T at every node',
            guard_max_assignment='-.05*max_t relu(d_t-.25)^2 at terminal node only',
            return_to_go='inclusive suffix sum of Psi',
            group_denominators='plain counts; empty groups contribute zero',
            feedback_detach='Caller chooses no_grad/detach; evaluate preserves pathwise gradients',
            propagation_builds=0, evidence_reused=True, counts=dict(self.counts),
            reward_definition_changed=False, GT_read=False,
        )

    def _weights(self, boxes):
        key = (boxes.device, boxes.dtype)
        if key not in self._cache:
            self._cache[key] = tuple(x.to(device=boxes.device, dtype=boxes.dtype)
                                     for x in (self._obs_weight, self._unobs_weight, self._edge_weight))
        return self._cache[key]

    def evaluate(self, boxes, E):
        """Return [B,T] factors and the original [B] utility, with gradients."""
        _require(boxes.ndim == 3 and tuple(boxes.shape[1:]) == (self.frames, 4),
                 'Candidate full physical trajectories [B,T,4] required')
        result = self.evidence.utility_batch(boxes, E)
        obs_weight, unobs_weight, edge_weight = self._weights(boxes)
        obs_unary = result['projection_scores'] * obs_weight
        unobs_unary = result['projection_scores'] * unobs_weight
        unary = obs_unary + unobs_unary
        zero = boxes.sum(-1) * 0
        incoming = result['link_scores'] * edge_weight
        pair = torch.cat((zero[:, :1], incoming), dim=-1)
        # Reuse the evidence's bound E cache.  No graph affinity/transport is
        # reconstructed; this is the same small per-node guard calculation.
        reference = self.evidence._fixed_reference(boxes, E)
        dz = torch.cat(((boxes[..., :2] - reference['center']) / reference['size'],
                        boxes[..., 2:].log() - reference['log_size']), -1)
        d = torch.linalg.vector_norm(dz, dim=-1) / 2
        excess = torch.relu(d - .25).square()
        guard_mean_factors = -.05 * excess / self.frames
        # amax matches scalar Tensor.max() including its gradient over ties.
        terminal_max = -.05 * excess.amax(-1)
        guard_max_factors = torch.cat((zero[:, :-1], terminal_max[:, None]), dim=-1)
        guardfactors = guard_mean_factors + guard_max_factors
        factors = unary + pair + guardfactors
        return dict(Psi=factors, U=result['U'], unary=unary, pair=pair, guardfactors=guardfactors,
                    reward_to_go=reward_to_go(factors), obs_unary=obs_unary, unobs_unary=unobs_unary,
                    guard_mean_factors=guard_mean_factors, guard_max_factors=guard_max_factors,
                    displacement=d, excess=excess,
                    projection_scores=result['projection_scores'], link_scores=result['link_scores'])

    __call__ = evaluate


Feedback = TrackedFeedback
