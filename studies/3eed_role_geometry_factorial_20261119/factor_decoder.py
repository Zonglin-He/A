"""Separate the two original feedback pathways while using the same decoder modules."""
import math

from study import CELLS
from decoder import Decoder, gather_slots, torch

SWITCHES = dict(zip(CELLS, ((False, False), (True, False), (False, True), (True, True)), strict=True))


class FactorDecoder(Decoder):
    def __init__(self, cell):
        super().__init__('joint')
        self.role_interaction, self.box_feedback = SWITCHES[cell]

    def forward(self, evidence):
        points = evidence['xyz']
        rgb_uv = evidence['uv'].float()
        projected = evidence['in_image']
        has_ref = evidence['has_reference']
        scene, language, anchors, state, prior = self.initial(evidence)
        keys, values = self.key(scene), self.value(scene)
        left = (points[:, None, None] - anchors[..., None, :3]) / (anchors[..., None, 3:6]*.65+.5)
        static_spatial = -.65*torch.log1p(left.square().sum(-1))
        current_spatial = static_spatial
        intermediate = []
        for step in range(self.turns):
            query = self.query(state + language[:, :, None]).flatten(1, 2)
            match = (query @ keys.transpose(1, 2)) / math.sqrt(query.shape[-1])
            weights = (match.reshape_as(static_spatial) + current_spatial).softmax(-1)
            summary = (weights.flatten(1, 2) @ values).reshape_as(state)
            statistic, centroid, pixels, mass = self.geometry(points, weights, rgb_uv, projected, anchors)
            inputs = torch.cat((summary, statistic), -1).flatten(0, 2)
            state = self.revise(inputs, state.flatten(0, 2)).reshape_as(state)
            boxes, angles = self.box_readout(anchors, state, statistic, centroid)
            unaries = self.slot_score(state).squeeze(-1) + prior
            compatibility = unaries.new_zeros((len(points), self.slots, self.slots))
            if self.role_interaction:
                compatibility, message = self.interaction(state, boxes, angles, pixels, mass, unaries, has_ref)
                if step+1 < self.turns:
                    state = state + self.relation_gate.sigmoid()*message
            if self.box_feedback and step+1 < self.turns:
                dr = (points[:, None, None]-boxes[..., None, :3]) / (boxes[..., None, 3:6]*.65+.5)
                current_spatial = static_spatial - self.spatial_gate.sigmoid()*torch.log1p(dr.square().sum(-1))
            joint = unaries[:, 0, :, None] + unaries[:, 1, None, :] + compatibility
            marginal = torch.stack((joint.logsumexp(-1), joint.logsumexp(-2)), 1)
            marginal = torch.where(has_ref[:, None, None], marginal, unaries)
            intermediate.append((boxes, angles, weights, marginal, joint, unaries))
        boxes, yaw, weights, marginal, joint, unaries = intermediate[-1]
        pair = joint.flatten(1).argmax(-1)
        best = torch.stack((pair // self.slots, pair % self.slots), -1)
        best = torch.where(has_ref[:, None], best, unaries.argmax(-1))
        return dict(boxes=gather_slots(boxes, best), support=gather_slots(weights, best),
                    choices=best, passes=intermediate, proposal_logits=self.proposal_logits(evidence, boxes, marginal))
