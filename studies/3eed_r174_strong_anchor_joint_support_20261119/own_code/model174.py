"""Source-trained candidate rescoring on verified R158 anchors."""
import math

import torch
import torch.nn.functional as F

from scope174 import CELLS, read, SPEC
from decoder import Decoder, box_distance, gather_slots, in_boxes, normalized_coordinate, pool


class StrongAnchorDecoder(Decoder):
    def __init__(self, cell, width=96):
        super().__init__('without_rgb', width=width)
        if cell not in CELLS:
            raise ValueError(cell)
        self.cross, self.self_geometry = read(SPEC)['design']['cells'][cell]
        del self.image, self.candidate_proj

    def initial(self, evidence):
        points = evidence['xyz']
        uv = evidence['uv'].float()
        visible = evidence['visible']
        pose = torch.cat((normalized_coordinate(points), (2*uv-1)*visible[..., None],
                          visible[..., None]), -1)
        scene = self.scene(self.point(evidence['points'].float()) + self.position(pose))
        tokens = self.token(evidence['text'].float())
        phrase = pool(tokens, evidence['valid'][:, None])
        roles = pool(tokens, evidence['role_mask'])
        language = self.language(torch.cat((roles[:, :2], phrase.expand(-1, 2, -1),
                                             roles[:, 2:3].expand(-1, 2, -1)), -1))
        anchors = torch.stack((evidence['subject_boxes'], evidence['reference_boxes']), 1)
        ref_prior = evidence['ref_logit']
        ref_prior = .22*(ref_prior-ref_prior.amax(-1, keepdim=True)).clamp_min(-15)
        prior = torch.stack((evidence['base_logit'], ref_prior), 1)
        return scene, language, anchors, prior

    @staticmethod
    def spatial_bias(points, anchors):
        delta = points[:, None, None] - anchors[..., None, :3]
        c, s = anchors[..., 6:].cos(), anchors[..., 6:].sin()
        offset = torch.stack((c*delta[..., 0] + s*delta[..., 1],
                              -s*delta[..., 0] + c*delta[..., 1], delta[..., 2]), -1)
        return -torch.log1p((offset / (anchors[..., None, 3:6]*.65+.5)).square().sum(-1))

    def forward(self, evidence, reference_override=None):
        points, uv = evidence['xyz'], evidence['uv'].float()
        scene, language, anchors, prior = self.initial(evidence)
        active = evidence['has_reference']
        reference_count = self.slots
        if reference_override is not None:
            # Diagnostic only, with all reference slots set to the same intervention.
            placement, weights, descriptor_weights = reference_override
            reference_count = 1
            anchors = torch.stack((anchors[:, 0], placement[:, None].expand(-1, self.slots, -1)), 1)
            prior = torch.stack((prior[:, 0], torch.zeros_like(prior[:, 1])), 1)
        center, size = anchors[..., :3], anchors[..., 3:6]
        anchor_pose = torch.cat((normalized_coordinate(center), size.log()), -1)
        state = language[:, :, None] + self.anchor(anchor_pose) + self.role_slot[None]
        keys, values = self.key(scene), self.value(scene)
        static_spatial = .65*self.spatial_bias(points, anchors)
        current_spatial = static_spatial
        base_yaw = anchors[..., 6]
        base_cos, base_sin = (2*base_yaw).cos(), (2*base_yaw).sin()
        passes = []
        for step in range(self.turns):
            query = self.query(state + language[:, :, None]).flatten(1, 2)
            match = (query @ keys.transpose(1, 2)) / math.sqrt(query.shape[-1])
            weight = (match.reshape_as(static_spatial).float() + current_spatial).softmax(-1)
            descriptor = weight
            if reference_override is not None:
                weight = torch.stack((weight[:, 0], weights[:, None].expand(-1, self.slots, -1)), 1)
                descriptor = torch.stack((weight[:, 0], descriptor_weights[:, None].expand_as(weight[:, 1])), 1)
            summary = (descriptor.flatten(1, 2) @ values).reshape_as(state)
            statistic, centroid, pixels, mass = self.geometry(points, weight, uv, evidence['visible'], anchors[..., :6])
            inputs = torch.cat((summary, statistic), -1).flatten(0, 2)
            state = self.revise(inputs, state.flatten(0, 2)).reshape_as(state)
            box, angle = self.box_readout(anchors[..., :6], state, statistic, centroid)
            local_cos, local_sin = angle.float().unbind(-1)
            angle = torch.stack((base_cos*local_cos-base_sin*local_sin,
                                 base_sin*local_cos+base_cos*local_sin), -1)
            box = torch.cat((box[..., :6], (base_yaw+box[..., 6])[..., None]), -1)
            if reference_override is not None:
                # Same observed reference location in every recurrent relationship read.
                box = torch.stack((box[:, 0], anchors[:, 1]), 1)
                true_angle = torch.stack((base_cos[:, 1], base_sin[:, 1]), -1)
                angle = torch.stack((angle[:, 0], true_angle), 1)
            unaries = prior + .75*self.slot_score(state).float().squeeze(-1).tanh()
            compatibility = unaries.new_zeros((len(points), self.slots, self.slots))
            if self.cross:
                compatibility, message = self.interaction(state, box, angle, pixels, mass, unaries, active)
                compatibility = .5*compatibility.float().tanh()
                if step+1 < self.turns:
                    state = state + self.relation_gate.sigmoid()*message
            if self.self_geometry and step+1 < self.turns:
                current_spatial = static_spatial + self.spatial_gate.sigmoid()*self.spatial_bias(points, box)
            ref = unaries[:, 1].log_softmax(-1)
            joint = unaries[:, 0, :, None] + ref[:, None, :] + compatibility
            # Subtracting the reference normalization makes the subject utility exactly
            # independent of reference scores in the two no-cross conditions.
            subject = joint.logsumexp(-1) if self.cross else unaries[:, 0]
            subject = torch.where(active[:, None], subject, unaries[:, 0])
            reference = joint.logsumexp(-2) if self.cross else unaries[:, 1]
            logits = torch.stack((subject, reference), 1)
            passes.append((box, angle, weight, logits, joint, unaries))
        slot = subject.argmax(-1)
        conditional = joint[torch.arange(len(points), device=slot.device), slot]
        reference_slot = conditional[:, :reference_count].argmax(-1)
        choices = torch.stack((slot, reference_slot), 1)
        return dict(choices=choices, boxes=gather_slots(anchors, choices), support=gather_slots(weight, choices),
                    passes=passes, subject_logits=subject)


def objective(part, output):
    labels = part['gt']
    quality = part['quality']
    known = torch.stack((torch.ones_like(part['known_ref']), part['known_ref']), 1)
    covered = known & (quality.amax(-1) > 0.)
    anchors = torch.stack((part['subject_boxes'], part['reference_boxes']), 1)
    closest = box_distance(anchors, labels).argmin(-1)
    slot = torch.where(covered, quality.argmax(-1), closest)
    valid_pair = covered.all(-1)
    rank_target = (quality/.12).softmax(-1)
    pair_target = (rank_target[:, 0, :, None]*rank_target[:, 1, None, :]).flatten(1)
    occupancy = in_boxes(part['xyz'], labels)
    occupancy = occupancy & ~(occupancy.all(1) & part['known_ref'][:, None])[:, None]
    positive = occupancy.sum(-1)
    supportable = known & (positive > 0)
    true_yaw = labels[..., 6]
    true_angle = torch.stack(((2*true_yaw).cos(), (2*true_yaw).sin()), -1)
    scale = labels[..., 3:6].clamp_min(.2)
    aspect = ((scale[..., 0]-scale[..., 1]).abs()/scale[..., :2].amax(-1)).clamp(.15, 1.)
    total = 0.
    for box, angle, weight, logits, pairs, _ in output['passes']:
        chosen = gather_slots(box, slot)
        slope = (chosen[..., :3]-labels[..., :3])/(scale+.25)
        center = F.smooth_l1_loss(slope, torch.zeros_like(scale), reduction='none', beta=.4).mean(-1)
        extent = F.smooth_l1_loss(chosen[..., 3:6].log(), scale.log(), reduction='none', beta=.35).mean(-1)
        rotation = aspect*(1.-(gather_slots(angle, slot)*true_angle).sum(-1))
        ranking = -(rank_target*logits.log_softmax(-1)).sum(-1)
        pair_loss = -(pair_target*pairs.flatten(1).log_softmax(-1)).sum(-1)
        selected_weight = gather_slots(weight, slot).float()
        support = -(selected_weight.clamp_min(1e-9).log()*occupancy).sum(-1)/positive.clamp_min(1)
        total += (2.*center*known + extent*known + .4*rotation*known).sum()/known.sum().clamp_min(1)
        total += .8*(ranking*covered).sum()/covered.sum().clamp_min(1)
        total += .24*(support*supportable).sum()/supportable.sum().clamp_min(1)
        total += .22*(pair_loss*valid_pair).sum()/valid_pair.sum().clamp_min(1)
    return total/len(output['passes'])
