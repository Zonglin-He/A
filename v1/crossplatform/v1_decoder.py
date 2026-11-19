"""The fixed two-round v1 language–instance decoder."""

import math

import torch
from torch import nn


def _coordinate(value):
    return value.sign() * torch.log1p(value.abs()) / 4.


def _mlp(input_width, width, output_width):
    return nn.Sequential(nn.Linear(input_width, width), nn.GELU(), nn.Linear(width, output_width))


def _pool(tokens, mask):
    weights = mask.float()
    return (weights @ tokens) / weights.sum(-1, keepdim=True).clamp_min(1.)


class JointDecoder(nn.Module):
    """Read four subject and four reference slots, then select an original anchor."""

    slots = 4

    def __init__(self):
        super().__init__()
        width = 96
        self.point = nn.Sequential(nn.LayerNorm(288), nn.Linear(288, width))
        self.position = _mlp(6, width, width)
        self.token = nn.Sequential(nn.LayerNorm(288), nn.Linear(288, width))
        self.language = _mlp(3 * width, width, width)
        self.anchor = _mlp(6, width, width)
        self.role_slot = nn.Parameter(torch.empty(2, self.slots, width))
        self.scene = nn.LayerNorm(width)
        self.query = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width))
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.revise = nn.GRUCell(2 * width, width)
        self.statistics = _mlp(11, width, width)
        self.slot_score = _mlp(width, width, 1)
        self.pair = _mlp(2 * width + 11, width, 1)
        self.from_ref = nn.Linear(width, width)
        self.from_subject = nn.Linear(width, width)
        self.relation_gate = nn.Parameter(torch.empty(()))
        self.spatial_gate = nn.Parameter(torch.empty(()))

    def _scene_and_roles(self, evidence):
        uv, visible = evidence['uv'].float(), evidence['visible']
        pose = torch.cat((_coordinate(evidence['xyz']), (2 * uv - 1) * visible[..., None],
                          visible[..., None]), -1)
        scene = self.scene(self.point(evidence['points'].float()) + self.position(pose))
        tokens = self.token(evidence['text'].float())
        phrase = _pool(tokens, evidence['valid'][:, None])
        roles = _pool(tokens, evidence['role_mask'])
        language = self.language(torch.cat((roles[:, :2], phrase.expand(-1, 2, -1),
                                             roles[:, 2:3].expand(-1, 2, -1)), -1))
        return scene, language

    @staticmethod
    def _spatial_bias(points, anchors):
        anchors = anchors[:, None]
        delta = points[:, None, None] - anchors[..., None, :3]
        cos, sin = anchors[..., 6:].cos(), anchors[..., 6:].sin()
        offset = torch.stack((cos * delta[..., 0] + sin * delta[..., 1],
                              -sin * delta[..., 0] + cos * delta[..., 1], delta[..., 2]), -1)
        return -torch.log1p((offset / (anchors[..., None, 3:6] * .65 + .5)).square().sum(-1))[:, 0]

    def _support_statistics(self, points, weight, uv, visible, anchors):
        weight, anchors = weight[:, None], anchors[:, None]
        centroid = torch.einsum('brkn,bnd->brkd', weight, points)
        offset = points[:, None, None] - centroid[..., None, :]
        variance = (weight[..., None] * offset.square()).sum(-2).clamp_min(1e-5)
        xy = (weight * offset[..., 0] * offset[..., 1]).sum(-1, keepdim=True)
        xy = xy / (variance[..., :1] * variance[..., 1:2]).sqrt().clamp_min(1e-5)
        visible_weight = weight * visible[:, None, None]
        mass = visible_weight.sum(-1)
        pixel = torch.einsum('brkn,bnd->brkd', visible_weight, uv)
        pixel = pixel / mass[..., None].clamp_min(1e-5)
        shift = (centroid - anchors[..., :3]) / (anchors[..., 3:6] + .4)
        moment = torch.cat((_coordinate(centroid), (.05 + variance.sqrt()).log(),
                            xy, 2 * pixel - 1, mass[..., None],
                            shift.norm(dim=-1, keepdim=True).clamp_max(10.)), -1)
        return self.statistics(moment)[:, 0], pixel[:, 0], mass[:, 0]

    def _pair_scores(self, state, boxes, angle, pixels, mass, active):
        n = self.slots
        subject, reference = state[:, :n], state[:, n:]
        subject_box, ref_box = boxes[:, :n], boxes[:, n:]
        delta = subject_box[:, :, None, :3] - ref_box[:, None, :, :3]
        subject_size, ref_size = subject_box[:, :, None, 3:6], ref_box[:, None, :, 3:6]
        relative = _coordinate(delta / ((subject_size + ref_size) * .5 + .25))
        ratio = subject_size.log() - ref_size.log()
        image_delta = pixels[:, :n, None] - pixels[:, None, n:]
        shared = mass[:, :n, None] * mass[:, None, n:]
        confidence = shared.clamp_min(1e-8).sqrt() * (shared > 0)
        view = torch.cat((4 * image_delta * confidence[..., None], confidence[..., None]), -1)
        yaw = angle[:, :n, None] - angle[:, None, n:]
        a = subject[:, :, None].expand(-1, -1, n, -1)
        b = reference[:, None].expand(-1, n, -1, -1)
        return self.pair(torch.cat((a, b, relative, ratio, view, yaw), -1)).squeeze(-1) * active[:, None, None]

    def _read(self, state, language, key, value, spatial, points, uv, visible, anchors, prior):
        query = self.query(state + language)
        logits = query @ key.transpose(1, 2) / math.sqrt(query.shape[-1])
        weight = (logits.float() + spatial).softmax(-1)
        summary = weight @ value
        statistic, pixels, mass = self._support_statistics(points, weight, uv, visible, anchors)
        incoming = torch.cat((summary, statistic), -1).flatten(0, 1)
        state = self.revise(incoming, state.flatten(0, 1)).reshape_as(state)
        unaries = prior + .75 * self.slot_score(state).float().squeeze(-1).tanh()
        return state, unaries, pixels, mass

    @torch.inference_mode()
    def forward(self, evidence):
        n = self.slots
        points, uv = evidence['xyz'], evidence['uv'].float()
        active, visible = evidence['has_reference'], evidence['visible']
        scene, role = self._scene_and_roles(evidence)
        raw = torch.stack((evidence['subject_boxes'], evidence['reference_boxes']), 1)
        anchors = torch.cat((raw[:, 0], raw[:, 1]), 1)
        ref = evidence['ref_logit']
        ref_prior = .22 * (ref - ref.amax(-1, keepdim=True)).clamp_min(-15)
        prior = torch.cat((evidence['base_logit'], ref_prior), 1)
        language = torch.cat((role[:, 0, None].expand(-1, n, -1),
                              role[:, 1, None].expand(-1, n, -1)), 1)
        pose = torch.cat((_coordinate(raw[..., :3]), raw[..., 3:6].log()), -1)
        initial = role[:, :, None] + self.anchor(pose) + self.role_slot[None]
        state = torch.cat((initial[:, 0], initial[:, 1]), 1)
        key, value = self.key(scene), self.value(scene)
        anchor_bias = self._spatial_bias(points, anchors)
        static = .65 * anchor_bias
        angle = torch.stack(((2 * anchors[..., 6]).cos(), (2 * anchors[..., 6]).sin()), -1)
        state, unaries, pixels, mass = self._read(
            state, language, key, value, static, points, uv, visible, anchors, prior)
        pair = self._pair_scores(state, anchors, angle, pixels, mass, active)
        subject, reference = state[:, :n], state[:, n:]
        ref_weights = (pair + unaries[:, None, n:]).softmax(-1)
        subject_weights = (pair.transpose(1, 2) + unaries[:, None, :n]).softmax(-1)
        subject_delta = (ref_weights @ self.from_ref(reference)) * active[:, None, None]
        reference_delta = (subject_weights @ self.from_subject(subject)) * active[:, None, None]
        gate = self.relation_gate.sigmoid()
        state = torch.cat((subject + gate * subject_delta, reference + gate * reference_delta), 1)
        feedback = torch.arange(2 * n, device=points.device) < n
        spatial = static + self.spatial_gate.sigmoid() * anchor_bias * feedback[None, :, None]
        state, unaries, pixels, mass = self._read(
            state, language, key, value, spatial, points, uv, visible, anchors, prior)
        pair = self._pair_scores(state, anchors, angle, pixels, mass, active)
        subject, reference = unaries[:, :n], unaries[:, n:].log_softmax(-1)
        joint = subject[:, :, None] + reference[:, None, :] + .5 * pair.float().tanh()
        scores = torch.where(active[:, None], joint.logsumexp(-1), subject)
        row = torch.arange(len(scores), device=scores.device)
        chosen = scores.argmax(-1)
        ref_choice = joint[row, chosen].argmax(-1)
        return dict(choices=torch.stack((chosen, ref_choice), -1),
                    boxes=torch.stack((raw[row, 0, chosen], raw[row, 1, ref_choice]), 1),
                    subject_logits=scores)
