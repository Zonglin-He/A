"""A recurrent role-conditioned support, box and geometry decoder."""
import math

import torch
from torch import nn
import torch.nn.functional as F


def normalized_coordinate(values):
    return values.sign() * torch.log1p(values.abs()) / 4.


def mlp(in_features, width, out_features):
    return nn.Sequential(nn.Linear(in_features, width), nn.GELU(), nn.Linear(width, out_features))


def pool(x, mask):
    w = mask.float()
    return (w @ x) / w.sum(-1, keepdim=True).clamp_min(1.)


def gather_slots(x, indices):
    return x.gather(2, indices[..., None, None].expand(-1, -1, 1, x.shape[-1])).squeeze(2)


class Decoder(nn.Module):
    """Four slots per role, with one tied update applied three times."""
    def __init__(self, arm, width=96, slots=4, turns=3):
        super().__init__()
        self.feedback = arm != 'without_feedback'
        self.use_rgb = arm != 'without_rgb'
        self.slots = slots
        self.turns = turns
        self.point = nn.Sequential(nn.LayerNorm(288), nn.Linear(288, width))
        self.image = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, width, bias=False))
        self.position = mlp(6, width, width)
        self.token = nn.Sequential(nn.LayerNorm(288), nn.Linear(288, width))
        self.language = mlp(3*width, width, width)
        self.anchor = mlp(6, width, width)
        self.role_slot = nn.Parameter(torch.randn(2, slots, width)*.03)
        self.scene = nn.LayerNorm(width)
        self.query = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width))
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.revise = nn.GRUCell(2*width, width)
        self.statistics = mlp(11, width, width)
        self.box = mlp(2*width, width, 9)
        self.slot_score = mlp(width, width, 1)
        self.pair = mlp(2*width + 11, width, 1)
        self.from_ref = nn.Linear(width, width)
        self.from_subject = nn.Linear(width, width)
        self.relation_gate = nn.Parameter(torch.tensor(-1.))
        self.spatial_gate = nn.Parameter(torch.tensor(-1.5))
        self.candidate_proj = mlp(4, width//2, 1)
        nn.init.zeros_(self.box[-1].weight)
        nn.init.zeros_(self.box[-1].bias)
        nn.init.zeros_(self.slot_score[-1].weight)
        nn.init.zeros_(self.slot_score[-1].bias)
        nn.init.normal_(self.pair[-1].weight, std=.01)
        nn.init.zeros_(self.pair[-1].bias)

    def initial(self, evidence):
        points, uv, valid = evidence['xyz'], evidence['uv'].float(), evidence['in_image']
        scene = self.point(evidence['points'].float())
        if self.use_rgb:
            scene = scene + self.image(evidence['rgb'].float()) * evidence['visible'][..., None]
        pose = torch.cat((normalized_coordinate(points), 2*uv-1, valid[..., None]), -1)
        scene = self.scene(scene + self.position(pose))
        tokens = self.token(evidence['text'].float())
        phrase = pool(tokens, evidence['valid'][:, None])
        role_phrase = pool(tokens, evidence['role_mask'])
        predicate = role_phrase[:, 2:3]
        language = self.language(torch.cat((role_phrase[:, :2], phrase.expand(-1, 2, -1),
                                             predicate.expand(-1, 2, -1)), -1))
        i = evidence['anchors']
        n = points.shape[0]
        prior = evidence['prior'][:, None].expand(-1, 2, -1, -1)
        priors = prior.gather(2, i[..., None].expand(n, 2, self.slots, 6))
        anchor_pose = torch.cat((normalized_coordinate(priors[..., :3]), priors[..., 3:].log()), -1)
        state = language[:, :, None] + self.anchor(anchor_pose) + self.role_slot[None]
        scores = evidence['prior_scores'].gather(-1, i)
        scores = scores - scores.amax(-1, keepdim=True)
        return scene, language, priors, state, .22*scores.clamp(min=-15)

    def geometry(self, points, weight, uv, in_image, anchors):
        centroid = torch.einsum('brkn,bnd->brkd', weight, points)
        offset = points[:, None, None] - centroid[..., None, :]
        var = (weight[..., None] * offset.square()).sum(-2).clamp_min(1e-5)
        xy = (weight * offset[..., 0] * offset[..., 1]).sum(-1, keepdim=True)
        xy = xy / (var[..., :1]*var[..., 1:2]).sqrt().clamp_min(1e-5)
        projected_mass = (weight * in_image[:, None, None]).sum(-1)
        pixel = torch.einsum('brkn,bnd->brkd', weight * in_image[:, None, None], uv)
        pixel = pixel / projected_mass[..., None].clamp_min(1e-5)
        center, dim = anchors[..., :3], anchors[..., 3:]
        shift = (centroid-center) / (dim+.4)
        moment = torch.cat((normalized_coordinate(centroid), (.05+var.sqrt()).log(),
                            xy, 2*pixel-1, projected_mass[..., None],
                            shift.norm(dim=-1, keepdim=True).clamp_max(10.)), -1)
        return self.statistics(moment), centroid, pixel, projected_mass

    def box_readout(self, anchors, state, features, centroid):
        center, dim = anchors[..., :3], anchors[..., 3:]
        d = self.box(torch.cat((state, features), -1))
        fraction = (d[..., 6:7]-2.).sigmoid()*.65
        # Centres are continuous throughout all passes. Proposal coordinates are only priors.
        center = center + fraction*(centroid-center) + d[..., :3].tanh()*(dim+1.5)
        dim = (dim * (.95*d[..., 3:6].tanh()).exp()).clamp(.05, 60.)
        yaw_vector = F.normalize(torch.stack((1+d[..., 7], d[..., 8]), -1), dim=-1, eps=1e-5)
        # The double-angle vector carries the yaw loss and geometry gradients.
        # Radians are an output representation and have a singular Jacobian at zero.
        axis = yaw_vector.detach().float()
        yaw = torch.atan2(axis[..., 1], axis[..., 0])*.5
        return torch.cat((center, dim, yaw[..., None]), -1), yaw_vector

    def interaction(self, state, boxes, angles, pixels, visible, unaries, has_ref):
        subject, ref = state.unbind(1)
        bs, br = boxes.unbind(1)
        delta = bs[:, :, None, :3]-br[:, None, :, :3]
        scale = (bs[:, :, None, 3:6]+br[:, None, :, 3:6])*.5+.25
        relative = normalized_coordinate(delta/scale)
        ratio = bs[:, :, None, 3:6].log()-br[:, None, :, 3:6].log()
        projected = pixels[:, 0, :, None, :]-pixels[:, 1, None, :, :]
        joint_visible = visible[:, 0, :, None]*visible[:, 1, None, :]
        confidence = joint_visible.clamp_min(1e-8).sqrt()*(joint_visible > 0)
        relative_yaw = angles[:, 0, :, None, :]-angles[:, 1, None, :, :]
        geom = torch.cat((relative, ratio, 4*projected*confidence[..., None],
                          confidence[..., None], relative_yaw), -1)
        u = subject[:, :, None].expand(-1, -1, self.slots, -1)
        v = ref[:, None].expand(-1, self.slots, -1, -1)
        compatibility = self.pair(torch.cat((u, v, geom), -1)).squeeze(-1)
        compatibility = compatibility * has_ref[:, None, None]
        a = (compatibility + unaries[:, 1, None]).softmax(-1)
        b = (compatibility.transpose(1, 2) + unaries[:, 0, None]).softmax(-1)
        messages = torch.stack((a @ self.from_ref(ref), b @ self.from_subject(subject)), 1)
        return compatibility, messages*has_ref[:, None, None, None]

    def proposal_logits(self, evidence, boxes, logits):
        # This categorical branch is trained identically in all arms.
        prior = evidence['prior'][:, None]
        box = boxes[:, 0, :, None]
        diff = ((prior[..., :3] - box[..., :3]) / (box[..., 3:6]+.35)).square().sum(-1)
        size = (prior[..., 3:6].log()-box[..., 3:6].log()).abs()
        native = evidence['prior_scores'][:, 0]
        matched = self.candidate_proj(torch.cat((diff.clamp_max(25)[..., None]/25, size), -1))
        affinity = -.55*torch.log1p(diff) -.30*size.mean(-1) + matched.squeeze(-1)
        return torch.logsumexp(logits[:, 0, :, None] + affinity, 1) + .15*native

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
            if self.feedback:
                compatibility, message = self.interaction(state, boxes, angles, pixels, mass, unaries, has_ref)
                if step+1 < self.turns:
                    state = state + self.relation_gate.sigmoid()*message
                    dr = (points[:, None, None]-boxes[..., None, :3]) / (boxes[..., None, 3:6]*.65+.5)
                    # Both roles' current support can change the next iteration's geometric read.
                    current_spatial = static_spatial - self.spatial_gate.sigmoid()*torch.log1p(dr.square().sum(-1))
            joint = unaries[:, 0, :, None] + unaries[:, 1, None, :] + compatibility
            marginal = torch.stack((joint.logsumexp(-1), joint.logsumexp(-2)), 1)
            # The natural-reference latent variable cannot alter the no-reference score.
            marginal = torch.where(has_ref[:, None, None], marginal, unaries)
            intermediate.append((boxes, angles, weights, marginal, joint, unaries))
        boxes, yaw, weights, marginal, joint, unaries = intermediate[-1]
        pair = joint.flatten(1).argmax(-1)
        best = torch.stack((pair // self.slots, pair % self.slots), -1)
        best = torch.where(has_ref[:, None], best, unaries.argmax(-1))
        return dict(boxes=gather_slots(boxes, best), support=gather_slots(weights, best),
                    choices=best, passes=intermediate, proposal_logits=self.proposal_logits(evidence, boxes, marginal))


def in_boxes(points, boxes):
    delta = points[:, None] - boxes[:, :, None, :3]
    c, s = boxes[..., 6:].cos(), boxes[..., 6:].sin()
    local = torch.stack((c*delta[..., 0] + s*delta[..., 1],
                         -s*delta[..., 0] + c*delta[..., 1], delta[..., 2]), -1)
    result = (local.abs() <= boxes[..., None, 3:6]*.5+1e-5).all(-1)
    return result & (boxes[..., 3:6] > 0).all(-1)[..., None]


def box_distance(boxes, labels):
    scale = labels[..., None, 3:6].clamp_min(.2)
    center = ((boxes[..., :3]-labels[..., None, :3]) / (scale+.25)).abs().mean(-1)
    size = (boxes[..., 3:6].clamp_min(.03).log()-scale.log()).abs().mean(-1)
    return center + .6*size


def objective(evidence, output):
    labels = evidence['gt']
    known = torch.stack((torch.ones_like(evidence['known_ref']), evidence['known_ref']), 1).float()
    count = known.sum().clamp_min(1.)
    # Assign each role to a slot once using only its source box, then supervise every pass.
    last = output['passes'][-1][0]
    slot = box_distance(last.detach(), labels).argmin(-1)
    occupancy = in_boxes(evidence['xyz'], labels)
    both = occupancy.all(1) & evidence['known_ref'][:, None]
    occupancy = occupancy & ~both[:, None]
    n_positive = occupancy.sum(-1)
    supported = known*(n_positive > 0)
    gt_yaw = labels[..., 6]
    truth_angle = torch.stack(((2*gt_yaw).cos(), (2*gt_yaw).sin()), -1)
    scale = labels[..., 3:6].clamp_min(.2)
    aspect = ((scale[..., 0]-scale[..., 1]).abs() / scale[..., :2].amax(-1)).clamp(.15, 1.)
    loss = 0.
    for box, angle, weights, logits, pairs, _ in output['passes']:
        result = gather_slots(box, slot)
        angle = gather_slots(angle, slot)
        weights = gather_slots(weights, slot)
        center = F.smooth_l1_loss((result[..., :3]-labels[..., :3])/(scale+.25),
                                 torch.zeros_like(scale), reduction='none', beta=.4).mean(-1)
        size = F.smooth_l1_loss(result[..., 3:6].log(), scale.log(),
                               reduction='none', beta=.35).mean(-1)
        rotation = aspect*(1-(angle*truth_angle).sum(-1))
        ranking = -logits.log_softmax(-1).gather(-1, slot[..., None]).squeeze(-1)
        support = -(weights.clamp_min(1e-9).log()*occupancy).sum(-1) / n_positive.clamp_min(1)
        pair_target = slot[:, 0]*box.shape[2] + slot[:, 1]
        pair_loss = F.cross_entropy(pairs.flatten(1), pair_target, reduction='none')
        pair_loss = (pair_loss*known[:, 1]).sum()/known[:, 1].sum().clamp_min(1)
        loss = loss + ((2.0*center + 1.0*size + .40*rotation + .55*ranking)*known).sum()/count
        loss = loss + .24*(support*supported).sum()/supported.sum().clamp_min(1) + .22*pair_loss
    target_candidate = box_distance(evidence['prior'], labels[:, 0]).argmin(-1)
    fixed_loss = F.cross_entropy(output['proposal_logits'], target_candidate)
    return loss / len(output['passes']) + .15*fixed_loss
