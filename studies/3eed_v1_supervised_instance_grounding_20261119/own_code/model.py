"""Trainable instance support and dual-role readout on frozen 3EED evidence."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


NATIVE = 256
VOTES = 64
ROI_POINTS = 64
WIDTH = 48


def mlp(a, b, c):
    return nn.Sequential(nn.Linear(a, b), nn.GELU(), nn.Linear(b, c))


def wrap_yaw(x):
    return torch.remainder(x + math.pi / 2, math.pi) - math.pi / 2


def heading_transform(offset, yaw):
    c, s = yaw.cos(), yaw.sin()
    x, y, z = offset.unbind(-1)
    return torch.stack((c*x+s*y, -s*x+c*y, z), -1)


def box_update(base, delta):
    local = .65 * base[..., 3:6] * delta[..., :3].tanh()
    shift = heading_transform(local, -base[..., 6])
    size = (base[..., 3:6] * (.70 * delta[..., 3:6].tanh()).exp()).clamp(.05, 24)
    yaw = wrap_yaw(base[..., 6] + math.pi/2 * delta[..., 6].tanh())
    return torch.cat((base[..., :3] + shift, size, yaw[..., None]), -1)


def projected(boxes, camera):
    world = F.pad(boxes[..., :3], (0, 1), value=1.)
    projected = world @ camera.transpose(-1, -2)
    depth = projected[..., 2]
    xy = projected[..., :2] / depth[..., None].clamp_min(.1)
    visible = (depth > .1) & ((xy >= 0) & (xy <= 1)).all(-1)
    return xy.clamp(-2., 3.), visible


@torch.no_grad()
def vote_indices(boxes, scores):
    """Stable score-prioritized spatial diversification of learned centers."""
    centers = boxes[..., :3].cpu().numpy()
    scores = scores.cpu().numpy()
    selected = []
    for xyz, score in zip(centers, scores):
        order = np.argsort(-score, kind='stable')
        cells = np.floor(xyz[order] / .75).astype(np.int32)
        _, positions = np.unique(cells, axis=0, return_index=True)
        keep = order[np.sort(positions)[:VOTES]]
        if len(keep) < VOTES:
            extra = order[~np.isin(order, keep)][:VOTES-len(keep)]
            keep = np.r_[keep, extra]
        selected.append(keep)
    return torch.as_tensor(np.stack(selected), device=boxes.device)


@torch.no_grad()
def neighborhoods(boxes, points):
    # Squared distances normalized by candidate extent, without a [B,C,N,3] broadcast.
    center = boxes[..., :3]
    metric = (boxes[..., 3:6] * .60 + .18).square().reciprocal()
    order = metric @ points.square().transpose(1, 2)
    order -= (2 * center * metric) @ points.transpose(1, 2)
    return order.topk(ROI_POINTS, dim=-1, largest=False, sorted=False).indices


def overlap_matrix(boxes):
    half = boxes[..., 3:6] / 2
    lo, hi = boxes[..., :3]-half, boxes[..., :3]+half
    shared = (torch.minimum(hi[:, :, None], hi[:, None, :])
              - torch.maximum(lo[:, :, None], lo[:, None, :])).clamp_min(0).prod(-1)
    volume = boxes[..., 3:6].prod(-1)
    return shared / (volume[:, :, None]+volume[:, None, :]-shared).clamp_min(1e-8)


class InstanceGroundingHead(nn.Module):
    """Score all carriers using learned point membership, two language roles and a null reference."""

    def __init__(self):
        super().__init__()
        self.seed = nn.Sequential(nn.LayerNorm(288), nn.Linear(288, WIDTH), nn.GELU())
        self.seed_pose = mlp(3, 32, WIDTH)
        self.vote = mlp(WIDTH, 64, 8)
        self.seed_match = nn.Linear(WIDTH, 64)
        self.anchor_sem = nn.Linear(64, WIDTH)
        self.geometry = mlp(11, 64, WIDTH)
        self.support_value = mlp(WIDTH+3, 64, WIDTH)
        self.support_logits = mlp(3, 32, 1)
        self.support_stats = mlp(7, 32, WIDTH)
        self.merge = nn.LayerNorm(WIDTH)
        self.refinement = mlp(WIDTH, 64, 7)
        self.quality = mlp(WIDTH, 48, 2)
        self.role_text = nn.Sequential(nn.LayerNorm(288), nn.Linear(288, WIDTH), nn.GELU())
        self.entity = mlp(3*WIDTH+5, 64, 1)
        self.subject_pair = nn.Linear(WIDTH, 24)
        self.reference_pair = nn.Linear(WIDTH, 24)
        self.null = mlp(2, 24, 1)
        self.prior_scale = nn.Parameter(torch.tensor(.15))
        self.relation_scale = nn.Parameter(torch.tensor(0.))
        nn.init.zeros_(self.refinement[-1].weight)
        nn.init.zeros_(self.refinement[-1].bias)
        nn.init.constant_(self.null[-1].bias, 3.0)

    def pool(self, batch, use_votes, use_mask):
        points, xyz = batch['points'], batch['seed_xyz']
        seed = self.seed(batch['seed_feat'].float())
        seed = seed + self.seed_pose(xyz.sign() * xyz.abs().log1p() / 4.)
        vote = self.vote(seed)
        vote_center = xyz + 5. * vote[..., :3].tanh()
        vote_size = (.30 + F.softplus(vote[..., 3:6]) * 1.5).clamp(.10, 24.)
        vote_angle = math.pi / 2 * vote[..., 6].tanh()
        vote_box = torch.cat((vote_center, vote_size, vote_angle[..., None]), -1)
        seed_query = F.normalize(self.seed_match(seed), dim=-1)
        vote_logit = vote[..., 7] + .20 * batch['seed_obj'].float().clamp(-8., 8.)
        proposal, queries = batch['anchors'], batch['anchor_query'].float()
        if use_votes:
            relevance = (seed_query @ batch['token_proj'][:, :2].float().transpose(1, 2)).amax(-1)
            chosen = vote_indices(vote_box, vote_logit + 2. * relevance)
            row = torch.arange(len(seed), device=seed.device)[:, None]
            proposal = torch.cat((proposal, vote_box[row, chosen]), 1)
            queries = torch.cat((queries, seed_query[row, chosen]), 1)
        proposal = torch.cat((proposal[..., :3], proposal[..., 3:6].clamp(.05, 32), proposal[..., 6:7]), -1)
        project, visible = projected(proposal, batch['camera'])
        geo = torch.cat((proposal[..., :3].sign()*proposal[..., :3].abs().log1p()/4,
                         proposal[..., 3:6].log(), (2*proposal[..., 6:7]).cos(),
                         (2*proposal[..., 6:7]).sin(), project, visible[..., None]), -1)
        anchor = self.anchor_sem(queries) + self.geometry(geo)
        patch = neighborhoods(proposal.detach(), points)
        rows = torch.arange(len(points), device=points.device)[:, None, None]
        patch_xyz = points[rows, patch]
        patch_feature = seed[rows, batch['nearest'][rows, patch].long()]
        local = heading_transform(patch_xyz - proposal[..., None, :3], proposal[..., 6, None])
        local = local / (proposal[..., None, 3:6] / 2 + .08)
        support = self.support_value(torch.cat((patch_feature, local.clamp(-3., 3.)), -1))
        logit = (support * anchor[..., None, :]).sum(-1) / math.sqrt(WIDTH)
        logit = logit + self.support_logits(local.clamp(-3., 3.)).squeeze(-1)
        geometric = local.square().sum(-1).clamp_max(36.).mul(-.7).exp()
        weights = geometric * logit.sigmoid() if use_mask else geometric
        denominator = weights.sum(-1, keepdim=True).clamp_min(1e-6)
        statistics = torch.cat(((weights[..., None]*local.clamp(-3.,3.)).sum(-2)/denominator,
                                (weights[..., None]*local.clamp(-3.,3.).square()).sum(-2)/denominator,
                                weights.mean(-1, keepdim=True)), -1)
        embedding = self.merge(anchor + (weights[..., None]*support).sum(-2)/denominator
                               + self.support_stats(statistics))
        repaired = box_update(proposal, self.refinement(embedding))
        quality = self.quality(embedding).sigmoid()
        choose = (quality[..., 1] >= quality[..., 0] + .025) & (geometric.sum(-1) >= 1.)
        output = torch.where(choose[..., None], repaired, proposal)
        return dict(proposal=proposal, refined=repaired, boxes=output, quality=quality,
                    geometry_changed=choose, embedding=embedding, queries=queries, patch=patch,
                    support_logit=logit, support_strength=weights.mean(-1),
                    instance_weight=weights, geometric_weight=geometric,
                    vote_box=vote_box, vote_logit=vote_logit)

    def unaries(self, batch, pool):
        vectors = pool['embedding']
        size = vectors.shape[1]
        roles = batch['text'].float()
        phrase = self.role_text(roles[:, 3])[:, None].expand(-1, size, -1)
        text = self.role_text(roles[:, :2])
        affinity = pool['queries'] @ batch['token_proj'][:, :2].float().transpose(1, 2)
        xy, visible = projected(pool['boxes'], batch['camera'])
        bands = batch['bands']
        edge = (bands[..., 0][:, None] - xy).clamp_min(0) + (xy - bands[..., 1][:, None]).clamp_min(0)
        edge = edge * (bands[..., 1] - bands[..., 0] < .999)[:, None] * visible[..., None]
        q = pool['quality'].amax(-1)
        ordinal = batch['reference_ord'].float()[:, None] * (2*xy[..., 0]-1) * visible
        logits = []
        for index in range(2):
            role = text[:, index, None].expand(-1, size, -1)
            spatial = -edge if index == 0 else torch.stack((ordinal, torch.zeros_like(ordinal)), -1)
            cue = torch.cat((affinity[..., index:index+1], spatial, q[..., None],
                             pool['support_strength'][..., None]), -1)
            prediction = self.entity(torch.cat((vectors, role, phrase, cue), -1)).squeeze(-1)
            logits.append(prediction + 3. * affinity[..., index])
        prior = F.pad(batch['prior'], (0, size-NATIVE), value=-2.0)
        subject = logits[0] + F.softplus(self.prior_scale) * prior
        return subject, logits[1], xy, visible

    def joint(self, batch, pool, subject, reference, xy, visible, mismatch=False):
        h = pool['embedding']
        side = batch['relation'].float()
        active = batch['reference_phrase'] & (side != 0)
        predicted_overlap = overlap_matrix(pool['boxes'].detach())
        duplicate = predicted_overlap >= .65
        density = duplicate.sum(-1).clamp_min(1).float()
        ref_score = reference - density.log()
        if mismatch:
            ref_score = ref_score.roll(13, dims=-1)
        features = torch.stack((ref_score.amax(-1), pool['support_strength'].amax(-1)), -1)
        null_score = self.null(features)
        ref_and_null = torch.cat((ref_score, null_score), -1)
        log_ref = ref_and_null.log_softmax(-1)
        left, right = self.subject_pair(h), self.reference_pair(h)
        appearance = left @ right.transpose(1, 2) / math.sqrt(left.shape[-1])
        homogeneous = F.pad(pool['boxes'][...,:3],(0,1),value=1.)
        depth = (homogeneous * batch['camera'][:,None,2]).sum(-1).clamp_min(.1).log()
        coordinates = torch.cat((xy,depth[...,None]),-1)
        axis = (side.abs().long()-1).clamp(0,2)
        location = coordinates.gather(-1,axis[:,None,None].expand(-1,coordinates.shape[1],1)).squeeze(-1)
        order = (location[:,:,None]-location[:,None,:]) * side.sign()[:,None,None]
        separations = torch.cdist(pool['boxes'][...,:3],pool['boxes'][...,:3])
        proximity = 1. - separations/3.
        geometry = torch.where((side == 4)[:,None,None],proximity.tanh(),(order/.10).tanh())
        seen = visible[:, :, None] & visible[:, None, :]
        relation = 1.5 * (appearance + geometry*seen).tanh()
        valid = active[:,None,None] & (~duplicate) & seen
        relation = relation * valid
        evidence = torch.logaddexp(
            (log_ref[:, None, :-1] + relation).logsumexp(-1), log_ref[:, -1:])
        score = subject + self.relation_scale.sigmoid() * evidence * active[:, None]
        # The physical pair winner is either valid or explicitly UNKNOWN.
        unknown_mass = log_ref[:, None, :-1].masked_fill(valid, -torch.inf).logsumexp(-1)
        unknown_mass = torch.logaddexp(unknown_mass, log_ref[:, -1:])
        pair = (ref_score[:, None] + relation).masked_fill(~valid,-1e4)
        pair = torch.cat((pair, unknown_mass[..., None] + ref_and_null.logsumexp(-1)[:,None,None]), -1)
        return dict(joint=score, pair=pair, reference_logits=ref_and_null, pair_valid=valid)

    def forward(self, batch, *, use_votes=True, use_mask=True, mismatch=False):
        pool = self.pool(batch,use_votes,use_mask)
        subject, reference, xy, visible = self.unaries(batch,pool)
        return dict(pool, subject=subject, unsuppressed_reference=reference,
                    camera_xy=xy, camera_visible=visible,
                    **self.joint(batch,pool,subject,reference,xy,visible,mismatch))


def proxy_overlap(boxes, gt, valid):
    """A differentiable training proxy. Final scoring uses exact oriented IoU."""
    delta = heading_transform(boxes[:, :, None, :3]-gt[:,None,:,:3],gt[:,None,:,6])
    a, b = boxes[:, :, None,3:6], gt[:, None,:,3:6]
    shared = (torch.minimum(delta+a/2,b/2)-torch.maximum(delta-a/2,-b/2)).clamp_min(0).prod(-1)
    union = a.prod(-1)+b.prod(-1)-shared
    anis = (b[...,0]-b[...,1]).abs()/(b[...,0]+b[...,1])
    theta = boxes[...,6,None]-gt[:,None,:,6]
    yaw_term = 1.-.6 * anis * theta.sin().square()
    return shared / union.clamp_min(1e-8) * yaw_term * valid[:,None]


def ownership(points, boxes, valid):
    delta = heading_transform(points[:, :, None] - boxes[:, None,:, :3], boxes[:,None,:,6])
    position = delta.abs() / (boxes[:,None,:,3:6] / 2).clamp_min(.05)
    distance = position.amax(-1)
    inside = (distance <= 1.) & valid[:, None]
    memberships = inside.sum(-1)
    nearest = distance.masked_fill(~valid[:, None], torch.inf).amin(-1)
    owner = inside.long().argmax(-1)
    return owner.masked_fill(memberships!=1,-1), memberships>1, nearest


def masked_mean(values, mask):
    return (values * mask).sum() / mask.sum().clamp_min(1)


def weighted_binary(logit, label, valid):
    loss = F.binary_cross_entropy_with_logits(logit, label.float(), reduction='none')
    return .5*(masked_mean(loss, label & valid)+masked_mean(loss, (~label) & valid))


def regression(boxes, target):
    normalized = (boxes[...,:3]-target[...,:3]) / target[...,3:6].clamp_min(.30)
    dimension = boxes[...,3:6].log()-target[...,3:6].log()
    error = F.smooth_l1_loss(normalized, torch.zeros_like(normalized), reduction='none').mean(-1)
    error += F.smooth_l1_loss(dimension,torch.zeros_like(dimension),reduction='none').mean(-1)
    anis = (target[...,3]-target[...,4]).abs()/(target[...,3]+target[...,4])
    return error + .35*anis*(1.-(2*(boxes[...,6]-target[...,6])).cos())


def weights_for_role(quality):
    # Several good boxes can describe the same role. Never select one arbitrary duplicate.
    weight = ((quality - .04).clamp_min(0)).square()
    return weight / weight.sum(-1, keepdim=True).clamp_min(1e-8)


def rank_loss(logits, target, applies):
    return masked_mean(-(logits.log_softmax(-1)*target).sum(-1),
                       applies & (target.sum(-1)>0))


def objective(out, batch):
    gt, valid = batch['gt'], batch['gt_valid']
    b = len(gt)
    row = torch.arange(b, device=gt.device)
    with torch.no_grad():
        proposal_q = proxy_overlap(out['proposal'],gt,valid)
        refined_q = proxy_overlap(out['refined'],gt,valid)
        ranked_q = proxy_overlap(out['boxes'],gt,valid)
        match_q, matched = proposal_q.max(-1)
        chosen_gt = gt[row[:,None],matched]
        matched_refined_q = refined_q.gather(-1, matched[...,None]).squeeze(-1)
        annot = match_q >= .055
        # The original annotation is always supervised. A missing reference is unknown.
        subject = weights_for_role(ranked_q[:,:,0])
        ref_idx = batch['gt_reference'].long().clamp_min(0)
        ref_q = ranked_q[row,:,ref_idx]
        ref_target = weights_for_role(ref_q)
        owners, ambiguous, _ = ownership(batch['points'],gt,valid)
        roi_owner = owners[row[:,None,None],out['patch']]
        roi_ambiguous = ambiguous[row[:,None,None],out['patch']]
        at_instance = roi_owner == matched[...,None]
        seed_owner, seed_ambiguous, boundary = ownership(batch['seed_xyz'],gt,valid)
        seed_gt = gt[row[:,None],seed_owner.clamp_min(0)]
    point_loss = weighted_binary(out['support_logit'],at_instance,annot[...,None] & ~roi_ambiguous)
    positive = seed_owner >= 0
    vote_loss = (weighted_binary(out['vote_logit'],positive,~seed_ambiguous & (boundary<1.8))
                 + 2.*masked_mean(regression(out['vote_box'],seed_gt),positive))
    geo = masked_mean(regression(out['refined'],chosen_gt),annot)
    quality_target = torch.stack((match_q,matched_refined_q), -1)
    quality_loss = masked_mean((out['quality']-quality_target).square().mean(-1), match_q > .02)
    active = (batch['gt_reference'] >= 0) & batch['reference_phrase'] & (batch['relation'] != 0)
    ones = torch.ones(b,device=gt.device,dtype=torch.bool)
    # Unknown natural references do not train pair or null identities.
    subject_logits = torch.where(active[:,None],out['joint'],out['subject'])
    subj_loss = rank_loss(subject_logits,subject,ones) + .5 * rank_loss(out['subject'],subject,ones)
    extended = F.pad(ref_target,(0,1))
    ref_loss = rank_loss(out['reference_logits'],extended,active)
    # Source-only absent-reference control. Remove all proposals that can describe the known referent.
    hidden_reference = (ref_q > .045) & active[:,None]
    without = torch.cat((out['reference_logits'][:,:-1].masked_fill(hidden_reference,-1e4),
                         out['reference_logits'][:,-1:]), -1)
    null_loss = masked_mean(-without.log_softmax(-1)[:,-1],active & hidden_reference.any(-1))
    valid_pair = out['pair_valid']
    pair_target = subject[:,:,None]*ref_target[:,None,:] * valid_pair
    pair_target = F.pad(pair_target,(0,1)).flatten(1)
    pair_target /= pair_target.sum(-1,keepdim=True).clamp_min(1e-8)
    joint_logits = (out['pair']+out['subject'][:,:,None]).flatten(1)
    pair_loss = rank_loss(joint_logits,pair_target,active)
    loss = (subj_loss + .55*ref_loss + .20*null_loss + .50*pair_loss
            + 2.5*geo + .80*vote_loss + .60*point_loss+1.0*quality_loss)
    metrics = dict(subject=subj_loss, reference=ref_loss, pair=pair_loss,
                   support=point_loss, geometry=geo, votes=vote_loss, quality=quality_loss, unknown=null_loss)
    return loss, {key:value.detach() for key,value in metrics.items()}
