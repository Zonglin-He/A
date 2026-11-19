"""Frozen v1 experiments for role retrieval, instance binding, and box geometry."""

import numpy as np
import torch

from crossplatform.v1_decoder import JointDecoder, _coordinate


def distinct_entity_relation(roles, has_reference):
    if not has_reference:
        return False
    edges = [edge for edge in roles['relations'] if 'T' in (edge['subject'], edge['reference'])]
    if len(edges) != 1 or edges[0]['predicate'] not in {'LEFT', 'RIGHT'}:
        return False
    ref = edges[0]['reference'] if edges[0]['subject'] == 'T' else edges[0]['subject']
    entity = next((item for item in roles['entities'] if item['role'] == ref), None)
    regions = {'area', 'background', 'bottom', 'center', 'edge', 'foreground', 'frame',
               'ground', 'image', 'lane', 'middle', 'picture', 'region', 'road', 'scene',
               'screen', 'side', 'sky', 'top', 'view'}
    return bool(entity and entity['head'].lower() not in regions)


class CandidateGeometry:
    """Use the same uniquely sampled LiDAR points for all geometry comparisons."""

    def __init__(self, raw, sampled):
        from reader158 import overlaps

        self.raw = raw
        self.overlap = overlaps(raw)
        self.points = np.unique(sampled, axis=0)
        self.support = {}
        self.pca = {}

    def inside_axis_box(self, box):
        # Include a one-ULP search margin, then apply the original crop predicate.
        half = box[3:6] / 2 + 1e-7
        limit = np.nextafter(box[0] - half[0], np.float32(-np.inf))
        first = np.searchsorted(self.points[:, 0], limit, side='left')
        limit = np.nextafter(box[0] + half[0], np.float32(np.inf))
        last = np.searchsorted(self.points[:, 0], limit, side='right')
        contained = (np.abs(self.points[first:last] - box[:3]) <= half).all(axis=1)
        return first + np.flatnonzero(contained)

    def members(self, index):
        if index not in self.support:
            self.support[index] = self.inside_axis_box(self.raw[index])
        return self.support[index]

    def orient(self, index):
        from crossplatform.candidate_frontend import orient_box

        if index not in self.pca:
            self.pca[index] = orient_box(self.points[self.members(index)], self.raw[index])
        return self.pca[index]

    def same_support(self, a, b):
        if a == b or np.array_equal(self.raw[a], self.raw[b]):
            return True
        if self.overlap[a, b] < .55:
            return False
        first, second = self.members(a), self.members(b)
        if min(len(first), len(second)) < 8:
            return False
        shared = len(np.intersect1d(first, second, assume_unique=True))
        return shared / (len(first) + len(second) - shared) >= .70

    def admissible(self, subjects, references):
        usable_ref = [len(self.members(index)) >= 4 for index in references]
        return np.array([[ok and not self.same_support(i, j) for j, ok in zip(references, usable_ref)]
                         for i in subjects], dtype=bool)

    def refine(self, index):
        """Move a selected box toward supported sibling estimates within local bounds."""
        base = self.orient(index)
        if len(self.members(index)) < 8:
            return base.copy(), 0
        siblings = [int(other) for other in np.flatnonzero(self.overlap[index] >= .55)
                    if np.all(self.raw[other, 3:6] > 0) and self.same_support(index, other)]
        if len(siblings) < 3:
            return base.copy(), len(siblings)
        median = np.median(self.raw[siblings], axis=0)
        cos, sin = np.cos(base[6]), np.sin(base[6])
        rotation = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], dtype=np.float32)
        local_shift = (median[:3] - base[:3]) @ rotation
        new = base.copy()
        new[:3] += np.clip(local_shift, -.10 * base[3:6], .10 * base[3:6]) @ rotation.T
        new[3:6] = base[3:6] * np.clip(median[3:6] / base[3:6], .9, 1.1)
        # Check measured points already inside the oriented selected box.
        points = self.points[self.members(index)]
        support = points[(np.abs((points - base[:3]) @ rotation) <= base[3:6] / 2 + 1e-7).all(1)]
        if len(support) < 8:
            return base.copy(), len(siblings)
        retained = (np.abs((support - new[:3]) @ rotation) <= new[3:6] / 2 + 1e-7).all(1)
        if retained.mean() < .90:
            return base.copy(), len(siblings)
        return new, len(siblings)


def diverse_order(utility, overlap, count):
    order = np.argsort(-utility, kind='stable').tolist()
    selected = [order[0]]
    for candidate in order[1:]:
        if (overlap[candidate, selected] <= .35).all():
            selected.append(candidate)
        if len(selected) == count:
            break
    if len(selected) < count:
        selected.extend(index for index in order if index not in selected)
    return selected[:count]


def reference_order(boxes, score, count):
    """Extend the v1 reference inhibition rule beyond the first four entries."""
    centers = boxes[:, :3]
    rank = score.clone()
    selected = []
    for _ in range(count):
        index = rank.argmax()
        selected.append(int(index))
        offset = (centers - centers[index]) / (boxes[index, 3:].clamp_min(.03) * .55 + .4)
        rank = rank - 4. * torch.exp(-.5 * offset.square().sum(-1))
        rank[index] = -torch.inf
    return selected


def interleave(base, evidence, overlap):
    """Preserve rank 1 and give alternate role evidence two of four slots."""
    alternate = np.argsort(-evidence, kind='stable').tolist()
    selected = [base[0]]
    for order in (alternate, base, alternate):
        eligible = next((i for i in order if i not in selected and (overlap[i, selected] <= .35).all()), None)
        selected.append(eligible if eligible is not None else next(i for i in base if i not in selected))
    return selected


class ReadoutInputs:
    """Reuse the frozen native roles without changing the decoder's unary scores."""

    def __init__(self, frame, evidence, native, geometry):
        from grounding158 import select_grounding_box

        self.geometry = geometry
        self.evidence = evidence
        self.device = evidence['xyz'].device
        self.active = bool(evidence['has_reference'][0])
        _, self.utility = select_grounding_box(frame.boxes, frame.probability)
        score = (native['queries'] @ native['tokens'].transpose(1, 2)) / .07
        weights = evidence['role_mask'][0, :2].float()
        # Use v1's sum-before-division numerics for the original ref scores.
        per_role = (score[0, :, None, :] * weights[None]).sum(-1) / weights.sum(-1).clamp_min(1)
        self.subject_native = per_role[:, 0].cpu().numpy()
        self.ref_native = per_role[:, 1].contiguous() if self.active else torch.zeros_like(per_role[:, 1])
        self.original_subject = diverse_order(self.utility, geometry.overlap, 8)
        boxes = torch.as_tensor(frame.boxes, device=self.device)
        self.original_reference = reference_order(boxes, self.ref_native, 8)

    def select(self, mode):
        subject, ref = self.original_subject, self.original_reference
        if mode == 'wide8':
            return subject, ref
        if mode == 'v1':
            return subject[:4], ref[:4]
        s, r = self.subject_native, self.ref_native.cpu().numpy()
        if mode == 'role_swap':
            if not self.active:
                return subject[:4], ref[:4]
            s, r = r, s
        extra_ref = interleave(ref, r - s, self.geometry.overlap) if self.active else ref[:4]
        return interleave(subject, s, self.geometry.overlap), extra_ref

    def for_slots(self, subjects, references):
        prior = np.log(np.maximum(self.utility[subjects], 1e-80))
        base = (prior - prior[0]).clip(-20., 0.).astype(np.float32)
        subject_boxes = np.stack([self.geometry.orient(i) for i in subjects])
        ref_boxes = np.stack([self.geometry.orient(i) for i in references])
        return dict(self.evidence,
                    subject_boxes=torch.as_tensor(subject_boxes[None], device=self.device),
                    reference_boxes=torch.as_tensor(ref_boxes[None], device=self.device),
                    base_logit=torch.as_tensor(base[None], device=self.device),
                    ref_logit=self.ref_native[None, references])


class ReadoutTrial(JointDecoder):
    """Use the frozen v1 layers for wider slots or a physical pair validity mask."""

    def _pair_for(self, state, boxes, angle, pixels, mass, active, split):
        if split == self.slots and state.shape[1] == 2 * self.slots:
            return self._pair_scores(state, boxes, angle, pixels, mass, active)
        subject, reference = state[:, :split], state[:, split:]
        sbox, rbox = boxes[:, :split, None], boxes[:, None, split:]
        relative = _coordinate((sbox[..., :3] - rbox[..., :3]) /
                               ((sbox[..., 3:6] + rbox[..., 3:6]) * .5 + .25))
        ratio = sbox[..., 3:6].log() - rbox[..., 3:6].log()
        shared = mass[:, :split, None] * mass[:, None, split:]
        confidence = shared.clamp_min(1e-8).sqrt() * (shared > 0)
        delta = pixels[:, :split, None] - pixels[:, None, split:]
        view = torch.cat((4 * delta * confidence[..., None], confidence[..., None]), -1)
        yaw = angle[:, :split, None] - angle[:, None, split:]
        a = subject[:, :, None].expand(-1, -1, reference.shape[1], -1)
        b = reference[:, None].expand(-1, split, -1, -1)
        return self.pair(torch.cat((a, b, relative, ratio, view, yaw), -1)).squeeze(-1) * active[:, None, None]

    def trial(self, evidence, admissible=None):
        subjects, references = evidence['subject_boxes'], evidence['reference_boxes']
        n, k = subjects.shape[1], references.shape[1]
        if admissible is None and n == k == self.slots:
            return self(evidence)['choices'][0]
        anchors = torch.cat((subjects, references), 1)
        points, uv, visible = evidence['xyz'], evidence['uv'].float(), evidence['visible']
        active = evidence['has_reference']
        scene, role = self._scene_and_roles(evidence)
        ref = evidence['ref_logit']
        ref_prior = .22 * (ref - ref.amax(-1, keepdim=True)).clamp_min(-15)
        prior = torch.cat((evidence['base_logit'], ref_prior), 1)
        language = torch.cat((role[:, :1].expand(-1, n, -1), role[:, 1:].expand(-1, k, -1)), 1)
        pose = torch.cat((_coordinate(anchors[..., :3]), anchors[..., 3:6].log()), -1)
        device = points.device
        indices = [torch.arange(count, device=device).clamp_max(self.slots - 1) for count in (n, k)]
        embedding = torch.cat((self.role_slot[0, indices[0]], self.role_slot[1, indices[1]]), 0)
        state = language + self.anchor(pose) + embedding[None]
        key, value = self.key(scene), self.value(scene)
        bias = self._spatial_bias(points, anchors)
        static = .65 * bias
        angle = torch.stack(((2 * anchors[..., 6]).cos(), (2 * anchors[..., 6]).sin()), -1)
        state, unaries, pixels, mass = self._read(
            state, language, key, value, static, points, uv, visible, anchors, prior)
        pair = self._pair_for(state, anchors, angle, pixels, mass, active, n)
        subject, reference = state[:, :n], state[:, n:]
        ref_weights = (pair + unaries[:, None, n:]).softmax(-1)
        subject_weights = (pair.transpose(1, 2) + unaries[:, None, :n]).softmax(-1)
        if admissible is not None:
            # Missing pair mass contributes zero message and is never redistributed.
            ref_weights = ref_weights * admissible
            subject_weights = subject_weights * admissible.transpose(1, 2)
        gate = self.relation_gate.sigmoid() * active[:, None, None]
        state = torch.cat((subject + gate * (ref_weights @ self.from_ref(reference)),
                           reference + gate * (subject_weights @ self.from_subject(subject))), 1)
        feedback = torch.arange(n + k, device=device) < n
        spatial = static + self.spatial_gate.sigmoid() * bias * feedback[None, :, None]
        state, unaries, pixels, mass = self._read(
            state, language, key, value, spatial, points, uv, visible, anchors, prior)
        pair = self._pair_for(state, anchors, angle, pixels, mass, active, n)
        subject, reference = unaries[:, :n], unaries[:, n:].log_softmax(-1)
        relation = .5 * pair.float().tanh()
        if admissible is not None:
            relation = relation * admissible
        joint = subject[:, :, None] + reference[:, None, :] + relation
        scores = torch.where(active[:, None], joint.logsumexp(-1), subject)
        row = torch.arange(len(scores), device=device)
        selected = scores.argmax(-1)
        chosen_ref = joint[row, selected]
        if admissible is not None:
            valid = admissible[row, selected]
            unknown = chosen_ref.masked_fill(valid, -torch.inf).logsumexp(-1)
            valid_score = chosen_ref.masked_fill(~valid, -torch.inf)
            ref_index = torch.where(unknown >= valid_score.amax(-1), -1, valid_score.argmax(-1))
        else:
            ref_index = chosen_ref.argmax(-1)
        return torch.stack((selected, ref_index), -1)[0]
