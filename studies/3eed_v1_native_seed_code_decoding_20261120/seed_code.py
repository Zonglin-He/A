"""Read frozen whole-seed logits on the training-label sets implied by K candidate boxes."""
from collections import defaultdict
from hashlib import sha256

import numpy as np
import torch

ARMS = ('v1', 'frozen_joint', 'birth', 'box_max', 'native_logit', 'native_focal',
        'top_logit', 'top_focal', 'native_logit_abstain', 'native_focal_abstain',
        'top_logit_abstain', 'top_focal_abstain', 'mismatch', 'mismatch_abstain')
FIELDS = ('frozen_joint','birth','box_max','native_logit','native_focal','top_logit','top_focal','mismatch')


def focal_costs(logits):
    z = np.asarray(logits, dtype=np.float64)
    log_pos = -np.logaddexp(0., -z)
    log_neg = -np.logaddexp(0., z)
    pos = -.25 * np.exp(2. * log_neg) * log_pos
    neg = -.75 * np.exp(2. * log_pos) * log_neg
    return pos, neg


def native_codes(seed_xyz, boxes, inside, device):
    """Match the single-valid-GT slice and ties of the released KPS cost."""
    xyz = torch.as_tensor(seed_xyz, device=device, dtype=torch.float32)
    b = torch.as_tensor(boxes, device=device, dtype=torch.float32)
    member = torch.as_tensor(inside, device=device)
    norm = (xyz[None] - b[:,None,:3]) / (b[:,None,3:6]+1e-6)
    distances = torch.sqrt((norm ** 2).sum(dim=-1) + 1e-6)
    costs = torch.where(member, distances, 100.)
    ids = torch.topk(costs.contiguous(), 4, largest=False, dim=-1).indices
    chosen = torch.zeros_like(member).scatter_(1,ids,True) & member
    return chosen.cpu().numpy()


def mask_in_label_frame(points, boxes):
    """points are model coordinates before Waymo front-view rotation (Drone z already shifted)."""
    from utils.box_util import extract_points_in_bbox_3d
    return np.stack([extract_points_in_bbox_3d(points,b)[1] for b in boxes])


def label_frame_boxes(boxes, platform, view):
    if platform != 'waymo':
        return np.asarray(boxes, dtype=np.float64)
    from crossplatform.round7.camera_projection import WAYMO_ANGLES, z_rotation
    angle = WAYMO_ANGLES[int(view)]
    b = np.asarray(boxes, dtype=np.float64).copy()
    b[:,:3] = b[:,:3] @ z_rotation(angle)
    b[:,6] -= np.deg2rad(angle)
    return b


def key(*parts):
    return sha256('|'.join(map(str,('seed-code-v1',*parts))).encode()).hexdigest()


def code_identity(mask):
    return tuple(np.flatnonzero(mask).tolist())


def matched_permutation(codes, platform, ordinal):
    buckets = defaultdict(list)
    instances = defaultdict(list)
    for slot, mask in enumerate(codes):
        instances[code_identity(mask)].append(slot)
    for identity, slots in instances.items():
        buckets[len(identity),len(slots)].append(slots)
    rng = np.random.default_rng(int(key('mismatch',platform,ordinal)[:16],16))
    take = np.arange(len(codes))
    for bucket in buckets.values():
        if len(bucket) < 2:
            continue
        cycle = rng.permutation(len(bucket))
        for a,b in zip(cycle,np.roll(cycle,1)):
            take[bucket[a]] = bucket[b]
    return take


def pick(values,joint):
    return int(np.lexsort((np.arange(len(values)), -joint, -values))[0])


def readout(codes, inside, logits, birth, joint, v1, platform, ordinal):
    counts = inside.sum(1).clip(max=4)
    # Fixed membership, independent of the native distance selection.
    top = np.zeros_like(codes)
    score_max = np.full(len(inside), -np.inf)
    for i,n in enumerate(counts):
        if n:
            eligible = np.flatnonzero(inside[i])
            ids = eligible[np.argsort(-logits[eligible],kind='stable')[:n]]
            top[i,ids] = True
            score_max[i] = logits[ids[0]]
    pos,neg = focal_costs(logits)
    fields = dict(frozen_joint=joint.astype(float),birth=logits[birth].astype(float),box_max=score_max)
    for label, masks in (('native',codes),('top',top)):
        for mode, field in (('logit',logits.astype(float)), ('focal',neg-pos)):
            fields[label+'_'+mode] = np.array([field[mask].sum() for mask in masks])
    perm = matched_permutation(codes,platform,ordinal)
    fields['mismatch'] = fields['native_focal'][perm]
    frozen = int(np.argmax(joint))
    selections = [v1, frozen, pick(fields['birth'], joint), pick(score_max,joint)]
    families = ('native_logit','native_focal','top_logit','top_focal')
    for abstain in (False, True):
        for arm in families:
            f = fields[arm]
            selections.append(frozen if abstain and f.max()<=0 else pick(f,joint))
    wrong = fields['mismatch']
    selections += [pick(wrong,joint),frozen if wrong.max()<=0 else pick(wrong,joint)]
    return dict(choices=np.asarray(selections,np.int16),fields=np.stack([fields[k] for k in FIELDS]),
                top=top,permutation=perm)
