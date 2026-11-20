"""Exchange disjoint congruent LiDAR crops. All coordinates are in the original model frame."""
from dataclasses import dataclass
import numpy as np


def half_extents(boxes):
    b = np.atleast_2d(boxes)
    c, s = abs(np.cos(b[:, 6])), abs(np.sin(b[:, 6]))
    return .5*np.column_stack((c*b[:, 3]+s*b[:, 4], s*b[:, 3]+c*b[:, 4], b[:, 5]))


@dataclass
class Exchange:
    centers: np.ndarray
    half: np.ndarray

    def masks(self, points):
        return np.stack([np.all(abs(points-c) <= self.half, axis=1) for c in self.centers])

    def move(self, points):
        masks = self.masks(points)
        assert not (masks[0] & masks[1]).any()
        out = points.copy()
        displacement = self.centers[1] - self.centers[0]
        out[masks[0]] += displacement
        out[masks[1]] -= displacement
        return out

    def stationary(self, boxes):
        # Conservative: an enclosing axis-aligned box touching either cell is excluded.
        radii = half_extents(boxes) + self.half
        return np.logical_and.reduce([np.any(abs(boxes[:, :3]-c) > radii, axis=1) for c in self.centers])


def select_exchange(boxes, score, returns):
    valid = np.flatnonzero(returns >= 4)
    if len(valid) < 2:
        return None
    ordered = valid[np.argsort(-score[valid], kind='stable')]
    top = ordered[0]
    extents = half_extents(boxes)
    for other in ordered[1:]:
        half = np.maximum(extents[top], extents[other])
        centers = boxes[[top, other], :3].copy()
        if np.any(abs(centers[1] - centers[0]) > 2*half):
            return np.array([top, other]), Exchange(centers, half)
    return None


def neutral_text(text, graph):
    roles = {e['role']: e for e in graph['entities']}
    others = {e['reference'] if e['subject']=='T' else e['subject'] for e in graph['relations']
              if e['predicate'] in ('LEFT','RIGHT') and ('T' in (e['reference'],e['subject']))}
    if len(others) != 1:
        return None
    ref = roles[next(iter(others))]
    lo, hi = ref['span']
    sl, sh = roles['T']['span']
    if not (0 <= lo < hi <= len(text)) or max(lo, sl) < min(hi, sh):
        return None
    phrase = text[lo:hi]
    if ref['head'].lower() not in phrase.lower() or phrase.lower().strip() in ('the object', 'object'):
        return None
    return text[:lo]+('The object' if lo==0 else 'the object')+text[hi:]
