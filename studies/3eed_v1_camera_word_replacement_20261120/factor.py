"""Balanced binary word interventions and physical likelihood replacement."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image
from scipy.special import expit, logsumexp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from crossplatform.round7.camera_projection import CameraSpec, project_model_points, read_camera_allowlist

ARMS = ('v1', 'add', 'replace', 'balanced_only', 'reverse_camera', 'wrong_slot', 'camera_only')
FIELD = re.compile(r'\b(?:location|position)(?:\s+information)?\s*:\s*', re.I)
REL_FIELD = re.compile(r'\b(?:relationship(?:s)?|relation(?:s)?|surrounding environment)\b[^;:.]{0,35}:', re.I)
INTRODUCE = re.compile(r'\bthere\s+(?:is|are|stands?|sits?)\b|\b(?:beside|next to|surrounded by)\b', re.I)
UNIT = re.compile(r'[^.,;:!?]+')
TOKEN = re.compile(r"[a-z]+|[-/]", re.I)
HORIZONTAL = {'left', 'right'}
SPATIAL = set('a an the of in on to at from and near close slightly somewhat just far extreme upper lower bottom top middle center centre central left right leftmost rightmost hand side half corner edge part portion area position section frame view image picture screen field -'.split())
CAMERA = set('view frame image picture screen'.split())
MIDDLE = set('center centre middle'.split())
HEADING = re.compile(r'\b(?:fac(?:e|es|ing)|point(?:s|ing)?|head(?:s|ing|ed)?|turn(?:s|ing|ed)?|oriented|direction|wheel|arm|hand|side visible|viewer sees)\b', re.I)
VALID_LOCATION = re.compile(r'\b(?:located|positioned|situated|appears?|position|location|standing|stationary|sitting|parked)\b', re.I)
OTHER_INTRO = re.compile(r'\b(?:another|other|there|their|its)\b', re.I)


@dataclass(frozen=True)
class Word:
    sign: int
    flipped: str
    spans: tuple


def _form(caption, start, end, field):
    spans = []
    for chunk in UNIT.finditer(caption, start, end):
        segment = chunk.group()
        tokens = list(TOKEN.finditer(segment))
        lower = [m.group().lower() for m in tokens]
        for i, word in enumerate(lower):
            if word not in HORIZONTAL:
                continue
            lo, hi = i, i + 1
            while lo and lower[lo-1] in SPATIAL:
                lo -= 1
            while hi < len(tokens) and lower[hi] in SPATIAL:
                hi += 1
            run = set(lower[lo:hi])
            prefix = segment[:tokens[lo].start()]
            if HEADING.search(prefix) or OTHER_INTRO.search(prefix):
                continue
            # Reject directions attached to an object, a lane or a road.
            if hi < len(tokens) and lower[hi-1] in {'a','an','the','of','to'}:
                continue
            definite = bool(run & CAMERA or run & MIDDLE)
            if not (definite or (field and hi == len(tokens)) or
                    (VALID_LOCATION.search(prefix) and run & {'corner', 'edge'})):
                continue
            if not field and re.search(r'\bwith\s+(?:an?|another|the)\b',prefix,re.I) and not VALID_LOCATION.search(prefix):
                continue
            spans.append((chunk.start() + tokens[i].start(), chunk.start() + tokens[i].end(), word))
    return spans


def image_word(caption):
    field = FIELD.search(caption)
    if field:
        end = re.search(r'[;.!?]|\b(?:relationships?|relations?|orientation|motion)\b[^:.]{0,35}:', caption[field.end():], re.I)
        spans = _form(caption, field.end(), field.end() + end.start() if end else len(caption), True)
    else:
        # Allow a sentence that continues to describe the subject ("It is located ...").
        boundary = min([len(caption)] + [m.start() for pat in (INTRODUCE, REL_FIELD)
                                        if (m := pat.search(caption))])
        for match in re.finditer(r'[.!?]\s+(?!(?:it|the object|the subject)\b)', caption[:boundary], re.I):
            boundary = match.start()
            break
        spans = _form(caption, 0, boundary, False)
    distinct = {s[2] for s in spans}
    if len(distinct) != 1:
        return Word(0, caption, ())
    word, = distinct
    flipped = caption
    for start, end, _ in reversed(spans):
        other = 'right' if word == 'left' else 'left'
        original = caption[start:end]
        if original.isupper():
            other = other.upper()
        elif original.istitle():
            other = other.title()
        flipped = flipped[:start] + other + flipped[end:]
    return Word(1 if word == 'right' else -1, flipped, tuple((a,b) for a,b,_ in spans))


@lru_cache(maxsize=256)
def camera(platform, index, calibration, image):
    spec = CameraSpec.from_camera_fields(platform, read_camera_allowlist(calibration),
                                         waymo_view_suffix=index, missing_distortion_policy='assume_zero_declared')
    with Image.open(image) as photo:
        return spec, photo.size


def horizontal(centers, platform, index, calibration, image):
    spec, size = camera(platform, index, str(calibration), str(image))
    projected = project_model_points(np.asarray(centers), spec, size)
    valid = projected['in_image']
    return np.where(valid, 2 * projected['uv'][:, 0] / size[0] - 1, 0.), valid


def scores(base, alternate, coordinate, valid, sign, slope, permutation):
    """Preserve frozen mass on unsupported slots; permute only comparable slots."""
    base = np.asarray(base, dtype=np.float64)
    base = base - logsumexp(base)
    result = np.tile(base, (len(ARMS), 1))
    use = np.flatnonzero(valid)
    if not sign or len(use) < 2:
        return result
    factual = base[use] - logsumexp(base[use])
    other = alternate[use] - logsumexp(alternate[use])
    other = other.astype(np.float64)
    other -= logsumexp(other)
    probability = expit(sign * slope * coordinate[use])
    # Both signs must have finite log likelihood, also for separable calibration data.
    log_f = -np.logaddexp(0., -sign * slope * coordinate[use])
    log_rev = -np.logaddexp(0., sign * slope * coordinate[use])
    mixed = np.logaddexp(factual, other) - np.log(2)
    local_perm = np.argsort(np.asarray(permutation)[use], kind='stable')
    wrong = np.empty_like(other)
    wrong[local_perm] = other[np.roll(local_perm, 1)]
    assert np.isfinite(probability).all()
    alternatives = (factual + log_f, mixed + log_f, mixed, mixed + log_rev,
                    np.logaddexp(factual, wrong) - np.log(2) + log_f, log_f)
    mass = logsumexp(base[use])
    for i, values in enumerate(alternatives, 1):
        result[i, use] = values - logsumexp(values) + mass
    return result
