"""Text-only spatial roles and conservative source-inventory supervision."""

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from PIL import Image

from scope import read

NOUNS = {
    'car': {'car', 'vehicle', 'sedan', 'suv', 'van', 'pickup', 'automobile', 'taxi', 'coupe', 'minivan'},
    'pedestrian': {'pedestrian', 'person', 'man', 'woman', 'boy', 'girl', 'child', 'individual',
                   'walker', 'worker', 'passerby'},
    'truck': {'truck', 'lorry'},
    'bus': {'bus', 'shuttle', 'coach'},
    'cyclist': {'cyclist', 'biker', 'bike', 'bicycle', 'rider'},
    'othervehicle': {'tractor', 'trailer', 'machine', 'machinery'},
}
WORDS = {word: category for category, variants in NOUNS.items() for word in variants}
PLURALS = {'cars', 'vehicles', 'pedestrians', 'persons', 'people', 'men', 'women', 'children',
           'trucks', 'buses', 'cyclists', 'bikes', 'group', 'pair'}
CANONICAL = {'othervehicle': 'vehicle', **{x: x for x in NOUNS if x != 'othervehicle'}}
CORNERS = np.array([(i, j, k) for i in (-.5, .5) for j in (-.5, .5) for k in (-.5, .5)])
GRAMMAR = re.compile(
    r'^(?:(?:find|locate|identify|select)\s+)?(?:the\s+)?(?P<subject>[^.!?;,:]{1,55}?)'
    r'(?:\s+(?:is|stands|sits|parked|located|seen)){0,2}\s+(?:to\s+the\s+|on\s+the\s+)?'
    r'(?P<rel>left|right)(?:\s+side)?\s+of\s+(?:(?:the|a|an)\s+)?'
    r'(?P<ref>[^.!?;,:]{1,80}?)(?:\s+in\s+(?:this|the)\s+(?:image|frame|picture))?[.!?]?$', re.I)
RELATIONS = {
    -1:r'\bleft(?: side)? of\s+(?:(?:the|a|an)\s+)?$',
    1:r'\bright(?: side)? of\s+(?:(?:the|a|an)\s+)?$',
    -2:r'\babove\s+(?:(?:the|a|an)\s+)?$',
    2:r'\b(?:below|beneath)\s+(?:(?:the|a|an)\s+)?$',
    -3:r'\bcloser to (?:the )?camera than\s+(?:(?:the|a|an)\s+)?$',
    3:r'\b(?:farther|further) from (?:the )?camera than\s+(?:(?:the|a|an)\s+)?$',
    4:r'\b(?:beside|near|next to|close to|nearest to|closest to)\s+(?:(?:the|a|an)\s+)?$',
}


@dataclass
class Roles:
    subject: tuple
    reference: tuple = (0, 0)
    relation_span: tuple = (0, 0)
    relation: int = 0
    reference_class: str | None = None

    def mask(self, offsets, valid):
        tokens = (offsets[:, 1] > offsets[:, 0]) & valid
        spans = (self.subject, self.reference, self.relation_span)
        masks = np.stack([tokens & (offsets[:, 0] < hi) & (offsets[:, 1] > lo) & (hi > lo)
                          for lo, hi in spans])
        if not masks[0].any():
            masks[0] = tokens
        return masks


def nominal(phrase):
    words = re.findall('[a-z]+', phrase.lower())
    if set(words) & (PLURALS | {'two','three','several','both','its','their','own'}):
        return None
    categories = {WORDS[x] for x in words if x in WORDS}
    return next(iter(categories)) if len(categories) == 1 else None


def parse_roles(caption, graph):
    entities = graph['entities']
    subject = tuple(entities[0]['span']) if entities else (0, len(caption))
    edges = [edge for edge in graph['relations'] if 'T' in (edge['subject'], edge['reference'])]
    if len(edges) == 1:
        edge = edges[0]
        reverse = edge['subject'] != 'T'
        role = edge['subject'] if reverse else edge['reference']
        reference = next((entity for entity in entities if entity['role'] == role), None)
        if reference is not None and edge['predicate'] in {'LEFT', 'RIGHT'}:
            category = nominal(caption[slice(*reference['span'])])
            if category:
                side = -1 if edge['predicate'] == 'LEFT' else 1
                return Roles(subject, tuple(reference['span']), (0, 0),
                             -side if reverse else side, category)
    # Whole-expression grammar only. It cannot turn position headers or orientation into a reference.
    match = GRAMMAR.fullmatch(caption)
    if match and nominal(match['subject']) and nominal(match['ref']) and not re.search(
            r'\b(?:left|right|near|above|below|behind|facing|not)\b',match['ref'],re.I):
        side = -1 if match['rel'].lower() == 'left' else 1
        return Roles(match.span('subject'), match.span('ref'), match.span('rel'), side,
                     nominal(match['ref']))
    # A lexical relation is accepted only when the parsed subject and reference
    # are in the same clause and the marker immediately owns the reference.
    candidates = []
    for entity in entities[1:]:
        ref = tuple(entity['span'])
        category = nominal(caption[slice(*ref)])
        if not category or ref[0] <= subject[1]:
            continue
        between = caption[subject[1]:ref[0]]
        if len(between.split()) > 13 or re.search(r'[.!?;:]|\b(?:not|never|facing|pointing|their|its|own)\b',
                                                  between,re.I):
            continue
        for sign, expression in RELATIONS.items():
            link = re.search(expression,between,re.I)
            if link and not nominal(between[:link.start()]):
                candidates.append(Roles(subject,ref,(subject[1]+link.start(),ref[0]),sign,category))
    if len(candidates) == 1:
        return candidates[0]
    # Keep an independently read reference phrase for unresolved spatial constructions.
    if len(entities) > 1:
        ref = tuple(entities[1]['span'])
        category = nominal(caption[slice(*ref)])
        if category:
            return Roles(subject, ref, reference_class=category)
    return Roles(subject)


def inventory(row):
    from crossplatform.round7.camera_projection import WAYMO_ANGLES, raw_to_model_points

    meta = read(Path(row['metapath']))
    item = meta['ground_info'][row['object_index']]
    entries = [(item['class'], item['bbox_3d'])]
    entries.extend((obj['class_other'], obj['bbox_3d_other']) for obj in item.get('others', ()))
    entries.extend((obj['class'], obj['bbox_3d']) for obj in meta['ground_info'])
    classes, raw = [], []
    for label, value in entries:
        box = np.asarray(value[:7], np.float64)
        if box.shape != (7,) or not np.isfinite(box).all() or (box[3:6] <= 0).any():
            if not raw:
                raise ValueError('Invalid primary source box')
            continue
        if any(np.linalg.norm(box[:3] - neighbor[:3]) < .12 for neighbor in raw):
            continue
        raw.append(box)
        classes.append(label.lower())
    boxes = np.stack(raw)
    boxes[:, :3] = raw_to_model_points(boxes[:, :3], row['platform'], row['suffix'])
    if row['platform'] == 'waymo':
        boxes[:, 6] += np.deg2rad(WAYMO_ANGLES[row['suffix']])
    boxes[:, 6] = (boxes[:, 6] + np.pi / 2) % np.pi - np.pi / 2
    return boxes.astype(np.float32), np.asarray(classes)


def project_inventory(row, boxes):
    from crossplatform.round7.camera_projection import CameraSpec, project_model_points, read_camera_allowlist

    with Image.open(row['imagepath']) as image:
        size = image.size
    spec = CameraSpec.from_camera_fields(
        row['platform'], read_camera_allowlist(row['metapath']), waymo_view_suffix=row['suffix'],
        missing_distortion_policy='assume_zero_declared')
    corners = CORNERS[None] * boxes[:, None, 3:6]
    x, y = corners[..., 0].copy(), corners[..., 1].copy()
    c, s = np.cos(boxes[:, 6, None]), np.sin(boxes[:, 6, None])
    corners[..., 0], corners[..., 1] = c*x - s*y, s*x + c*y
    center = project_model_points(boxes[:, :3], spec, size)
    vertices = project_model_points((corners+boxes[:, None, :3]).reshape(-1, 3), spec, size)
    field = np.c_[vertices['uv'] / size, vertices['depth']].reshape(len(boxes),8,3)
    center_pos = np.c_[center['uv'] / size, center['depth']]
    return (center['in_image'] & np.isfinite(field).all((1,2)), center_pos,
            field.min(1), field.max(1))


def match_natural(role, classes, boxes, projection):
    if not role.relation or not role.reference_class:
        return -1
    visible, center, mins, maxs = projection
    if not visible[0]:
        return -1
    possibilities = visible & (classes == role.reference_class)
    possibilities[0] = False
    if role.relation == 4:
        distance = np.linalg.norm(boxes[:,:3]-boxes[0,:3],axis=-1)
        radius = np.maximum(2.,.75*(boxes[:,3]+boxes[0,3]))
        close = np.flatnonzero(possibilities & (distance < radius))
        if len(close) == 1 and (possibilities & (distance < radius+.5)).sum() == 1:
            return int(close[0])
        return -1
    axis = abs(role.relation)-1
    centers = center[:,axis]
    lo,hi = mins[:,axis],maxs[:,axis]
    direction = role.relation * (centers[0] - centers)
    gap = lo[0] - hi if role.relation > 0 else lo - hi[0]
    allowed = np.flatnonzero(possibilities & (direction > 0))
    if len(allowed) == 1 and gap[allowed[0]] > (.18 if axis==2 else .010):
        return int(allowed[0])
    return -1


def synthetic_pair(classes, projection, rank):
    visible, center, mins, maxs = projection
    horizontal,lo,hi = center[:,0],mins[:,0],maxs[:,0]
    options = []
    for ref in np.flatnonzero(visible):
        anchor_class = classes[ref]
        if anchor_class not in CANONICAL:
            continue
        same = np.flatnonzero((classes == anchor_class) & visible)
        if len(same) == 1:
            description = CANONICAL[anchor_class]
        else:
            order = same[np.argsort(horizontal[same], kind='stable')]
            if ref == order[0] and lo[order[1]] - hi[ref] > .01:
                description = 'leftmost ' + CANONICAL[anchor_class]
            elif ref == order[-1] and lo[ref] - hi[order[-2]] > .01:
                description = 'rightmost ' + CANONICAL[anchor_class]
            else:
                continue
        for category in set(classes[visible]) & CANONICAL.keys():
            same_category = (classes == category) & visible
            same_category[ref] = False
            # Make the relation necessary with at least one same-category distractor.
            if same_category.sum() < 2:
                continue
            for sign, word in ((-1, 'left'), (1, 'right')):
                delta = sign * (horizontal - horizontal[ref])
                candidates = np.flatnonzero(same_category & (delta > 0))
                order = candidates[np.argsort(delta[candidates], kind='stable')]
                if not len(order):
                    continue
                best = int(order[0])
                separated = lo[best] - hi[ref] if sign > 0 else lo[ref] - hi[best]
                if separated <= .01 or (len(order)>1 and delta[order[1]] - delta[best] <= .015):
                    continue
                prefix = ['Find the nearest ', 'Locate the nearest ', 'Identify the nearest '][rank % 3]
                ab = prefix + CANONICAL[category]
                caption = f'{ab} to the {word} of the {description} in the image.'
                options.append((caption, best, int(ref), sign))
    if not options:
        return None
    return sorted(options)[rank % len(options)]
