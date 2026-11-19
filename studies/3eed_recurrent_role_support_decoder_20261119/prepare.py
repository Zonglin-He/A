"""Seal independent source-only natural and uniquely labelled relational training examples."""
import argparse
from collections import Counter, defaultdict, deque
from functools import lru_cache
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image

from common import (PLATFORMS, PRIVATE, camera_view, canonical, frame, json_atomic,
                    make_lock, ordinary_rows, read, recording, sha, source_path, stable)
from camera_projection import CameraSpec, project_model_points
from independent_core_v2 import transform_gt

XLIMIT = .015
SYNTHETIC_CAP = {'train': 768, 'monitor': 192}


@lru_cache(maxsize=32)
def meta(path):
    return read(path)


def gt(raw, row):
    size = 7 if row['platform'] == 'waymo' else 9
    if raw is None or len(raw) < 7:
        return None
    raw = np.asarray((list(raw) + [0., 0.])[:size])
    if not np.isfinite(raw).all() or np.min(raw[3:6]) <= 0:
        return None
    return transform_gt(raw, row['platform'], camera_view(row))


def natural_rows(source):
    rows = []
    removed = Counter()
    allowed = ordinary_rows(source)
    paths = {Path(r[k]).resolve() for r in allowed for k in ('metapath', 'pointpath', 'imagepath')}
    for raw in read(source_path(source)):
        if not raw['positive'] or any(Path(raw[k]).resolve() in paths
                                      for k in ('metapath', 'pointpath', 'imagepath')):
            removed['split_path_or_not_positive'] += 1
            continue
        objects = meta(raw['metapath'])['ground_info']
        clip = canonical(raw['caption'])
        candidates = [i for i, a in enumerate(objects) if canonical(a['caption']) == clip]
        if len(candidates) != 1:
            removed['ambiguous_annotation_join'] += 1
            continue
        j = candidates[0]
        box = gt(objects[j]['bbox_3d'], raw)
        if box is None:
            removed['invalid_source_box'] += 1
            continue
        row = {k: raw[k] for k in ('platform', 'metapath', 'pointpath', 'imagepath', 'suffix')}
        row['caption'] = clip
        row['seed'] = int(stable('source-point-sample', frame(raw), raw['suffix'])[:8], 16)
        row['boxes'] = [box.tolist(), [0.] * 7]
        row['known_reference'] = False
        rows.append(row)
    return rows, dict(removed)


def splits(rows, source):
    counts = Counter(recording(r) for r in rows)
    if source == 'waymo':
        groups = sorted(counts, key=lambda g: stable(source, 'held-recording', g))
    else:
        groups = sorted(counts, key=lambda g: (counts[g], stable(source, 'held-recording', g)))
    n = max(1, round(.1 * len(groups)))
    assert len(groups) > n
    held_out = set(groups[:n])
    return {split: [r for r in rows if (recording(r) in held_out) == (split == 'monitor')]
            for split in ('train', 'monitor')}


def inventory(row, objects):
    entries = []
    for a in objects:
        entries.append((a['class'], a['bbox_3d']))
        entries.extend((b['class_other'], b['bbox_3d_other']) for b in a.get('others', []))
    seen = {}
    for kind, box in entries:
        mapped = gt(box, row)
        word = ' '.join(str(kind).casefold().replace('_', ' ').replace('-', ' ').split())
        if mapped is None or re.fullmatch(r'[a-z][a-z ]*', word) is None:
            continue
        key = tuple(np.round(mapped, 2))
        if key in seen:
            continue
        # Near-coincident copies across descriptions correspond to one annotation support.
        if any(np.linalg.norm(mapped[:3] - b[:3]) < .12 and
               np.max(np.abs(mapped[3:6] - b[3:6])) < .30 for _, b in seen.values()):
            continue
        seen[key] = word, mapped
    return list(seen.values())


def projected_intervals(items, row, fields):
    keys = ('image_intrinsic', 'image_extrinsic', 'camera_distortion')
    camera = CameraSpec.from_camera_fields(
        row['platform'], {k: fields[k] for k in keys if k in fields},
        waymo_view_suffix=camera_view(row), missing_distortion_policy='assume_zero_declared')
    with Image.open(row['imagepath']) as img:
        size = img.size
    centers = np.stack([b[:3] for _, b in items])
    boxes = np.stack([b for _, b in items])
    center_projection = project_model_points(centers, camera, size)
    mask = center_projection['in_image']
    unit = np.asarray([(x, y, z) for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])
    corners = unit[None] * boxes[:, None, 3:6]
    x, y = corners[..., 0].copy(), corners[..., 1].copy()
    c, s = np.cos(boxes[:, 6, None]), np.sin(boxes[:, 6, None])
    corners[..., 0], corners[..., 1] = c * x - s * y, s * x + c * y
    proj = project_model_points((corners + centers[:, None]).reshape(-1, 3), camera, size)
    u = proj['uv'].reshape(-1, 8, 2)[..., 0] / size[0]
    valid = np.isfinite(u).all(1) & mask
    return u.min(1), u.max(1), valid, center_projection['uv'][:, 0] / size[0]


def relation(row):
    fields = meta(row['metapath'])
    entries = inventory(row, fields['ground_info'])
    names = np.asarray([x[0] for x in entries])
    if len(names) < 3 or len(set(names)) < 2 or len(set(names)) == len(names):
        return None
    lo, hi, visible, u = projected_intervals(entries, row, fields)
    options = []
    for label in sorted(set(names[visible])):
        anchors = np.flatnonzero((names == label) & visible)
        anchors = anchors[np.argsort(u[anchors], kind='stable')]
        descriptions = []
        if len(anchors) == 1:
            descriptions.append((anchors[0], label))
        else:
            left, next_left = anchors[:2]
            next_right, right = anchors[-2:]
            if hi[left] + XLIMIT < lo[next_left]:
                descriptions.append((left, 'leftmost ' + label))
            if hi[next_right] + XLIMIT < lo[right]:
                descriptions.append((right, 'rightmost ' + label))
        for anchor, ref_phrase in descriptions:
            for kind in sorted(set(names[visible])):
                matched = np.flatnonzero((names == kind) & visible & (np.arange(len(names)) != anchor))
                if len(matched) < 2:
                    continue
                for side, sign in (('left', -1.), ('right', 1.)):
                    gap = sign * (u[matched] - u[anchor])
                    candidates = np.flatnonzero(gap > 0)
                    if candidates.size == 0:
                        continue
                    order = candidates[np.argsort(gap[candidates], kind='stable')]
                    best = order[0]
                    subject = matched[best]
                    separation = lo[anchor] - hi[subject] if sign < 0 else lo[subject] - hi[anchor]
                    if separation <= XLIMIT:
                        continue
                    if len(order) > 1 and gap[order[1]] - gap[best] <= XLIMIT:
                        continue
                    options.append((int(subject), int(anchor), side, ref_phrase))
    if not options:
        return None
    subject, anchor, side, reference = min(options, key=lambda x: stable('pair', frame(row), *x))
    lead = f'The {names[subject]}'
    link = f' closest horizontally to the {side} of '
    tail = ' in the image'
    phrase = lead + link + f'the {reference}' + tail
    return dict(row, caption=phrase, boxes=[entries[subject][1].tolist(), entries[anchor][1].tolist()],
                known_reference=True, spans=[[0, len(lead)], [len(lead) + len(link), len(phrase)-len(tail)],
                                             [len(lead), len(lead) + len(link)]],
                side=-1 if side == 'left' else 1)


def make_relations(rows, cap):
    frames = {}
    for r in rows:
        frames.setdefault((frame(r), camera_view(r)), r)
    by_recording = defaultdict(list)
    for key, r in frames.items():
        by_recording[recording(r)].append((stable('relation', *key), r))
    pending = {g: deque(r for _, r in sorted(seq, key=lambda x: x[0]))
               for g, seq in by_recording.items()}
    result, inspected = [], 0
    while pending and len(result) < cap:
        for g in sorted(pending, key=lambda key: stable('buckets', key)):
            r = pending[g].popleft()
            if not pending[g]:
                del pending[g]
            inspected += 1
            candidate = relation(r)
            if candidate is not None:
                result.append(candidate)
            if len(result) >= cap:
                break
    return result, inspected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sources', choices=PLATFORMS, nargs='+')
    args = parser.parse_args()
    PRIVATE.mkdir(exist_ok=True, mode=0o700)
    make_lock()
    for source in args.sources:
        start = time.perf_counter()
        folder = PRIVATE / source
        folder.mkdir(exist_ok=True, mode=0o700)
        receipt = folder / 'ROSTER_SEAL.json'
        if receipt.is_file():
            for name, digest in read(receipt)['rosters'].items():
                assert sha(folder / name) == digest
            continue
        rows, rejected = natural_rows(source)
        overview, hashes = {}, {}
        for split, entries in splits(rows, source).items():
            additions, inspected = make_relations(entries, SYNTHETIC_CAP[split])
            out = folder / f'{split}_PRIVATE.json'
            rows_all = entries + additions
            json_atomic(out, rows_all)
            hashes[out.name] = sha(out)
            overview[split] = {'natural': len(entries), 'relations': len(additions),
                               'recordings': len({recording(r) for r in entries}),
                               'inspected_frames_for_synthesis': inspected,
                               'left': sum(r['side'] < 0 for r in additions)}
        json_atomic(receipt, {'overview': overview, 'removed': rejected, 'rosters': hashes,
                              'input_lock': sha(PRIVATE/'INPUT_LOCK.json')})
        print('SOURCE_ROSTER', source, overview, rejected, round(time.perf_counter()-start, 1), flush=True)


if __name__ == '__main__':
    main()
