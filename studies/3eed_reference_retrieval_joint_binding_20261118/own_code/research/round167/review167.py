"""Generate a local output-blind counterfactual panel, then score its adjudications."""
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
from pathlib import Path
import re
import textwrap
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scope167 import ARMS, HERE, ORDER, OUT, PLATFORMS, WORK, atomic_json, read, roster, sha
from audit167 import blocks, row_edges
from geometry162 import CameraX
from labels166 import RIGHT, joint_status, label_pair, occupied_owners, physical_labels, truth_status
from score165 import many_iou, write_table
from visual162 import LINES, corners, projection

PANEL = WORK / 'predicate_panel_PRIVATE'
REGISTRY = PANEL / 'REGISTRY_PRIVATE.json'
TRUTH = PANEL / 'ADJUDICATED_PRIVATE.json'
REVIEW_PLAN = HERE / 'experiment/EVALUATION.json'


def camera(old, k, edges, *, image=None):
    h = image.height if image is not None else 0
    return SimpleNamespace(edges=edges, role_spans=old['spans'][k], width=int(old['width'][k]), height=h,
                           camera_intrinsic=old['camera_intrinsic'][k], model_to_camera=old['model_to_camera'][k],
                           distortion=old['distortion'][k])


def catalog(target):
    first_source = next(s for s, t in ORDER if t == target)
    out = []
    for _, data, _, old in blocks('ordinary', first_source, target):
        for k, j in enumerate(data['ordinals']):
            if not data['flip'][k]:
                continue
            j = int(j)
            row = roster()[target][j]
            edges, ((role, sign),) = row_edges(old, k)
            cam = camera(old, k, edges)
            if min(cam.role_spans[0, 0], cam.role_spans[role, 0]) < 0:
                continue
            names, boxes, _, _ = physical_labels(row, old['seeds'][k])
            _, ref = label_pair(row, cam, 0, role, sign, names, boxes)
            opposite = []
            if ref > 0:
                proj = CameraX(cam.camera_intrinsic, cam.model_to_camera, cam.distortion, cam.width)
                x = proj.project(boxes[ref, :3])[0]
                for i, box in enumerate(boxes[1:], 1):
                    if i == ref or names[0] != names[i]:
                        continue
                    v = proj.project(box[:3])[0]
                    if 0 <= x <= 1 and 0 <= v <= 1 and sign * (v - x) > .03:
                        opposite.append(i)
            order = hashlib.sha256(f'R167 predicate review|{target}|{j}'.encode()).hexdigest()
            out.append(dict(target=target, source=first_source, ordinal=j, frame=str(Path(row['metapath']).resolve()),
                            rank=order, role=role, sign=sign, hit=bool(opposite)))
    return sorted(out, key=lambda r: r['rank'])


def choose_rows(items):
    chosen, frames = [], set()
    for hit, limit in ((True, 8), (False, 4)):
        count = 0
        for row in items:
            if row['hit'] != hit or row['frame'] in frames:
                continue
            chosen.append(row)
            frames.add(row['frame'])
            count += 1
            if count == limit:
                break
    return sorted(chosen, key=lambda r: r['rank'])


@lru_cache(maxsize=4)
def locate(source, target, ordinal):
    from scope166 import BLOCK
    for _, data, _, earlier in blocks('ordinary', source, target, f'{ordinal // BLOCK:04d}.npz'):
        idx = np.flatnonzero(data['ordinals'] == ordinal)
        if not len(idx):
            continue
        k, = idx
        return int(k), data, earlier
    raise ValueError('Missing registered panel ordinal')


def role_phrase(caption, span):
    a, b = map(int, span)
    return re.sub(r'^[\s,.;]+|[\s,.;]+$', '', caption[a:b])


def render(item, row, cam, names, boxes, clauses, rgb):
    rgb = rgb.copy()
    rgb.thumbnail((1200, 690))
    labels = rgb.copy()
    draw = ImageDraw.Draw(labels)
    font = ImageFont.truetype('DejaVuSans.ttf', 15)
    smaller = ImageFont.truetype('DejaVuSans.ttf', 12)
    scale = [rgb.width, rgb.height]
    occupied = []
    palette = ('#52ff8e', '#ffb463', '#50e1ff', '#ffcfee', '#eefd72', '#bcc1ff')
    for i, box in enumerate(boxes):
        color = palette[i % len(palette)]
        verts = projection(cam, corners(box)) * scale
        for a, b in LINES:
            if np.isfinite(verts[[a, b]]).all():
                draw.line([tuple(verts[a]), tuple(verts[b])], fill=color, width=2)
        px, py = projection(cam, box[None, :3])[0] * scale
        if not (0 <= px < rgb.width and 0 <= py < rgb.height):
            continue
        mark = str(i)
        lens = 12 + 8 * len(mark)
        options = [(px + a, py + b) for a, b in ((0, -21), (0, 2), (-30, -17), (27, -6), (-35, 17), (34, 21), (-22, -36), (29, -37))]
        for x, y in options:
            area = (x, y, x + lens, y + 17)
            if area[0] < 0 or area[2] >= rgb.width or area[1] < 0 or area[3] >= rgb.height:
                continue
            if any(area[0] < q[2] and area[2] > q[0] and area[1] < q[3] and area[3] > q[1] for q in occupied):
                continue
            break
        occupied.append(area)
        draw.line([(px, py), (x, y + 8)], fill=color, width=1)
        draw.rectangle(area, fill='#14171e', outline=color)
        draw.text((x + 5, y + 1), mark, font=smaller, fill=color)
    content = [f"Case {item['case']:02} / {item['target']} · point to the objects in both clauses", *clauses]
    content.extend(textwrap.wrap('Annotation inventory  ' + '   '.join(f'{i} {k}' for i, k in enumerate(names)), width=145))
    content.extend(textwrap.wrap('Source expression  ' + row['caption'], width=145))
    top = 17 + 22 * len(content)
    canvas = Image.new('RGB', (1200, top + rgb.height), '#fafafa')
    canvas.paste(labels, ((canvas.width - labels.width) // 2, top))
    text = ImageDraw.Draw(canvas)
    for i, line in enumerate(content):
        text.text((12, 8 + i * 22), line, fill='#21232b', font=font)
    canvas.save(PANEL / f"{item['case']:02}.jpg", quality=93)
    rgb.save(PANEL / f"{item['case']:02}_raw.jpg", quality=92)


def prepare():
    from score167 import eval_lock
    eval_lock()
    assert not REGISTRY.exists()
    PANEL.mkdir(parents=True, exist_ok=True, mode=0o700)
    cases, stats = [], {}
    for target in PLATFORMS:
        items = catalog(target)
        stats[target] = dict(eligible_physical_expressions=len(items), possible_by_gt_geometry=sum(x['hit'] for x in items))
        for item in choose_rows(items):
            row = roster()[target][item['ordinal']]
            k, _, old = locate(item['source'], target, item['ordinal'])
            edges, ((role, sign),) = row_edges(old, k)
            assert role == item['role'] and sign == item['sign']
            with Image.open(row['imagepath']) as f:
                rgb = f.convert('RGB')
            cam = camera(old, k, edges, image=rgb)
            names, boxes, _, _ = physical_labels(row, old['seeds'][k])
            words = [role_phrase(row['caption'], cam.role_spans[j]) for j in (0, role)]
            a, b = words
            clauses = [f"LEFT | {a} to the left of {b} in this image.", f"RIGHT | {a} to the right of {b} in this image."]
            entry = {key: item[key] for key in ('target', 'source', 'ordinal', 'role', 'sign', 'hit')}
            entry.update(case=len(cases) + 1, categories=names, clauses=clauses)
            render(entry, row, cam, names, boxes, clauses, rgb)
            cases.append(entry)
    atomic_json(REGISTRY, cases)
    digest = dict(n=len(cases), platform_catalog=stats, by_target=dict(Counter(x['target'] for x in cases)),
                  registry=sha(REGISTRY), selection_plan=sha(REVIEW_PLAN),
                  prediction_seal=sha(HERE / 'PREDICTION_SEAL.json'), model_outputs_hidden=True, FINAL_access=False)
    atomic_json(HERE / 'REVIEW_PANEL_SEAL.json', digest)
    print('R167 blind review prepared', len(cases), stats, flush=True)


def score():
    p = read(HERE / 'REVIEW_PANEL_SEAL.json')
    cases = read(REGISTRY)
    answers = read(TRUTH)
    assert sha(REGISTRY) == p['registry'] and sha(HERE / 'PREDICTION_SEAL.json') == p['prediction_seal']
    assert [r['case'] for r in cases] == [r['case'] for r in answers]
    rows = []
    for case, label in zip(cases, answers, strict=True):
        tar, ordinal, role, side = (case[k] for k in ('target', 'ordinal', 'role', 'sign'))
        row = roster()[tar][ordinal]
        for source, target in ORDER:
            if target != tar:
                continue
            k, data, old = locate(source, target, ordinal)
            edge, ((r, sign),) = row_edges(old, k)
            assert r == role and sign == side and data['flip'][k]
            names, gt, seeds, ids = physical_labels(row, old['seeds'][k])
            assert names == case['categories']
            original = label.get('left', -1) if side == 1 else label.get('right', -1)
            flipped = label.get('right', -1) if side == 1 else label.get('left', -1)
            ref = label.get('ref', -1)
            valid = label['status'] == 'verified_pair'
            if valid:
                assert original == 0 and 0 < flipped < len(names) and 0 < ref < len(names) and ref != flipped
                cam = camera(old, k, edge)
                projection_x = CameraX(cam.camera_intrinsic, cam.model_to_camera, cam.distortion, cam.width)
                xx = [projection_x.project(gt[i, :3])[0] for i in (original, ref, flipped)]
                assert all(0 <= x <= 1 for x in xx) and side * (xx[1] - xx[0]) > 0 and side * (xx[2] - xx[1]) > 0
            before = occupied_owners(data['boxes'][k], seeds, ids, len(names))
            after = occupied_owners(data['flip_boxes'][k], seeds, ids, len(names))
            for i, arm in enumerate(ARMS[1:], 1):
                info = dict(case=case['case'], source=source, target=target, arm=arm,
                            status=label['status'], valid_pair=int(valid),
                            selection_bin='geometry_opportunity' if case['hit'] else 'remaining',
                            subject_switched=int(data['winners'][k, i, 0] != data['flip_winners'][k, i, 0]))
                out = dict.fromkeys(('original_subject', 'flipped_subject', 'original_reference', 'flipped_reference',
                                     'original_both', 'flipped_both', 'both_subjects', 'both_endpoints_twice',
                                     'original_iou', 'flipped_iou'), '')
                if valid:
                    os = truth_status(before[i, 0], original)
                    rs = truth_status(before[i, role], ref)
                    fs = truth_status(after[i, 0], flipped)
                    rf = truth_status(after[i, role], ref)
                    out.update(original_subject=os, flipped_subject=fs, original_reference=rs, flipped_reference=rf,
                               original_both=joint_status(os, rs), flipped_both=joint_status(fs, rf),
                               both_subjects=int(os == fs == RIGHT), both_endpoints_twice=int(os == fs == rs == rf == RIGHT),
                               original_iou=float(many_iou(data['boxes'][k, i, :1], gt[original])[0]),
                               flipped_iou=float(many_iou(data['flip_boxes'][k, i, :1], gt[flipped])[0]))
                rows.append(dict(**info, **out))
    path = OUT / 'counterfactual_visual.csv'
    write_table(path, rows)
    m = []
    for arm in ARMS[1:]:
        for target in ('all', *PLATFORMS):
            r = [x for x in rows if x['arm'] == arm and (target == 'all' or x['target'] == target)]
            good = [x for x in r if x['valid_pair']]
            m.append(dict(arm=arm, target=target, all_attempted=len(r), valid_pairs=len(good),
                          original_subject=sum(x['original_subject'] == RIGHT for x in good),
                          flipped_subject=sum(x['flipped_subject'] == RIGHT for x in good),
                          both_subjects=sum(x['both_subjects'] for x in good),
                          both_endpoints_twice=sum(x['both_endpoints_twice'] for x in good),
                          original_acc25=sum(x['original_iou'] > .25 for x in good),
                          original_acc50=sum(x['original_iou'] > .50 for x in good),
                          flipped_acc25=sum(x['flipped_iou'] > .25 for x in good),
                          flipped_acc50=sum(x['flipped_iou'] > .50 for x in good)))
    aggregate = OUT / 'counterfactual_visual_summary.csv'
    write_table(aggregate, m)
    audit = dict(panels=len(cases), physical_statuses=dict(Counter(x['status'] for x in answers)),
                 reviewer='single output-blinded local image reader',
                 files={f.name: sha(f) for f in (path, aggregate)}, review_digest=sha(TRUTH), FINAL_access=False)
    atomic_json(HERE / 'VISUAL_REVIEW_AUDIT.json', audit)
    print('R167 blinded counterfactual scores', audit['physical_statuses'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=('prepare', 'score'))
    {'prepare': prepare, 'score': score}[p.parse_args().phase]()
