"""A small output-blind suite of constructed, same-image left/right expressions."""
import argparse
from collections import Counter
from contextlib import ExitStack
import fcntl
import gc
import re
from types import SimpleNamespace

import numpy as np

from scope167 import ARMS, HERE, ORDER, OUT, PLATFORMS, ROOT, WORK, atomic_json, lock, read, roster, sha, status
from retrieval167 import repaired
from review167 import REGISTRY, locate, camera
from evidence162 import EvidenceGrounder
from geometry162 import CameraX
from labels166 import RIGHT, joint_status, occupied_owners, physical_labels, truth_status
from predict165 import points_for
from score165 import many_iou, sensor_values, write_table
from common160 import npz_atomic

DIR = WORK / 'controlled_pairs_PRIVATE'
SPEC = HERE / 'experiment/CONTROLLED_PAIRS.json'
SENTENCE = re.compile(r'^Find (.+?) to the (left|right) of (.+?) in the camera image\.$')


def parse(text):
    match = SENTENCE.fullmatch(text)
    assert match is not None
    return match[1], match[3], (1 if match[2] == 'left' else -1)


def frozen_receipt():
    base = read(HERE / 'REVIEW_PANEL_SEAL.json')
    assert sha(REGISTRY) == base['registry'] and sha(HERE / 'PREDICTION_SEAL.json') == base['prediction_seal']
    return dict(clauses=sha(DIR / 'QUERIES.json'), answers=sha(DIR / 'ANSWERS.json'),
                spec=sha(SPEC), program=sha(HERE / 'challenge167.py'), final_access=False)


def evidence(model, row, a, b):
    from core89 import route89, tensor_route89
    from grounding158 import select_grounding_box
    from interface66 import capture
    from model51 import SENTINEL
    from model76 import tensor_data, tensor_prefix
    from prepare76 import geometry
    from crossplatform.ground_single_frame import orient_box

    phrase = f'{a} relative to {b} in the camera image.'
    points, ids = points_for(row)
    native, _ = capture(model.base, points, ids, phrase)
    shape = model.base.tokenizer(phrase + SENTINEL, return_offsets_mapping=True)
    offsets = np.asarray(shape['offset_mapping'])
    assert offsets.shape[0] == native['tokens'].shape[0]
    lo = phrase.index(b, len(a))
    spans = [(0, len(a)), (lo, lo + len(b))]
    roles = np.stack([(offsets[:, 1] > offsets[:, 0]) & (offsets[:, 1] > l) & (offsets[:, 0] < h) for l, h in spans])
    assert roles.any(1).all() and not (roles[0] & roles[1]).any()
    graph = {'entities': [dict(role=k, span=list(s)) for k, s in zip(('T', 'R1'), spans, strict=True)], 'relations': []}
    modified = dict(row, caption=phrase)
    physical, _ = geometry(modified, native)
    prefix, pixels = tensor_prefix(native), tensor_data(physical, modified)
    m = model.matcher
    seen = []
    layer = model.base.cross_encoder.layers[-1].cross_layer.cross_lv
    assert m.route is None and not m.use_interface
    with ExitStack() as hooks:
        hooks.callback(layer.register_forward_hook(lambda _m, _a, out: seen.append(out[1][0].cpu().numpy())).remove)
        probability, _, _ = m.main(model.base, prefix, pixels, 'bidirectional')
    attention, = seen
    route, _ = route89(modified, dict(native, frozen_probability=probability.cpu().numpy()), physical,
                       graph, roles, attention, offsets, {})
    m.route = tensor_route89(route)
    m.use_interface = True
    try:
        p, _, _ = m.main(model.base, prefix, pixels, 'bidirectional')
    finally:
        m.route = None
        m.use_interface = False
        m.local = {}
    boxes = native['boxes']
    winner, utility = select_grounding_box(boxes, p.cpu().numpy())
    ev = SimpleNamespace(boxes=boxes, points=points, seeds=native['xyz'][0], old_winner=winner,
                         old_pca=orient_box(points, boxes[winner]), utility=utility,
                         role_mask=roles, affinity=(roles @ attention) / roles.sum(1)[:, None],
                         visible=physical['visible'], uv=physical['uv_grid'] / [48, 32],
                         edges=np.array([[0, 1, 1]], dtype=np.int16))
    return ev, native['query'], native['tokens'], route['hypotheses_PRIVATE']


def predict():
    import torch
    from model51 import fingerprint

    lock()
    receipt = frozen_receipt()
    assert not (HERE / 'CONTROLLED_PREDICTION_SEAL.json').exists()
    atomic_json(HERE / 'CONTROLLED_LOCK.json', receipt)
    registry = {r['case']: r for r in read(REGISTRY)}
    queries = read(DIR / 'QUERIES.json')
    assert len(queries) == 5 and len({r['case'] for r in queries}) == 5
    status('counterfactual_review', controlled_queries=5)
    saved = {}
    with (ROOT / 'research/GPU_RESOURCE.lock').open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(1)
        for source in PLATFORMS:
            model = EvidenceGrounder(source)
            state = fingerprint(model.base), fingerprint(model.matcher)
            for q in queries:
                target = registry[q['case']]['target']
                if target == source:
                    continue
                ordinal = registry[q['case']]['ordinal']
                row = roster()[target][ordinal]
                s, r, l = parse(q['left'])
                s2, r2, rr = parse(q['right'])
                assert (s, r, l, rr) == (s2, r2, 1, -1)
                k, _, old = locate(source, target, ordinal)
                with torch.inference_mode():
                    ev, queries3d, tokens, route = evidence(model, row, s, r)
                prior = {key: old[key][k] for key in ('starts', 'budget')}
                prior['routes'] = route
                result = repaired(ev, prior, queries3d, tokens)
                assert result['flip']
                name = f"{q['case']:02d}_{source}_to_{target}.npz"
                dest = DIR / name
                npz_atomic(dest, **result, seeds=ev.seeds, arms=np.asarray(ARMS))
                saved[name] = sha(dest)
            assert state == (fingerprint(model.base), fingerprint(model.matcher))
            del model
            gc.collect()
            torch.cuda.empty_cache()
    assert len(saved) == 10
    atomic_json(HERE / 'CONTROLLED_PREDICTION_SEAL.json', dict(n=10, files=saved,
                design=sha(HERE / 'CONTROLLED_LOCK.json'), both_clauses_parsed=True,
                shared_neutral_encoding=True, unchanged_model_fingerprints=True, FINAL_access=False))
    print('R167 controlled scene pairs sealed', len(saved), flush=True)


def score():
    p = read(HERE / 'CONTROLLED_PREDICTION_SEAL.json')
    assert p['n'] == 10 and not p['FINAL_access'] and read(HERE / 'CONTROLLED_LOCK.json') == frozen_receipt()
    assert sha(HERE / 'CONTROLLED_LOCK.json') == p['design']
    registry = {r['case']: r for r in read(REGISTRY)}
    out = []
    for q in read(DIR / 'ANSWERS.json'):
        case = q['case']
        target = registry[case]['target']
        ordinal = registry[case]['ordinal']
        row = roster()[target][ordinal]
        for source, dest in ORDER:
            if dest != target:
                continue
            name = f'{case:02d}_{source}_to_{target}.npz'
            path = DIR / name
            assert sha(path) == p['files'][name]
            k, _, previous = locate(source, target, ordinal)
            cam0 = camera(previous, k, ())
            cam = CameraX(cam0.camera_intrinsic, cam0.model_to_camera, cam0.distortion, cam0.width)
            with np.load(path, allow_pickle=False) as f:
                d = dict(f)
            _, gt, pts, ids = physical_labels(row, d['seeds'])
            x = [cam.project(gt[q[key], :3])[0] for key in ('left', 'ref', 'right')]
            assert 0 <= x[0] < x[1] < x[2] <= 1
            seeds = [occupied_owners(d[k], pts, ids, len(gt)) for k in ('boxes', 'flip_boxes')]
            for i, arm in enumerate(ARMS[1:], 1):
                info = dict(panel_case=case, source=source, target=target, arm=arm)
                states = []
                for side, key, owners in zip(('left', 'right'), ('boxes', 'flip_boxes'), seeds, strict=True):
                    box = d[key][i, :1]
                    iou = float(many_iou(box, gt[q[side]])[0])
                    audit_iou, = sensor_values(box, gt[q[side]], target, int(row['suffix'] or 0))
                    assert abs(iou - audit_iou) < 1e-7 and all((iou > t) == (audit_iou > t) for t in (.25, .5))
                    subject, ref = truth_status(owners[i, 0], q[side]), truth_status(owners[i, 1], q['ref'])
                    states.append((subject, ref))
                    info.update({f'{side}_iou': iou, f'{side}_subject': subject, f'{side}_ref': ref,
                                 f'{side}_both': joint_status(subject, ref)})
                info.update(both_subjects=int(states[0][0] == states[1][0] == RIGHT),
                            both_endpoints_twice=int(all(x == RIGHT for s in states for x in s)))
                out.append(info)
    assert len(out) == 40
    path = OUT / 'controlled_counterfactual.csv'
    write_table(path, out)
    summary = []
    for arm in ARMS[1:]:
        cohort = [r for r in out if r['arm'] == arm]
        summary.append(dict(arm=arm, n=len(cohort), **{key: sum(r[key] == RIGHT for r in cohort) for key in
                ('left_subject','left_ref','left_both','right_subject','right_ref','right_both','both_subjects','both_endpoints_twice')}))
    tab = OUT / 'controlled_counterfactual_summary.csv'
    write_table(tab, summary)
    atomic_json(HERE / 'CONTROLLED_EVALUATION_AUDIT.json', dict(physical_pairs=5, direction_pairs=10, rows=40,
                both_signs_sensor_iou_audited=True, files={x.name: sha(x) for x in (path, tab)}, FINAL_access=False))
    print('R167 controlled pairs scored', 10, flush=True)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('stage', choices=('predict', 'score'))
    {'predict': predict, 'score': score}[args.parse_args().stage]()
