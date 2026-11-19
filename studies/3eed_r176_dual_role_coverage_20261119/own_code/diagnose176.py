"""Locate the first failed candidate stage after R175 predictions have been sealed."""
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import sys

import numpy as np
import torch

from scope176 import (FACTOR, N, ORDER, ORIGINAL, PLATFORMS, PRED_ARMS, PRIOR,
                      PRIVATE, R165_DATA, ROOT, SPEC, array_sha, camera_view, frame,
                      input_lock, json_atomic, locked, npz_atomic, ordinary_rows,
                      points_for, read, recording, sha, state)
from score175 import original
from roles175 import roles_batch
from train175 import load_split, select
from model175 import RelationalDecoder
from score import labels
from batch_yaw_iou import corners, many_iou
from crossplatform.ground_single_frame import orient_box

STAGES = ('A256_no_pair', 'four_slots_no_pair', 'output_miss', 'success')
ARMS = ('pair_only', 'full')
ARM_IDS = [PRED_ARMS.index(arm) for arm in ARMS]
DUAL_CUTS = np.array(read(SPEC)['diagnosis']['dual_cuts'])
SUBJECT_CUTS = np.array(read(SPEC)['diagnosis']['subject_cuts'])
# The historical box evaluator uses tolerant polygon-membership tests and the
# sealed boxes store float32 dimensions. Allow a margin on both pruning and audit.
BOUND_TOLERANCE = 1e-6


def reviewed_references():
    sys.path.append(str(FACTOR))
    from reference_truth import library, phrase_key, truth_pins, verified_refs

    panel = sorted(verified_refs().items())
    rows = [ordinary_rows(t)[i] for (t, i), _ in panel]
    contexts = roles_batch(rows)
    _, inventory = library()
    result = {s: {} for s in PLATFORMS}
    for ((target, ordinal), candidates), row, context in zip(panel, rows, contexts, strict=True):
        lo, hi = context['spans'][1]
        key = phrase_key(row['caption'][lo:hi])
        options = {(j, n) for phrase, j, n, _ in candidates if key and phrase == key}
        item = dict(status='reference_unmatched', relation=context['relation'])
        if len(options) == 1:
            ref, n = options.pop()
            names, boxes = inventory(row['metapath'], target, camera_view(row), row['object_index'])
            assert len(names) == n and ref > 0
            item.update(status='verified', ref_box=boxes[ref].tolist())
        result[target][str(ordinal)] = item
    value = dict(inputs=truth_pins(), records=result, FINAL_access=False)
    json_atomic(PRIVATE/'REVIEW_JOIN_PRIVATE.json', value)
    return result


def iou_upper_bound(raw, truth):
    """Yaw-independent bound using circumscribed horizontal and exact vertical extents."""
    box = np.asarray(raw, dtype=np.float64)
    size = box[:, 3:6].clip(min=1e-6)
    radius = np.linalg.norm(size[:, :2], axis=1)[:, None] / 2
    delta = box[:, :3] - truth[:3]
    target = truth.copy()
    target[:3] = 0
    horizontal = corners(target[None])[0]
    low = np.maximum(delta[:, :2] - radius, horizontal.min(0))
    high = np.minimum(delta[:, :2] + radius, horizontal.max(0))
    target_area = np.prod(truth[3:5])
    max_area = np.minimum(np.prod((high - low).clip(min=0), axis=1),
                          np.minimum(np.prod(size[:, :2], axis=1), target_area))
    height = (np.minimum(delta[:, 2] + size[:, 2]/2, truth[5]/2) -
              np.maximum(delta[:, 2] - size[:, 2]/2, -truth[5]/2)).clip(min=0)
    intersection = max_area * height
    return intersection / (np.prod(size, axis=1) + target_area * truth[5] - intersection)


def candidate_ious(raw, truths, anchor_ids, slot_boxes, load_points, minimum):
    """Evaluate all candidates that can clear a tested cut, with original PCA and no GT selection."""
    proposed = np.column_stack((raw, np.zeros(len(raw), dtype=raw.dtype)))
    proposed[:, 3:6] = np.maximum(raw[:, 3:6], 1e-6)
    proposed[anchor_ids.ravel()] = slot_boxes.reshape(-1, 7)
    upper = np.stack([iou_upper_bound(raw, t) for t in truths])
    needed = (upper >= minimum - BOUND_TOLERANCE).any(0)
    needed[anchor_ids.ravel()] = True
    unknown = np.flatnonzero(needed & ~np.isin(np.arange(len(raw)), anchor_ids))
    # The fixed PCA leaves the axis-aligned candidate unchanged when it cannot
    # meet the aspect-ratio rule, or has an originally degenerate side.
    positive = (raw[unknown, 3:6] > 0).all(1)
    positive_ids = unknown[positive]
    refine = positive_ids[raw[positive_ids, 3]/raw[positive_ids, 4] >= 1.5]
    if len(refine):
        points = load_points()
        x_order = np.argsort(points[:, 0], kind='stable')
        x = points[x_order, 0]
        for i in refine:
            center_x, radius_x = float(raw[i, 0]), float(raw[i, 3])/2
            slack = 1e-5 * max(1.0, abs(center_x), radius_x)
            start = np.searchsorted(x, center_x - radius_x - slack, side='left')
            stop = np.searchsorted(x, center_x + radius_x + slack, side='right')
            proposed[i] = orient_box(points[x_order[start:stop]], raw[i])
    values = np.zeros((len(truths), len(raw)), dtype=np.float64)
    for k, truth in enumerate(truths):
        values[k, needed] = many_iou(proposed[needed], truth)
    assert np.all(values <= upper + BOUND_TOLERANCE), float((values - upper).max())
    return values, proposed, needed


def same_query_pair(good, ids):
    subject, reference = ids[0][good[0]], ids[1][good[1]]
    return (bool(len(subject)), bool(len(reference)),
            bool(len(subject) and len(reference) and (subject[:, None] != reference).any()))


def dual_diagnosis(values, anchors, choices):
    roles = np.arange(2)[:, None]
    slots = values[roles, anchors]
    selected_ids = anchors[roles, choices.T]
    chosen = values[roles, selected_ids].T
    universal = np.broadcast_to(np.arange(values.shape[1]), values.shape)
    reasons, stages = [], []
    for cuts in DUAL_CUTS:
        good_full = same_query_pair(values > cuts[:, None], universal)
        good_slots = same_query_pair(slots > cuts[:, None], anchors)
        passed = (chosen > cuts).all(-1) & (selected_ids[0] != selected_ids[1])
        stage = np.where(not good_full[2], 0, np.where(not good_slots[2], 1, 2 + passed.astype(int)))
        assert not good_slots[2] or good_full[2]
        assert not passed.any() or good_slots[2]
        reasons.append([good_full, good_slots])
        stages.append(stage.tolist())
    return dict(good_roles=reasons, stage=stages,
                selected_ious=chosen.tolist(), role_slot_ious=slots.tolist(),
                slot_best=slots.max(1).tolist())


def source_panel(source):
    data = load_split(source, 'monitor', device='cpu')
    roster = read(PRIOR/source/'monitor_PRIVATE.json')
    n = len(roster)
    ids = data['known_ref'].nonzero().flatten()
    assert len(data['prior']) == n
    local = select(data, ids)
    choices = []
    for arm in ARMS:
        model = RelationalDecoder(arm).eval().requires_grad_(False)
        model.load_state_dict(torch.load(PRIOR/source/f'{arm}.pth', weights_only=True, map_location='cpu'))
        with torch.inference_mode():
            choices.append(np.concatenate([model(select(local, slice(i,i+16)))['choices'].numpy()
                                           for i in range(0,len(ids),16)]))
    choices = np.stack(choices, 1)
    out = []
    for j, i in enumerate(ids.tolist()):
        row = roster[i]
        assert row['known_reference']
        f = {k:local[k][j].numpy() for k in ('prior', 'gt', 'anchors', 'slot_boxes', 'quality')}
        for role in range(2):
            expected = many_iou(f['slot_boxes'][role], f['gt'][role])
            np.testing.assert_allclose(expected, f['quality'][role], atol=4e-8, rtol=0)
        v, _, _ = candidate_ious(f['prior'], f['gt'].astype(float), f['anchors'], f['slot_boxes'],
                                  lambda: points_for(row)[0], float(DUAL_CUTS.min()))
        out.append(dict(source=source, target=source, origin=row['origin'],
                        frame=frame(row), recording=recording(row), index=i, relation=row['relation'],
                        status='verified', **dual_diagnosis(v,f['anchors'],choices[j])))
    dst = PRIVATE/f'dual_source_{source}.json'
    json_atomic(dst, out)
    print('R176_SOURCE_DUAL', source, len(out), flush=True)
    return sha(dst)


def target_direction(direction):
    source, target = direction
    route = f'{source}_to_{target}'
    base = R165_DATA/route
    raw_seal = read(base/'SEAL.json')
    path = ORIGINAL/route
    pred_seal = read(path/'SEAL.json')
    roster = ordinary_rows(target)
    references = read(PRIVATE/'REVIEW_JOIN_PRIVATE.json')['records'][target]
    evaluated = []
    paired = []
    _, ref_iou = original(route)
    with np.load(PRIOR/'score'/f'{route}.npz', allow_pickle=False) as score:
        cached = {k:score[k] for k in ('slots', 'ious', 'relation', 'ordinal')}
        assert str(score['prediction']) == sha(path/'SEAL.json')
    for name, digest in sorted(pred_seal['files'].items()):
        assert sha(base/name) == raw_seal['files'][name] and sha(path/name) == digest
        with np.load(base/name, allow_pickle=False) as raw, np.load(path/name, allow_pickle=False) as pred:
            np.testing.assert_array_equal(raw['ordinals'], pred['ordinals'])
            data = {k:raw[k] for k in ('raw', 'ordinals', 'points_sha', 'indices_sha')}
            f = {k:pred[k] for k in ('slot_boxes', 'anchor_ids', 'choice', 'relation')}
            assert tuple(pred['arms']) == PRED_ARMS
        for j, ordinal in enumerate(data['ordinals']):
            i = int(ordinal)
            row = roster[i]
            gt, _ = labels(row)
            audit = references.get(str(i))
            pair = audit and audit['status'] == 'verified'
            if pair:
                truths = np.stack((gt, np.asarray(audit['ref_box'])))
            else:
                truths = gt[None]
            def load_points():
                points, ids = points_for(row)
                assert array_sha(points) == data['points_sha'][j].tobytes()
                assert array_sha(ids) == data['indices_sha'][j].tobytes()
                return points
            anchors, slots = f['anchor_ids'][j], f['slot_boxes'][j]
            v, _, _ = candidate_ious(data['raw'][j], truths, anchors, slots, load_points,
                                      float(DUAL_CUTS.min()) if pair else float(SUBJECT_CUTS.min()))
            chosen = f['choice'][j, ARM_IDS]
            q = v[0, anchors[0]]
            np.testing.assert_allclose(q, cached['slots'][i], rtol=0, atol=2e-12)
            np.testing.assert_allclose(q[chosen[:, 0]], cached['ious'][i, 4+np.array(ARM_IDS)],
                                       rtol=0, atol=2e-12)
            np.testing.assert_allclose(ref_iou[i], cached['ious'][i, :4], rtol=0, atol=2e-12)
            a = v[0].max() > SUBJECT_CUTS
            b = q.max() > SUBJECT_CUTS
            c = q[chosen[:,0]][None,:] > SUBJECT_CUTS[:,None]
            assert np.all(~b | a) and np.all(~c | b[:,None])
            stage = np.where(~a[:,None], 0, np.where(~b[:,None], 1, 2 + c.astype(int)))
            evaluated.append(dict(ordinal=i, stage=stage, pair_iou=q[chosen[0,0]],
                                  full_iou=q[chosen[1,0]], r158_iou=ref_iou[i,3],
                                  relation=f['relation'][j]))
            if audit:
                assert audit['relation'] == f['relation'][j]
                result = dict(source=source, target=target, index=i, status=audit['status'],
                              origin='human_review', relation=audit['relation'],
                              frame=frame(row), recording=recording(row))
                if pair:
                    result.update(dual_diagnosis(v,anchors,chosen))
                paired.append(result)
        print('R176_SUBJECT', route, len(evaluated), len(roster), flush=True)
    np.testing.assert_array_equal([r['ordinal'] for r in evaluated], np.arange(len(roster)))
    dst = PRIVATE/f'subject_{route}.npz'
    npz_atomic(dst, **{k:np.asarray([r[k] for r in evaluated]) for k in evaluated[0]},
               source_prediction=sha(path/'SEAL.json'), inputs=sha(PRIVATE/'INPUT_LOCK.json'))
    dd = PRIVATE/f'dual_target_{route}.json'
    json_atomic(dd, paired)
    return {dst.name:sha(dst), dd.name:sha(dd)}


def main():
    lock = input_lock()
    json_atomic(PRIVATE/'INPUT_LOCK.json', lock)
    locked()
    state('coverage_diagnosis', 'Post-seal A256, four-slot and selected output attribution')
    reviewed_references()
    outputs = {'REVIEW_JOIN_PRIVATE.json':sha(PRIVATE/'REVIEW_JOIN_PRIVATE.json')}
    with ProcessPoolExecutor(max_workers=3, mp_context=get_context('spawn')) as pool:
        for s, digest in zip(PLATFORMS, pool.map(source_panel, PLATFORMS), strict=True):
            outputs[f'dual_source_{s}.json'] = digest
    with ProcessPoolExecutor(max_workers=6, mp_context=get_context('spawn')) as pool:
        for value in pool.map(target_direction, ORDER):
            outputs.update(value)
    n = sum(len(ordinary_rows(t)) for _, t in ORDER)
    assert n == N
    json_atomic(PRIVATE/'DIAGNOSIS_SEAL.json', dict(n=n, files=outputs,
                inputs=sha(PRIVATE/'INPUT_LOCK.json'), FINAL_access=False))
    print('R176_DIAGNOSIS_SEALED', n, len(outputs), flush=True)


if __name__ == '__main__':
    main()
