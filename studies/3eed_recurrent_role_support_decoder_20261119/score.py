"""Evaluate all sealed endpoints on the fixed ordinary-Validation rows."""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path
import csv
import hashlib
import math
import secrets

import numpy as np

from common import (ARMS, HERE, METRICS, ORDER, PLATFORMS, PRIVATE, RESULTS, ROOT, camera_view,
                    canonical, frame, json_atomic, npz_atomic, ordinary_rows, read, recording,
                    sha, verify_inputs)
from batch_yaw_iou import many_iou
from independent_core_v2 import transform_gt

BASE, NAMES = 'r158_pca', ('r158_pca', *ARMS)
PREDICTIONS = PRIVATE/'predictions'
PAIRS = (('joint', BASE), ('joint', 'without_feedback'), ('joint', 'without_rgb'),
         ('joint', 'fixed_proposals'), ('without_feedback', BASE),
         ('without_rgb', BASE), ('fixed_proposals', BASE))


def verify_predictions():
    verify_inputs()
    lock = PRIVATE/'EVALUATION_LOCK.json'
    seal = read(PREDICTIONS/'PREDICTION_SEAL.json')
    assert seal['contract'] == sha(lock)
    assert not (seal['FINAL_access'] or seal['target_truth_used'])
    assert seal['n'] == 18620 and len(seal['directions']) == 6
    p = read(lock)
    assert p['target_parameter_updates'] == 0
    assert p['input_lock'] == sha(PRIVATE/'INPUT_LOCK.json')
    for k, digest in p['code'].items():
        assert sha(HERE/k) == digest
    for direction, (digest, n) in seal['directions'].items():
        path = PREDICTIONS/direction/'SEAL.json'
        assert sha(path) == digest
        value = read(path)
        assert value['n'] == n and value['contract'] == seal['contract']
        assert not value['target_box_labels_used']
        assert all(sha(path.parent/f) == w for f, w in value['files'].items())
    return seal


@lru_cache(maxsize=256)
def gt_frame(path):
    return read(path)['ground_info']


def labels(row):
    annotations = gt_frame(row['metapath'])
    addressed = annotations[row['object_index']]
    assert canonical(addressed['caption']) == row['caption']
    truth = transform_gt(np.asarray(addressed['bbox_3d'], np.float64), row['platform'], camera_view(row))
    return truth, annotations


def local_points(points, box):
    shift = points-box[:3]
    s, c = math.sin(float(box[6])), math.cos(float(box[6]))
    pos = np.stack((c*shift[..., 0] + s*shift[..., 1],
                    -s*shift[..., 0] + c*shift[..., 1], shift[..., 2]), axis=-1)
    return np.all(np.abs(pos) <= box[3:6]*.5+1e-5, axis=-1)


def inventory(row, truth, labels):
    """Preserve the annotated subject at slot zero, with canonical deduplication."""
    nodes = [o['bbox_3d'] for o in labels]
    nodes.extend(o['bbox_3d_other'] for a in labels for o in a.get('others', []))
    boxes = [truth]
    seen = {tuple(truth)}
    width = len(labels[row['object_index']]['bbox_3d'])
    for raw in nodes:
        if not raw or len(raw) < 7 or not np.isfinite(raw[:7]).all() or min(raw[3:6]) <= 0:
            continue
        raw = np.asarray((list(raw)+[0., 0.])[:width], np.float64)
        box = transform_gt(raw, row['platform'], camera_view(row))
        key = tuple(box)
        if key not in seen:
            b = np.asarray(boxes)
            similar = (np.linalg.norm(b[:, :3]-box[:3], axis=1) < .15) & (
                       np.max(np.abs(b[:, 3:6]-box[3:6]), axis=1) < .35)
            if not similar.any():
                boxes.append(box)
            seen.add(key)
    return boxes


def owner_for(box, seeds, ids, count):
    region = local_points(seeds, box)
    n = int(region.sum())
    if n < 2:
        return -1
    mass = np.bincount(ids[region], minlength=count+1)
    best = mass[:count].argmax()
    hit = mass[best]
    mass[best] = 0
    return int(best) if hit >= .55*n and hit-mass.max() >= .15*n else -1


def source_direction(task):
    source, target = task
    top = PREDICTIONS/f'{source}_to_{target}'
    stored = read(top/'SEAL.json')
    observations = ordinary_rows(target)
    scores, support, flags, identity, ceilings = [], [], [], [], []
    ordinals = []
    wide = ROOT/'crossplatform/round165/ordinary'/top.name
    r165 = read(wide/'BBF_CUDA_RECEIPT.json')
    assert sha(wide/'MATCHED_BOXES_CUDA_PRIVATE.npz') == r165['output_sha256']
    with np.load(wide/'MATCHED_BOXES_CUDA_PRIVATE.npz', allow_pickle=False) as src:
        previous_box = src['boxes'][:, list(src['arms']).index(BASE)]
        previous_order = src['ordinals']
    with np.load(wide/'IOU_AUDIT_PRIVATE.npz', allow_pickle=False) as src:
        assert str(src['input_sha256']) == r165['output_sha256']
        previous_iou = src['values'][:, -1]
        np.testing.assert_array_equal(previous_order, src['ordinals'])
    for f, digest in sorted(stored['files'].items()):
        assert sha(top/f) == digest
        with np.load(top/f, allow_pickle=False) as f:
            z = dict(f)
        assert tuple(z['arms']) == NAMES
        sub = z['ordinals']
        np.testing.assert_array_equal(z['boxes'][:, 0], previous_box[sub])
        for j, i in enumerate(sub):
            row = observations[int(i)]
            truth, objects = labels(row)
            value = many_iou(z['boxes'][j], truth)
            seeds = z['seeds'][j]
            truth_points = local_points(seeds, truth)
            p = z['support'][j, :, 0].astype(np.float64)
            support.append(p[:, truth_points].sum(axis=-1)/p.sum(axis=-1))
            scores.append(value)
            final_slots = z['subject_slots'][j]
            slot_iou = many_iou(final_slots.reshape(-1, 7), truth).reshape(len(ARMS), -1).max(1)
            assert (slot_iou[:3]+1e-10 >= value[1:4]).all()
            ceilings.append(slot_iou)
            instances = inventory(row, truth, objects)
            seeds = np.unique(seeds, axis=0)
            hit = np.stack([local_points(seeds, b) for b in instances])
            owners = np.where(hit.sum(0) == 1, hit.argmax(0), len(instances))
            identity.append([owner_for(b, seeds, owners, len(instances)) for b in z['boxes'][j]])
        flags.append(np.c_[z['is_relation'], z['left_right'], z['seed_visible']])
        ordinals.extend(int(x) for x in sub)
    o, value = np.asarray(ordinals), np.stack(scores)
    np.testing.assert_array_equal(o, previous_order)
    np.testing.assert_allclose(value[:, 0], previous_iou, rtol=0., atol=2e-12)
    npz_atomic(PRIVATE/'score'/f'{top.name}.npz', ordinals=o, values=value, owners=np.asarray(identity),
               flags=np.concatenate(flags), support=np.asarray(support),
               slot_ceiling=np.asarray(ceilings),
               input_digest=sha(top/'SEAL.json'))
    print('POST_SEAL_SCORE', top.name, len(o), flush=True)
    return top.name


def save_csv(name, rows):
    if not rows:
        raise ValueError(f'No results for {name}')
    path = RESULTS/name
    with path.with_suffix('.tmp').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix('.tmp').replace(path)


def partitions(f):
    return {'all': np.ones(len(f), bool), 'reference_mention': f[:, 0].astype(bool),
            'direct_left_right': f[:, 1].astype(bool)}


def metric_arrays(x):
    return np.stack((x, x > .25, x > .50), axis=-1).astype(float)


def paired_scopes(x, cohort, macro):
    metrics = [metric_arrays(s[cohort]) for s in x]
    stats = (np.mean([a.mean(0) for a in metrics if len(a)], 0) if macro else
             np.concatenate(metrics).mean(0))
    return stats*100., np.concatenate(metrics)


def summary():
    data = {}
    for source, target in ORDER:
        name = f'{source}_to_{target}'
        with np.load(PRIVATE/'score'/f'{name}.npz', allow_pickle=False) as part:
            z = dict(part)
        rows = ordinary_rows(target)
        data[name] = dict(z, groups=np.asarray([recording(rows[int(i)]) for i in z['ordinals']]),
                          frames=np.asarray([frame(rows[int(i)]) for i in z['ordinals']]))
    endpoints, changes, support_stats, recordings, ceilings = [], [], [], [], []
    for cohort in ('all', 'reference_mention', 'direct_left_right'):
        sub = {k: partitions(s['flags'])[cohort] for k,s in data.items()}
        for name in (*data, 'six_equal', 'row_weighted'):
            groups = (list(data) if name in ('six_equal', 'row_weighted') else [name])
            parts = [metric_arrays(data[g]['values'][sub[g]]) for g in groups if sub[g].any()]
            if not parts:
                continue
            merged = np.concatenate(parts)
            macro = name == 'six_equal'
            stats = (np.mean([v.mean(0) for v in parts], axis=0) if macro else merged.mean(0))*100.
            ids = np.concatenate([data[g]['owners'][sub[g]] for g in groups])
            reg = np.concatenate([data[g]['groups'][sub[g]] for g in groups])
            fr = np.concatenate([data[g]['frames'][sub[g]] for g in groups])
            for j, arm in enumerate(NAMES):
                endpoints.append(dict(scope=name, cohort=cohort, arm=arm, n=len(merged),
                                      frames=len(set(fr)), recordings=len(set(reg)), **dict(zip(METRICS, stats[j])),
                                      subject_owners_correct=int((ids[:, j] == 0).sum()),
                                      subject_owners_unknown=int((ids[:, j] < 0).sum())))
                if j:
                    a = np.concatenate([data[g]['support'][sub[g], j-1] for g in groups])
                    support_stats.append(dict(scope=name, cohort=cohort, arm=arm, n=len(a),
                                              mean_GT_support_mass=float(a.mean()),
                                              zero_GT_support=int((a == 0).sum())))
                    if arm != 'fixed_proposals':
                        capacity = [data[g]['slot_ceiling'][sub[g], j-1] for g in groups if sub[g].any()]
                        capable = [metric_arrays(a) for a in capacity]
                        oracle = (np.mean([a.mean(0) for a in capable], 0) if macro
                                  else np.concatenate(capable).mean(0))*100.
                        joined = np.concatenate(capacity)
                        ceilings.append(dict(scope=name, cohort=cohort, arm=arm, n=len(joined),
                                             oracle_mIoU=float(oracle[0]), oracle_Acc25=float(oracle[1]),
                                             oracle_Acc50=float(oracle[2]),
                                             missed_within_slots_at25=int(((merged[:,j,1]==0)&(joined>.25)).sum()),
                                             missed_within_slots_at50=int(((merged[:,j,2]==0)&(joined>.50)).sum())))
            for after, before in PAIRS:
                i, j = NAMES.index(after), NAMES.index(before)
                d = merged[:, i]-merged[:, j]
                metrics = {}
                for m, delta, vals in zip(METRICS, stats[i]-stats[j], d.T, strict=True):
                    tol = 1e-10
                    metrics.update({m+'_pp':delta, m+'_gain':int((vals>tol).sum()),
                                    m+'_loss':int((vals< -tol).sum()),
                                    m+'_tie':int((np.abs(vals)<=tol).sum())})
                changes.append(dict(scope=name, cohort=cohort, after=after, before=before, n=len(d), **metrics))
    for direction, data_row in data.items():
        values = metric_arrays(data_row['values'])
        for after, before in (('joint', BASE), ('joint', 'without_feedback')):
            i, j = NAMES.index(after), NAMES.index(before)
            sums = []
            for group in np.unique(data_row['groups']):
                where = data_row['groups'] == group
                sums.append((int(where.sum()), *((values[where, i]-values[where, j]).mean(0)*100)))
            arr = np.asarray(sums)
            record = {'direction': direction, 'after': after, 'before': before,
                      'recordings': len(arr), 'rows': len(values)}
            for m, delta in zip(METRICS, arr[:, 1:].T, strict=True):
                record.update({m+'_positive':int((delta>1e-10).sum()), m+'_negative':int((delta< -1e-10).sum()),
                               m+'_range_min':float(delta.min()), m+'_range_max':float(delta.max())})
            recordings.append(record)
        # Anonymous, paired scalar rows, retaining both tails. No frame, case ID or ordinal is exported.
        # The keyed order prevents matching by original benchmark position.
        key_file = PRIVATE/'scalar_shuffle_PRIVATE.bin'
        if not key_file.exists():
            key_file.write_bytes(secrets.token_bytes(32))
        key = key_file.read_bytes()
        rank = sorted(range(len(values)), key=lambda i: hashlib.blake2b(
            (direction+str(i)).encode(), key=key, digest_size=8).digest())
        csv_rows = []
        for j in rank:
            row = {'has_reference_mention': int(data_row['flags'][j, 0]),
                   'direct_left_right': int(data_row['flags'][j, 1])}
            row.update({arm: f'{data_row["values"][j, a]:.17g}' for a, arm in enumerate(NAMES)})
            csv_rows.append(row)
        save_csv(f'paired_rows_{direction}.csv', csv_rows)
    save_csv('endpoints.csv', endpoints)
    save_csv('paired_deltas.csv', changes)
    save_csv('point_support.csv', support_stats)
    save_csv('recording_signs.csv', recordings)
    save_csv('slot_ceiling.csv', ceilings)


def main():
    verify_predictions()
    RESULTS.mkdir(exist_ok=True)
    (PRIVATE/'score').mkdir(exist_ok=True, mode=0o700)
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context('spawn')) as p:
        for name in p.map(source_direction, ORDER):
            print('SCORED_COMPLETE', name, flush=True)
    summary()
    receipt = {'predictions': sha(PREDICTIONS/'PREDICTION_SEAL.json'),
               'scores': {p.name:sha(p) for p in (PRIVATE/'score').glob('*.npz')},
               'public_results': {p.name:sha(p) for p in RESULTS.glob('*.csv')}, 'FINAL_access': False}
    json_atomic(PRIVATE/'SCORING_SEAL.json', receipt)


if __name__ == '__main__':
    main()
