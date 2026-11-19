"""Separately recompute sealed oriented boxes via the sensor-frame convex evaluator."""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import csv
from multiprocessing import get_context
from pathlib import Path

import ijson
import numpy as np

from common import (ORDER, PRIVATE, RESULTS, ROOT, camera_view, canonical, json_atomic,
                    ordinary_rows, read, sha)
from score import NAMES, PAIRS, PREDICTIONS, verify_predictions
from audit160 import sensor_frame
from independent_core_v2 import transform_gt, yaw_matrix
from obb_iou import oriented_iou


@lru_cache(maxsize=256)
def independent_annotations(path):
    with Path(path).open('rb') as stream:
        return [(row['caption'], np.asarray(row['bbox_3d'], np.float64))
                for row in ijson.items(stream, 'ground_info.item', use_float=True)]


def sensor_iou(b, t, target, view):
    b = np.array([sensor_frame(x, target, view) for x in np.asarray(b, np.float64)])
    t = sensor_frame(t.astype(np.float64), target, view)
    labels = np.zeros(len(b))
    tr = yaw_matrix(t[6])
    for j, o in enumerate(b):
        rot = yaw_matrix(o[6])
        if (np.abs(t[:3]-o[:3]) <= (np.abs(rot)@o[3:6] + np.abs(tr)@t[3:6])*.5+1e-9).all():
            labels[j] = oriented_iou(o[:3], o[3:6], rot, t[:3], t[3:6], tr)
    return labels


def compare(directions):
    source, target = directions
    name = f'{source}_to_{target}'
    bases = ordinary_rows(target)
    folder = PREDICTIONS/name
    files = read(folder/'SEAL.json')['files']
    previous = PRIVATE/'score'/f'{name}.npz'
    assert sha(previous) == read(PRIVATE/'SCORING_SEAL.json')['scores'][previous.name]
    with np.load(previous, allow_pickle=False) as cached:
        assert str(cached['input_digest']) == sha(folder/'SEAL.json')
        primary = cached['values']
        order = cached['ordinals']
    pos, disagreement, max_error, certain_entropy = 0, [0, 0], 0., 0.
    for f in sorted(files):
        with np.load(folder/f, allow_pickle=False) as z:
            ordinals, boxes = z['ordinals'], z['boxes']
            entropy = z['entropy']
            assert float(z['visibility'].min()) >= 0. and float(z['visibility'].max()) < 1.00001
            assert np.isfinite(z['seeds']).all()
            assert np.array_equal(ordinals, order[pos:pos+len(ordinals)])
            np.testing.assert_allclose(z['support'].sum(-1, dtype=np.float64), 1., atol=.003, rtol=0.)
        for ind, x in zip(ordinals, boxes, strict=True):
            row = bases[int(ind)]
            caption, raw = independent_annotations(row['metapath'])[row['object_index']]
            assert canonical(caption) == row['caption']
            cam = camera_view(row)
            truth = transform_gt(raw, target, cam)
            other = sensor_iou(x, truth, target, cam)
            ref = primary[pos]
            max_error = max(max_error, float(np.abs(other-ref).max()))
            for k, t in enumerate((.25, .50)):
                disagreement[k] += int(np.count_nonzero((other>t) != (ref>t)))
            pos += 1
        certain_entropy += float(entropy[:, 0, 0].sum())
    assert pos == len(order) == len(bases)
    assert max_error < 1e-7 and disagreement == [0, 0]
    print('INDEPENDENT_CONVEX', name, pos, max_error, flush=True)
    return name, dict(n=pos, max_abs_iou_error=max_error, threshold_disagreements=disagreement,
                      mean_subject_support_entropy=abs(certain_entropy)/pos)


def public_accuracy():
    protected = (PRIVATE/'SCORING_SEAL.json')
    for name, digest in read(protected)['public_results'].items():
        assert sha(RESULTS/name) == digest
    with (RESULTS/'endpoints.csv').open(encoding='utf-8') as f:
        endpoints = list(csv.DictReader(f))
    checks, negatives = 0, 0
    for source, target in ORDER:
        name = f'{source}_to_{target}'
        with (RESULTS/f'paired_rows_{name}.csv').open(encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        assert set(rows[0]) == {'has_reference_mention', 'direct_left_right', *NAMES}
        values = np.asarray([[float(row[a]) for a in NAMES] for row in rows])
        assert len(rows) == len(ordinary_rows(target))
        bits = np.stack((values, values>.25, values>.50), axis=-1)
        set_masks = {'all': np.ones(len(rows), bool),
                     'reference_mention': np.asarray([r['has_reference_mention']=='1' for r in rows]),
                     'direct_left_right': np.asarray([r['direct_left_right']=='1' for r in rows])}
        for cohort, used in set_masks.items():
            part = [r for r in endpoints if r['scope']==name and r['cohort']==cohort]
            if not used.any():
                assert not part
                continue
            assert [r['arm'] for r in part] == list(NAMES)
            for j, row in enumerate(part):
                reported = np.asarray([float(row[x]) for x in ('mIoU','Acc25','Acc50')])
                np.testing.assert_allclose(100*bits[used,j].mean(0), reported, atol=1e-9, rtol=0.)
                assert int(row['n']) == int(used.sum())
                checks += 1
        negatives += int((values[:,1] < values[:,0]-1e-10).sum())
    with (RESULTS/'paired_deltas.csv').open(encoding='utf-8') as f:
        pairs = list(csv.DictReader(f))
    main = next(x for x in pairs if x['scope']=='six_equal' and x['cohort']=='all'
                and x['before']=='r158_pca' and x['after']=='joint')
    assert int(main['mIoU_loss']) == negatives
    return {'public_direction_cohort_rows_recomputed': checks,
            'public_main_negative_rows_confirmed': negatives}


def main():
    verify_predictions()
    scoring = read(PRIVATE/'SCORING_SEAL.json')
    assert scoring['predictions'] == sha(PREDICTIONS/'PREDICTION_SEAL.json')
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context('spawn')) as pool:
        out = dict(pool.map(compare, ORDER))
    receipt = dict(n=sum(row['n'] for row in out.values()), directions=out,
                   independently_recomputed_scalar_results=public_accuracy(),
                   scored_receipt=sha(PRIVATE/'SCORING_SEAL.json'), FINAL_access=False)
    assert receipt['n'] == 18620
    json_atomic(RESULTS/'independent_audit.json', receipt)
    print('AUDIT_COMPLETED', receipt['n'], flush=True)


if __name__ == '__main__':
    main()
