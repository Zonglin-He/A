"""Recount all anonymous rows and replay sealed boxes on the original inference path."""

import csv

import numpy as np
import torch

from scope183 import ARMS, DIRECTIONS, PLATFORMS, atomic, check_input, digest, frame, ordinary, read
from predict183 import OUT, LockedEvaluator
from score183 import METRICS, RESULTS, metrics, prediction_seal, shards
from train import prepare_device
from capture182 import FIT_FEATURES, capture_fingerprint


def csv_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def check_scalars():
    data = csv_rows(RESULTS/'anonymous_rows.csv')
    assert len(data) == 18620
    groups = {f'{s}_to_{t}': [r for r in data if r['source'] == s and r['target'] == t] for s, t in DIRECTIONS}
    base = {}
    for key, group in groups.items():
        vals = np.array([[float(row[k]) for k in ARMS] for row in group])
        extra = np.array([[int(row[f'outside_{k}']) for k in ARMS] for row in group], bool)
        four = np.array([float(row['four_best']) for row in group])
        base[key] = vals, extra, four
    joined = tuple(np.concatenate([base[d][i] for d in groups]) for i in range(3))
    for name in ('all_pooled', 'six_equal'):
        base[name] = joined
    for row in csv_rows(RESULTS/'direction.csv'):
        d, a = row['direction'], row['arm']
        i = ARMS.index(a)
        if d == 'six_equal':
            sm = [metrics(base[name][0][:, i]) for name in groups]
            result = {m: float(np.mean([x[m] for x in sm])) for m in METRICS}
        else:
            result = metrics(base[d][0][:, i])
        for k, actual in result.items():
            assert abs(actual - float(row[k])) < 1e-10
        assert len(base[d][0]) == int(row['n'])
    for row in csv_rows(RESULTS/'transitions.csv'):
        values, outside, four = base[row['direction']]
        i, t = ARMS.index(row['arm']), float(row['threshold'])
        a, b = values[:, i] > t, values[:, 0] > t
        rx, hx = a & ~b, b & ~a
        expected = dict(total_rescues=rx.sum(), total_harms=hx.sum(), net=rx.sum()-hx.sum(),
                        rescued_by_outside=(rx & outside[:, i]).sum(),
                        harms_by_outside=(hx & outside[:, i]).sum(),
                        rescued_original_four_miss=(rx & (four <= t)).sum())
        for key, actual in expected.items():
            assert actual == int(row[key])
    return len(data)


def verify():
    prepare_device()
    total = check_scalars()
    assert read(FIT_FEATURES / 'SEALED.json')['implementation'] == capture_fingerprint()
    pin = prediction_seal()
    data = ordinary()
    boxes = 0
    for source in PLATFORMS:
        obj = LockedEvaluator(source)
        for s, t in DIRECTIONS:
            if s != source:
                continue
            key = f'{s}_to_{t}'
            first = next(shards(key, pin['directions'][key]))
            for j in (0, 61):
                row = data[t][int(first['ordinals'][j])]
                check_input(row)
                reproduced = obj.evaluate(frame(row))
                for k in ('boxes', 'four', 'choices', 'outside'):
                    np.testing.assert_array_equal(reproduced[k], first[k][j])
                boxes += len(ARMS)
        del obj
        torch.cuda.empty_cache()
    assert boxes == 84
    atomic(RESULTS/'verify.json', dict(all_anonymous_rows_recounted=total, all_acc_and_threshold_counts_match=True,
         endpoints_reproduced_bit_exact=boxes, standard_R182_path_used=True, R182_source_dependencies_match=True,
         predictions_seal_sha256=digest(OUT/'SEALED.json'),target_labels_read=False,FINAL_access=False))


if __name__ == '__main__':
    verify()
