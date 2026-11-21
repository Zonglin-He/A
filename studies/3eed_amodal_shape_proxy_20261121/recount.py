"""Recompute the exported metrics from anonymous scalar rows."""

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, median


OUT = Path(__file__).resolve().parent / 'results'


def table(file):
    with (OUT / f'{file}.csv').open(newline='') as stream:
        return list(csv.DictReader(stream))


def group(rows, row):
    result = rows
    if row['panel'] == 'all_recording_separated':
        result = [r for r in result if r['separation'] == 'new_recording']
    elif row['panel'] != 'all':
        result = [r for r in result if r['panel'] == row['panel']]
    if row['target'] != 'macro':
        result = [r for r in result if r['target'] == row['target']]
    if row['scope'] != 'all':
        result = [r for r in result if r['fitted'] == 'True']
    if row['scope'] == 'high_support':
        result = [r for r in result if float(r['support_inside_gt_fraction']) >= .8]
    assert len(result) == int(row['n'])
    return result


def macro(rows, values):
    groups = defaultdict(list)
    for r, x in zip(rows, values, strict=True):
        groups[r['target']].append(x)
    return mean(mean(xs) for xs in groups.values())


def metrics(rows, arm):
    iou = [float(r[arm]) for r in rows]
    return [100. * macro(rows, values) for values in ([x > .25 for x in iou], [x > .50 for x in iou], iou)]


def verify():
    rows = table('rows')
    seal = json.loads((OUT / 'prediction_seal.json').read_text())
    evaluation = json.loads((OUT / 'evaluation_receipt.json').read_text())
    assert len(rows) == len({r['uid'] for r in rows}) == seal['frames'] == evaluation['frames'] == 416
    assert sum(r['panel'] == 'validation' for r in rows) == 288
    assert sum(r['separation'] == 'new_recording' for r in rows) == 224
    assert sum(r['fitted'] == 'True' for r in rows) == seal['matched_fit_and_selection_replayed']
    assert Counter(r['status'] for r in rows) == seal['stages']
    assert all(r['B'] == r['C'] and r['B_center_m'] == r['C_center_m']
               for r in rows if r['fitted'] == 'False')
    assert all(0. <= float(r[a]) <= 1. for r in rows for a in ('A', 'B', 'C'))
    assert all(hashlib.sha256((OUT / f'{f}.csv').read_bytes()).hexdigest() == checksum
               for f, checksum in evaluation['results'].items())
    assert evaluation['prediction_seal'] == hashlib.sha256((OUT/'prediction_seal.json').read_bytes()).hexdigest()

    worst = 0.
    report, paired, audits = table('metrics'), table('paired'), table('ray_audit')
    for r in report:
        subset = group(rows, r)
        actual = metrics(subset, r['arm'])
        worst = max(worst, *(abs(x - float(r[k])) for x, k in zip(actual, ('acc25', 'acc50', 'miou'))))
        center = [float(x[r['arm']+'_center_m']) for x in subset if x[r['arm']+'_center_m']]
        assert len(center) == int(r['center_n'])
        if center:
            assert abs(median(center) - float(r['center_median_m'])) < 1.e-10
    for r in paired:
        subset = group(rows, r)
        a, b = r['arm'], r['reference']
        pairs = [(float(x[a]), float(x[b])) for x in subset]
        for threshold, k in ((.25, '25'), (.50, '50')):
            assert int(r['rescue'+k]) == sum(v > threshold >= u for v, u in pairs)
            assert int(r['harm'+k]) == sum(u > threshold >= v for v, u in pairs)
        actual = [x - y for x, y in zip(metrics(subset, a), metrics(subset, b))]
        worst = max(worst, *(abs(x - float(r[k])) for x, k in
            zip(actual, ('delta25_pp', 'delta50_pp', 'delta_miou_pp'))))
        matched = [x for x in subset if x[a+'_center_m'] and x[b+'_center_m']]
        offsets = [float(x[a+'_center_m']) - float(x[b+'_center_m']) for x in matched]
        assert len(offsets) == int(r['center_pairs'])
        assert sum(v < -1.e-10 for v in offsets) == int(r['center_closer'])
        assert sum(v > 1.e-10 for v in offsets) == int(r['center_farther'])
        if matched:
            assert abs(macro(matched, offsets) - float(r['mean_center_delta_m'])) < 1.e-10
    for r in audits:
        subset = group(rows, r)
        arm = r['arm']
        rays = sum(int(x['audit_rays'] or 0) for x in subset)
        hits = sum(int(x[f'{arm}_audit_hits'] or 0) for x in subset)
        assert rays == int(r['audit_rays']) and hits == int(r['audit_hits'])
        if rays:
            loss = sum(float(x[f'{arm}_audit_abs_capped_sum_m'] or 0.) for x in subset)
            assert abs(loss / rays - float(r['audit_mean_capped_m'])) < 1.e-10
    assert worst < 1.e-10
    return dict(frames=len(rows), metric_rows=len(report), paired_rows=len(paired), audit_rows=len(audits),
                max_metric_difference_pp=worst, fallbacks_equal=True, FINAL_access=False)


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
