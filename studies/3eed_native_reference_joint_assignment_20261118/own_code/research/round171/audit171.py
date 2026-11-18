"""Recompute the sealed assignments by enumeration and audit the exported scalar results."""
from collections import Counter
from math import fsum

import numpy as np

from scope171 import ARMS, HERE, ORDER, OUT, PLATFORMS, WORK, atomic_json, read, sha
from predict171 import MODES, REFERENCE_BANKS, read_npz
from review171 import verify_predictions
from score165 import table


def check_assignments(manifest):
    checks = 0
    total = Counter()
    for path, digest in manifest['files'].items():
        p = WORK / 'predictions' / path
        assert sha(p) == digest
        d = read_npz(p)
        scores = list(map(float, d['subject_unary']))
        c, f = d['choices'], d['method_boxes']
        sign = int(d['sign'])
        assert tuple(d['arms']) == ARMS and c.shape == (len(ARMS), 2)
        assert sign in (-1, 1) and str(d['design']) == manifest['design']
        for mat in (d['native_left'], d['mapped_left']):
            assert np.isfinite(mat).all() and (np.abs(mat) <= 1.).all() and (mat[:, -1] == 0.).all()
        valid = d['D_links'] >= 0
        main = d['D_native_unary'][:-1]
        subset = d['D_native_linked_unary'][:-1]
        assert np.array_equal(np.isfinite(subset), valid)
        if valid.any():
            assert np.array_equal(subset[valid], main[valid] - main[valid].max())
        else:
            assert d['D_native_linked_unary'][-1] == 0.
        for k, (bank, mode) in enumerate(zip(REFERENCE_BANKS, MODES, strict=True)):
            if not bank:
                expected = (0, -1)
                b = np.full(7, np.nan)
            else:
                mapped = bank.endswith('_mapped')
                x = list(map(float, d[bank + '_unary']))
                left = d['mapped_left'] if mapped else d['native_left']
                ref_top = x.index(max(x))
                if mode == 'independent':
                    expected = (scores.index(max(scores)), ref_top)
                else:
                    pool = [ref_top] if mode == 'fixed_reference' else range(len(x))
                    expected = max(((i, j) for i in range(len(scores)) for j in pool),
                                   key=lambda ij: scores[ij[0]] + (x[ij[1]] + sign * float(left[ij])))
                boxes = d['mapped_boxes'] if mapped else d['D_natural_boxes']
                assert len(boxes) + 1 == len(x)
                b = boxes[expected[1]] if expected[1] < len(boxes) else np.full(7, np.nan)
            assert tuple(c[k]) == expected
            np.testing.assert_array_equal(f[k, 0], d['subject_boxes'][expected[0]])
            np.testing.assert_array_equal(f[k, 1], b)
            checks += 1
        total[path.split('/')[0]] += 1
    assert dict(total) == manifest['rows']
    return checks


def groups(cases, group):
    if group == 'pooled':
        return [cases]
    if group == 'six_equal':
        keys = ORDER
    elif group in PLATFORMS:
        keys = [(s, t) for s, t in ORDER if t == group]
    else:
        keys = [tuple(group.split('_to_'))]
    return [x for s, t in keys if (x := [r for r in cases if (r['source'], r['target']) == (s, t)])]


def case_key(row):
    return row['cohort'], row['case'], row['source']


def unit(row, metric):
    value = float(row['official_iou'])
    return value if metric == 'mIoU' else float(value > {'Acc25': .25, 'Acc50': .50}[metric])


def audit():
    proof = verify_predictions()
    checked_choices = check_assignments(proof)
    seal = read(HERE / 'EVALUATION_AUDIT.json')
    assert seal['predictions'] == sha(HERE / 'PREDICTION_SEAL.json') and not seal['FINAL_access']
    assert seal['new_truth'] == sha(HERE / 'VISUAL_TRUTH_SEAL.json') and seal['maximum_sensor_iou_error'] < 1e-7
    assert all(sha(OUT / name) == digest for name, digest in seal['files'].items())
    cases, rows = [table(OUT / (f + '.csv')) for f in ('cases', 'arms')]
    flow = table(OUT / 'reference_flow.csv')
    measurements = table(OUT / 'metrics.csv')
    deltas = table(OUT / 'paired.csv')
    case_checks = Counter((x['cohort'], x['case']) for x in cases)
    assert len(case_checks) == 84 and set(case_checks.values()) == {2}
    assert len(cases) == 168 and sum(x['cohort'] == 'new_frames' for x in cases) == 72
    lookup = {(*case_key(x), x['arm']): x for x in rows}
    assert len(lookup) == len(rows) == checked_choices == 168 * len(ARMS)
    for r in rows:
        if r['fallback'] == '1':
            assert r['official_iou'] == lookup[(*case_key(r), 'r158_pca')]['official_iou']
            assert r['has_reference'] == '0' and r['subject_changed'] == '0'
    panels = {}
    for r in measurements:
        k = tuple(r[field] for field in ('cohort', 'scope', 'group'))
        if k not in panels:
            cohort, scope, group = k
            selected = [x for x in cases if x['cohort'] == cohort]
            if scope == 'supported_rewrite':
                selected = [x for x in selected if x['rewrite'] == 'supported']
            elif scope == 'both_verified':
                selected = [x for x in selected if x['verified_pair'] == '1']
            panels[k] = groups(selected, group)
    metrics = ('mIoU', 'Acc25', 'Acc50')
    max_error = 0.
    stats = 0
    for r in measurements:
        x = panels[tuple(r[k] for k in ('cohort', 'scope', 'group'))]
        y = [[lookup[(*case_key(z), r['arm'])] for z in g] for g in x]
        joint = [a for g in y for a in g]
        assert sum(x['reference_identity'] == '1' for x in joint) == int(r['reference_right'])
        assert sum(x['subject_identity'] == '1' for x in joint) == int(r['subject_right'])
        assert int(r['n']) == len(joint)
        if r['dual_right'] != '':
            assert sum(x['dual_correct'] == '1' for x in joint) == int(r['dual_right'])
        for m in metrics:
            value = 100 * fsum(fsum(unit(a, m) for a in g)/len(g) for g in y)/len(y)
            max_error = max(max_error, abs(value - float(r[m])))
            stats += 1
    for r in deltas:
        x = panels[tuple(r[k] for k in ('cohort', 'scope', 'group'))]
        for m in metrics:
            d = [[unit(lookup[(*case_key(z), r['arm'])], m) - unit(lookup[(*case_key(z), r['control'])], m) for z in g] for g in x]
            delta = 100 * fsum(fsum(g)/len(g) for g in d)/len(d)
            flat = [v for g in d for v in g]
            assert int(r[m+'_gains']) == sum(v > 1e-10 for v in flat)
            assert int(r[m+'_losses']) == sum(v < -1e-10 for v in flat)
            max_error = max(max_error, abs(delta - float(r[m+'_pp'])))
            stats += 1
    assert len(flow) == len({(*case_key(r), r['variant']) for r in flow}) == 2 * len(cases)
    for v in table(OUT / 'coverage.csv'):
        q = [r for r in flow if r['cohort'] == v['cohort'] and r['variant'] == v['variant']
             and (v['group'] == 'pooled' or r['target'] == v['group'])]
        p = [r for r in q if r['native_rank'] != '']
        assert len(q) == int(v['directions']) and len(p) == int(v['ref_n'])
        assert sum(int(r['lost_at_mapping']) for r in p) == int(v['lost_at_mapping'])
        for k in ('native', 'mapped', 'linked_native'):
            assert sum(int(r[k+'_rank']) > 0 for r in p) == int(v[k+'_coverage'])
        stats += 4
    assert max_error < 1e-9
    out = dict(arms_recomputed_by_exhaustive_enumeration=checked_choices, direction_records=len(cases),
                physical_frames=len(case_checks), reference_flow_rows=len(flow), verified_scalar_recomputations=stats,
                maximum_scalar_error=max_error, original_r170_controls_bit_exact=proof['r170_control_parity_rows'],
                fallback_rows_retained=sum(r['fallback'] == '1' for r in rows),
                all_failures_and_negatives_retained=True, FINAL_access=False)
    atomic_json(HERE / 'SCALAR_AUDIT.json', out)
    print(out, flush=True)
    return out


if __name__ == '__main__':
    audit()
