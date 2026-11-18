"""Evaluate sealed frozen predictions and keep paired scalar evidence for every arm."""
from collections import Counter
import csv
from pathlib import Path

import numpy as np

from edge_predict import ARMS, HERE, PRIVATE, R171, WORK, pins, read, sha, atomic_json


COMPARISONS = (
    ('role_interaction_joint', 'r158_pca'),
    ('role_interaction_joint', 'r171_joint'),
    ('role_interaction_joint', 'same_evidence_independent'),
    ('role_interaction_joint', 'same_evidence_geometric'),
    ('role_interaction_joint', 'role_interaction_mismatched'),
    ('same_evidence_geometric', 'r171_joint'),
    ('r171_joint', 'r171_independent'),
)
METRICS = ('mIoU', 'Acc25', 'Acc50')


def csv_atomic(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    template = path.with_suffix('.tmp')
    with template.open('w', newline='') as f:
        out = csv.DictWriter(f, list(rows[0]))
        out.writeheader()
        out.writerows(rows)
    template.replace(path)


def verify():
    base = PRIVATE / 'full'
    seal = read(base / 'SEAL.json')
    assert not seal['target_labels_read_by_prediction'] and seal['unchanged_weights']
    assert not seal['protected_data_access'] and not seal['pilot']
    assert read(base / 'LOCK.json') == pins() and seal['lock'] == sha(base / 'LOCK.json')
    assert seal['parent'] == sha(R171 / 'PREDICTION_SEAL.json')
    old = read(R171 / 'PREDICTION_SEAL.json')
    assert seal['files'].keys() == old['files'].keys() and len(seal['files']) == 168
    assert all(sha(base / key) == digest for key, digest in seal['files'].items())
    assert all(sha(WORK / 'predictions' / key) == digest for key, digest in old['files'].items())
    return base, seal, old


def iou_metrics(ious):
    return np.array((ious, ious > .25, ious > .50), dtype=float)


def both_known(subject, ref):
    return subject >= 0 and ref >= 0


def per_case(base, old, cohort, entry, blind, key, source, recording):
    from labels166 import occupied_owners, physical_labels
    from predict171 import read_npz
    from score165 import many_iou, sensor_values
    from score170 import label, status_for

    row, case, target = entry['row'], entry['case'], entry['target']
    name = f'{cohort}/{case:02d}_{source}.npz'
    a = read_npz(base / name)
    d = read_npz(WORK / 'predictions' / name)
    assert a['parent'] == old['files'][name] and tuple(a['arms']) == ARMS
    assert a['pin'] == sha(base / 'LOCK.json')
    sub, ref = label(key, blind['subject']), label(key, blind['reference'])
    names, gt, seeds, owners = physical_labels(row, d['seeds'])
    assert len(names) == key['inventory_size'] and max(sub, ref) < len(names)
    positions = a['choices']
    shapes = a['boxes']
    have_ref = np.isfinite(shapes[:, 1]).all(1)
    sbox, rbox = d['subject_boxes'], d['D_natural_boxes']
    np.testing.assert_array_equal(shapes[:, 0], sbox[positions[:, 0]])
    np.testing.assert_array_equal(shapes[have_ref, 1], rbox[positions[have_ref, 1]])
    np.testing.assert_array_equal(shapes[:3], d['method_boxes'][[0, 5, 8]])
    assigned = occupied_owners(shapes, seeds, owners, len(names))
    values = many_iou(shapes[:, 0], gt[0])
    independent = sensor_values(shapes[:, 0], gt[0], target, int(row['suffix'] or 0))
    np.testing.assert_allclose(values, independent, rtol=0., atol=1e-7)
    np.testing.assert_array_equal(iou_metrics(values)[1:], iou_metrics(independent)[1:])

    def verified_IoU():
        if sub < 0:
            return [''] * len(ARMS)
        return [float(x) for x in many_iou(shapes[:, 0], gt[sub])]

    has_dual = both_known(sub, ref)
    sk = occupied_owners(sbox, seeds, owners, len(names))
    rk = occupied_owners(rbox, seeds, owners, len(names)) if len(rbox) else np.empty(0, int)
    sr = a['subject_valid']
    rr = a['reference_valid']
    truth = dict(known_subject=int(sub >= 0), known_reference=int(ref >= 0), known_dual=int(has_dual),
                 available_subject=int((sk == sub).any()) if sub >= 0 else '',
                 available_reference=int((rk == ref).any()) if ref >= 0 else '',
                 subject_has_intervention=int((sk[sr] == sub).any()) if sub >= 0 else '',
                 reference_has_intervention=int((rk[rr] == ref).any()) if ref >= 0 else '')
    common = dict(cohort=cohort, frame=case, source=source, target=target, recording=recording,
                  reason=str(a['reason']), rewrite=entry['rewrite']['status'], **truth)
    edge = a['edge']
    forward, reverse = a['source_forward'], a['source_reverse']
    rows = []
    for i, (arm, viou, (s, r)) in enumerate(zip(ARMS, verified_IoU(), positions, strict=True)):
        rows.append(dict(**common, arm=arm, official_iou=float(values[i]), verified_subject_iou=viou,
                         sent_reference=int(have_ref[i]), fallback=int(not have_ref[i]),
                         subject_status=status_for(sub, assigned[i, 0]),
                         reference_status=status_for(ref, assigned[i, 1], have_ref[i]),
                         dual_correct=int((assigned[i] == (sub, ref)).all()) if has_dual and i else '',
                         subject_changed=int(s != 0), signed_LEFT=float(d['sign'] * d['native_left'][s, r])
                         if have_ref[i] else '', subject_choice=int(s), reference_choice=int(r)))
    evidence = dict(**common, subject_regions=len(sr), reference_regions=len(rr),
                    attention_reconstruction_max_error=float(a['attention_error']),
                    seconds=float(a['seconds']), abs_edge_max=float(np.max(np.abs(edge))),
                    abs_forward_max=float(np.max(np.abs(forward), initial=0)),
                    abs_reverse_max=float(np.max(np.abs(reverse), initial=0)),
                    effective_shuffle=int(not np.array_equal(edge, a['mismatched'])))
    for name_are, array in [('forward', forward), ('reverse', reverse), ('edge', edge[np.ix_(sr, rr)])]:
        if has_dual and forward.size:
            good = (sk[sr] == sub)[:, None] & (rk[rr] == ref)[None]
            known = (sk[sr] >= 0)[:, None] & (rk[rr] >= 0)[None]
            bad = known & ~good
        else:
            good = bad = np.zeros_like(array, bool)
        evidence[name_are + '_good_minus_bad_best'] = float(array[good].max() - array[bad].max()) if good.any() and bad.any() else ''
    return rows, evidence, float(np.max(np.abs(values-independent)))


def directions(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row['source'], row['target']), []).append(row)
    return list(groups.values())


def subset(name, rows):
    if name == 'all':
        return rows
    if name == 'rewrite':
        return [r for r in rows if r['rewrite'] == 'supported']
    if name == 'identity':
        return [r for r in rows if r['known_dual']]
    return [r for r in rows if r['reason'] == 'eligible']


def macro(rows, metric):
    return 100. * np.mean([np.mean([metric(r) for r in group], axis=0) for group in directions(rows)], axis=0)


def orders(row):
    return row['cohort'], row['frame'], row['source']


def summaries(rows):
    from labels166 import RIGHT

    methods, deltas = [], []
    scopes = ('all', 'rewrite', 'identity', 'eligible')
    for cohort in ('replay170', 'new_frames', 'combined'):
        panel = [r for r in rows if cohort == 'combined' or r['cohort'] == cohort]
        for scope in scopes:
            pool = subset(scope, panel)
            directions_in = sorted({(r['source'], r['target']) for r in pool})
            for direction in [None, *directions_in]:
                selected = [r for r in pool if direction is None or (r['source'], r['target']) == direction]
                by_arm = {arm: sorted((r for r in selected if r['arm'] == arm), key=orders) for arm in ARMS}
                if not by_arm[ARMS[0]]:
                    continue
                route = 'direction_macro' if direction is None else '_to_'.join(direction)
                meta = dict(cohort=cohort, scope=scope, direction=route, n=len(by_arm[ARMS[0]]),
                            frames=len({(r['cohort'], r['frame']) for r in selected}),
                            recordings=len({r['recording'] for r in selected}), directions=len(directions(selected)))
                for arm, content in by_arm.items():
                    scores = macro(content, lambda r: iou_metrics(r['official_iou']))
                    methods.append(dict(**meta, arm=arm, **dict(zip(METRICS, scores, strict=True)),
                                        subject_verified=sum(r['known_subject'] for r in content),
                                        subject_correct=sum(r['subject_status'] == RIGHT for r in content),
                                        reference_verified=sum(r['known_reference'] for r in content),
                                        reference_correct=sum(r['reference_status'] == RIGHT for r in content),
                                        dual_verified=sum(r['known_dual'] for r in content),
                                        dual_correct=sum(r['dual_correct'] == 1 for r in content) if arm != ARMS[0] else '',
                                        abstained=sum(r['fallback'] for r in content)))
                for a, b in COMPARISONS:
                    first, second = by_arm[a], by_arm[b]
                    assert [orders(r) for r in first] == [orders(r) for r in second]
                    y = [dict(**r, difference=iou_metrics(r['official_iou']) - iou_metrics(s['official_iou']))
                         for r, s in zip(first, second, strict=True)]
                    score = macro(y, lambda r: r['difference'])
                    stacked = np.stack([r['difference'] for r in y])
                    matched = dict(zip(METRICS, score, strict=True))
                    counts = {name+'_'+key: int(np.sum(sign*stacked[:, j] > 1e-10))
                              for j, name in enumerate(METRICS) for key, sign in [('gain', 1), ('loss', -1)]}
                    deltas.append(dict(**meta, arm=a, control=b, **matched, **counts,
                                       joint_gains=sum(x['dual_correct'] == 1 and q['dual_correct'] != 1
                                                       for x, q in zip(first, second, strict=True)),
                                       joint_losses=sum(x['dual_correct'] != 1 and q['dual_correct'] == 1
                                                        for x, q in zip(first, second, strict=True))))
    return methods, deltas


def main():
    from edge_predict import PLATFORMS
    from score171 import select_manifest

    base, seal, old = verify()
    out = HERE / 'results/edge'
    assert not (out / 'SCORE_SEAL.json').exists()
    inputs = {str(p.relative_to(HERE)): sha(p) for p in (Path(__file__).resolve(), base / 'SEAL.json')}
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / 'SCORE_LOCK.json', inputs)
    rows, evidence = [], []
    errors = []
    seen = set()
    for cohort, records in select_manifest().items():
        alias = {}
        for entry, blind, key in records:
            rec = entry['recording']
            alias.setdefault(rec, f"{cohort[:3]}_recording_{len(alias)+1:02d}")
            for source in PLATFORMS:
                if source == entry['target']:
                    continue
                name = f"{cohort}/{entry['case']:02d}_{source}.npz"
                assert name not in seen
                seen.add(name)
                r, e, err = per_case(base, old, cohort, entry, blind, key, source, alias[rec])
                rows.extend(r)
                evidence.append(e)
                errors.append(err)
    assert seen == seal['files'].keys() and len(rows) == 168 * len(ARMS)
    assert sum(e['known_dual'] for e in evidence) == 60
    stats, contrasts = summaries(rows)
    for name, content in [('rows', rows), ('metrics', stats), ('contrasts', contrasts), ('evidence', evidence)]:
        csv_atomic(out / (name+'.csv'), content)
    atomic_json(out / 'SCORE_SEAL.json', dict(
        files={p.name: sha(p) for p in out.glob('*.csv')}, records=len(evidence),
        reasons=dict(Counter(e['reason'] for e in evidence)), max_iou_sensor_recheck_error=max(errors),
        physical_frames=len({(r['cohort'], r['frame']) for r in evidence}),
        audited_dual_rows=sum(r['known_dual'] for r in evidence), protected_data_access=False))
    print('Sealed scoring', len(evidence), Counter(e['reason'] for e in evidence), flush=True)


if __name__ == '__main__':
    main()
