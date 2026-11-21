"""Score the sealed paired predictions and the unused audit rays."""

import csv

import numpy as np

from predict_proxy import (ARMS, BASE, HERE, OABC, OUT, PRIVATE, SOURCE_SEAL,
                          SPEC, digest, read, read_source, write)
from evaluate_surface import groups as panel_groups, macro, stats
from feasible_geometry import CameraRays, rotation
from score import canonical_box, score_boxes, truth_for_slot
from score_confirmation import cluster_interval, paired_stats
from score_region import save
from shape_proxy import QuarticRays
from surface_grounding import read_panel


DIAGNOSTIC = 'C_box_yaw'
COMPARISONS = (('C', 'B'), ('C', 'A'), ('B', 'A'))


def audit_observation(entry, source):
    rays = np.asarray(source['audit']).reshape(-1, 3)
    assert len(rays) == entry['audit_count']
    if not len(rays):
        return {}
    transform = np.asarray(source['transform'])
    cap = max(.1, .5 * min(source['dims'][:2]))
    raw = {}
    for arm, surface in (('B', CameraRays), ('C', QuarticRays)):
        box = np.asarray(entry['boxes'][arm])
        camera = surface(rays, transform, box[6], box[3:6], cap)
        depth, hit, _ = (x[0] for x in camera.entries(box[:3] @ rotation(box[6])))
        raw[arm] = hit, np.abs(depth - camera.range)
    common = raw['B'][0] & raw['C'][0]
    return dict(audit_rays=len(rays), audit_common_hits=int(common.sum()),
        **{f'{arm}_audit_hits': int(hit.sum()) for arm, (hit, _) in raw.items()},
        **{f'{arm}_audit_abs_capped_sum_m': float(np.where(hit, np.minimum(residual, cap), cap).sum())
           for arm, (hit, residual) in raw.items()},
        **{f'{arm}_audit_common_abs_sum_m': float(residual[common].sum())
           for arm, (_, residual) in raw.items()})


def score():
    seal = read(OUT / 'prediction_seal.json')
    path = PRIVATE / 'predictions.json'
    assert (digest(path), digest(SPEC), digest(SOURCE_SEAL), digest(BASE / 'private/panel.json')) == (
        seal['predictions'], seal['protocol'], seal['source_seal'], seal['panel'])
    assert all(digest(HERE / name) == value for name, value in seal['implementation'].items())
    assert not seal['target_labels_used'] and not seal['FINAL_access'] and seal['retained_withheld_raw'] == 0
    _, source = read_source()
    panel, estimates = read_panel(), read(path)
    with (OABC / 'results/rows.csv').open(newline='') as f:
        old = {r['uid']: r for r in csv.DictReader(f)}
    assert set(old) == set(source) == set(estimates) == {r['uid'] for r in panel}
    rows, iou_gap, parity_gap = [], 0., 0.
    for frame in panel:
        uid = frame['uid']
        result, geometry = estimates[uid], source[uid]
        assert result['boxes']['A'] == geometry['boxes']['A']
        assert result['boxes']['B'] == geometry['boxes']['plain_pair']
        truth = canonical_box(truth_for_slot(frame, read(frame['calibration'])['ground_info']))
        raw_boxes = dict(result['boxes'])
        if result['used']:
            raw_boxes[DIAGNOSTIC] = result['box_pose_proxy']
            np.testing.assert_array_equal(raw_boxes['B'][3:6], raw_boxes['C'][3:6])
            np.testing.assert_array_equal(raw_boxes['B'][3:7], raw_boxes[DIAGNOSTIC][3:7])
        boxes = {name: canonical_box(value) for name, value in raw_boxes.items() if value is not None}
        scores = dict.fromkeys(ARMS, 0.)
        if boxes:
            values, gap = score_boxes(np.stack(list(boxes.values())), truth)
            scores.update(zip(boxes, map(float, values), strict=True))
            iou_gap = max(iou_gap, gap)
        parity_gap = max(parity_gap, abs(scores['A'] - float(old[uid]['A'])),
                         abs(scores['B'] - float(old[uid]['plain_pair'])))
        info = {k: frame[k] for k in ('uid', 'target', 'panel', 'recording', 'separation')}
        info.update(scores)
        info.update({f'{name}_center_m': float(np.linalg.norm(box[:3] - truth[:3]))
                     for name, box in boxes.items()})
        info.update(fitted=result['used'], status=result['status'])
        if geometry['fit_node']:
            support = np.asarray(geometry['support'])
            local = np.abs((support - truth[:3]) @ rotation(truth[6]))
            info['support_inside_gt_fraction'] = float((local <= truth[3:6] / 2. + .05).all(1).mean())
        if result['used']:
            info.update(box_pose=result['box_pose'], shape_pose=result['shape_pose'],
                        fit_count=result['fit_count'], selection_count=result['selection_count'],
                        center_shift_m=float(np.linalg.norm(boxes['C'][:3] - boxes['B'][:3])))
            for arm in ('B', 'C'):
                info[f'{arm}_fit_loss_m'] = result['fit_loss'][arm]
                info[f'{arm}_selection_loss_m'] = result['select_loss'][arm]
            info.update(audit_observation(result, geometry))
        else:
            assert raw_boxes['B'] == raw_boxes['C']
        rows.append(info)
    assert iou_gap < 1.e-8 and parity_gap < 1.e-8
    assert sum(r['fitted'] for r in rows) == seal['matched_fit_and_selection_replayed']
    return rows, dict(max_iou_scorer_gap=iou_gap, max_prior_iou_gap=parity_gap,
                      prior_iou_values_matched=2 * len(rows))


def groups(rows):
    yield ('all', 'macro'), rows
    for platform in dict.fromkeys(r['target'] for r in rows):
        yield ('all', platform), [r for r in rows if r['target'] == platform]
    yield from panel_groups(rows)


def summarize(rows):
    measures, paired, physics, intervals = [], [], [], {}
    for (panel, platform), panel_rows in groups(rows):
        fit = [r for r in panel_rows if r['fitted']]
        support = [r for r in fit if r['support_inside_gt_fraction'] >= .8]
        for scope, group in (('all', panel_rows), ('matched_fit', fit), ('high_support', support)):
            if not group:
                continue
            tag = dict(panel=panel, target=platform, scope=scope, n=len(group))
            fit_scope = scope != 'all'
            names = (*ARMS, DIAGNOSTIC) if fit_scope else ARMS
            for arm in names:
                values = [r[f'{arm}_center_m'] for r in group if f'{arm}_center_m' in r]
                measures.append(dict(**tag, arm=arm, recordings=len({r['recording'] for r in group}),
                    fitted=sum(r['fitted'] for r in group), **stats(group, arm), center_n=len(values),
                    center_median_m=float(np.median(values)) if values else None))
            comparisons = (*COMPARISONS, (DIAGNOSTIC, 'B')) if fit_scope else COMPARISONS
            for a, b in comparisons:
                common = [r for r in group if f'{a}_center_m' in r and f'{b}_center_m' in r]
                differences = np.array([r[f'{a}_center_m'] - r[f'{b}_center_m'] for r in common])
                paired.append(dict(**tag, arm=a, reference=b, **paired_stats(group, a, b),
                    center_pairs=len(common), center_closer=int(np.sum(differences < -1.e-10)),
                    center_farther=int(np.sum(differences > 1.e-10)),
                    mean_center_delta_m=macro(differences, common) if common else None,
                    median_center_delta_m=float(np.median(differences)) if common else None))
                if a == 'C' and scope == 'all':
                    intervals[f'{panel}/{platform}/{a}-{b}'] = cluster_interval(group, a, b)
            if not fit_scope:
                continue
            n_rays = sum(r.get('audit_rays', 0) for r in group)
            n_common = sum(r.get('audit_common_hits', 0) for r in group)
            for arm in ('B', 'C'):
                physics.append(dict(**tag, arm=arm,
                    fit_mean_depth_cost_m=macro([r[f'{arm}_fit_loss_m'] for r in group], group),
                    selection_mean_depth_cost_m=macro([r[f'{arm}_selection_loss_m'] for r in group], group),
                    audit_rays=n_rays, audit_hits=sum(r.get(f'{arm}_audit_hits', 0) for r in group),
                    audit_common=n_common,
                    audit_mean_capped_m=sum(r.get(f'{arm}_audit_abs_capped_sum_m', 0.) for r in group) /
                       n_rays if n_rays else None,
                    audit_common_hit_mae_m=sum(r.get(f'{arm}_audit_common_abs_sum_m', 0.) for r in group) /
                       n_common if n_common else None,
                    changed_yaw=sum(r['box_pose'] != r['shape_pose'] for r in group),
                    center_shift_median_m=float(np.median([r['center_shift_m'] for r in group]))))
    return measures, paired, physics, intervals


def run():
    rows, checks = score()
    metrics, paired, physics, intervals = summarize(rows)
    for name, data in dict(rows=rows, metrics=metrics, paired=paired, ray_audit=physics).items():
        save(OUT / f'{name}.csv', data)
    write(OUT / 'evaluation_receipt.json', dict(prediction_seal=digest(OUT / 'prediction_seal.json'),
        evaluation_code=digest(__file__), frames=len(rows), checks=checks,
        results={f: digest(OUT / f'{f}.csv') for f in ('rows', 'metrics', 'paired', 'ray_audit')},
        recording_intervals=intervals, target_labels_after_seal=True, FINAL_access=False))
    print('Scored', len(rows), 'frames.', checks, flush=True)


if __name__ == '__main__':
    run()
