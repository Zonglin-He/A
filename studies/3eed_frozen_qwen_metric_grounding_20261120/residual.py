"""Descriptive post-seal check of remaining error and physical support."""
import csv
import numpy as np
from paths import OUT, PRIV, PLATFORMS, digest, read
from geometry import camera, rigid_xyz
from score import aggregate, canonical_box, save_csv, score_boxes, truth_for_slot


def main():
    predictions = read(PRIV / 'predictions.json')
    assert digest(PRIV / 'predictions.json') == read(OUT / 'geometry_seal.json')['predictions']
    with (OUT / 'all_rows.csv').open() as stream:
        scores = {r['uid']: r for r in csv.DictReader(stream)}
    records = []
    for frame in read(PRIV / 'panel.json'):
        uid = frame['uid']
        scalar = scores[uid]
        record = dict(target=frame['target'], valid=bool(predictions[uid]['sensor_boxes']),
                      fit=int(scalar['fit_bins']), purity=float(scalar['fit_inside_gt_fraction'])
                      if scalar.get('fit_inside_gt_fraction') else None)
        records.append(record)
        if not record['valid']:
            continue
        box = canonical_box(predictions[uid]['sensor_boxes']['ray_depth'])
        truth = canonical_box(truth_for_slot(frame, read(frame['calibration'])['ground_info']))
        _, _, transform = camera(frame)
        origin = rigid_xyz(box[:3], transform)
        target = rigid_xyz(truth[:3], transform)
        inverse = np.linalg.inv(transform)
        hypotheses = {'ray_depth': box}
        for component, center in [('depth', origin / origin[2] * target[2]),
                                  ('bearing', target / target[2] * origin[2]),
                                  ('center', target)]:
            candidate = box.copy()
            candidate[:3] = rigid_xyz(center, inverse)
            hypotheses[component] = candidate
        for component, cols in [('size', slice(3, 6)), ('yaw', slice(6, 7))]:
            candidate = box.copy()
            candidate[cols] = truth[cols]
            hypotheses[component] = candidate
        ious, _ = score_boxes(np.stack(list(hypotheses.values())), truth)
        record.update(dict(zip(hypotheses, ious)))
        assert abs(record['ray_depth'] - float(scalar['ray_depth'])) < 1e-7
        assert abs(record['center'] - float(scalar['gt_center'])) < 1e-7
        record['depth_error'] = abs(origin[2] - target[2])
        record['center_error_over_width'] = np.linalg.norm(box[:3] - truth[:3]) / truth[4]

    output = []
    for target in (*PLATFORMS, 'all'):
        platform = [r for r in records if target == 'all' or r['target'] == target]
        valid = [r for r in platform if r['valid']]
        groups = {'all_including_invalid': platform, 'supported': [r for r in valid if r['fit'] >= 5],
                  'insufficient_support_valid_3d': [r for r in valid if r['fit'] < 5],
                  'majority_on_target': [r for r in valid if r['fit'] >= 5 and r['purity'] >= .5],
                  'minority_on_target': [r for r in valid if r['fit'] >= 5 and r['purity'] < .5]}
        for group, rows in groups.items():
            if not rows:
                continue
            observed = [r for r in rows if r['valid']]
            metrics = ('ray_depth','depth','bearing','center','size','yaw') if group == 'all_including_invalid' else ('ray_depth',)
            for substitute in metrics:
                output.append(dict(target=target, group=group, gt_substitution=substitute,
                    n=len(rows), n_valid=len(observed),
                    median_abs_depth_m=np.median([r['depth_error'] for r in observed]),
                    median_center_error_over_gt_minor_width=np.median([r['center_error_over_width'] for r in observed]),
                    **aggregate([r.get(substitute, 0.) for r in rows])))
    save_csv(OUT / 'residual_diagnosis.csv', output)


if __name__ == '__main__':
    main()
