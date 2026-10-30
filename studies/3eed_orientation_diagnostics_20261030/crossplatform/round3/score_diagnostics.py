"""Offline diagnostic scoring after GT-free P2/P3 seals; never an adaptation input."""
import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation
from scipy.stats import rankdata, spearmanr

R = Path(__file__).resolve().parent
OLD = R.parent / 'round2'
sys.path.insert(0, str(R.parents[1] / 'tools'))
from audit_round2_3eed import iou


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def metric(box, gt):
    return iou(np.asarray(box, dtype=float), np.asarray(gt, dtype=float))


def target(sample):
    meta = json.loads(Path(sample['meta_path']).read_text())
    gt = np.asarray(meta['ground_info'][sample['object_index']]['bbox_3d'][:7], dtype=float)
    if sample['platform'] == 'drone':
        gt[2] += 1.8
    if sample['platform'] == 'waymo':
        angle = [0, -45, 45, -90, 90][sample['Waymo_view_suffix']]
        gt[:3] = Rotation.from_euler('z', angle, degrees=True).apply(gt[:3])
        gt[6] += np.deg2rad(angle)
    return gt


def orientation_diagnostics(source, gt, feasible_yaws):
    injected = source.copy(); injected[6] = gt[6]
    angles = np.arange(720, dtype=float) * np.pi / 720
    box = source.copy()
    def objective(theta):
        box[6] = float(theta)
        return -metric(box, gt)
    values = np.array([-objective(a) for a in angles])
    feasible = [(float(t), -objective(t)) for t in [float(source[6]), float(gt[6]), *feasible_yaws]]
    local = [j for j in range(720) if values[j] >= values[(j - 1) % 720] and values[j] >= values[(j + 1) % 720]]
    # No refinement on a fully flat zero-overlap curve; it carries no yaw information.
    selected = sorted(local, key=lambda j: (-values[j], j))[:3] if np.ptp(values) > 1e-12 else []
    for j in selected:
        opt = minimize_scalar(objective, bounds=(angles[j] - np.pi / 720, angles[j] + np.pi / 720),
                              method='bounded', options={'xatol': 1e-9, 'maxiter': 100})
        feasible.append((float(opt.x), float(-opt.fun)))
    j = int(values.argmax()); feasible.append((float(angles[j]), float(values[j])))
    best_yaw, best_value = max(feasible, key=lambda v: v[1])
    best = source.copy(); best[6] = best_yaw
    gt_value = metric(injected, gt)
    assert best_value + 1e-10 >= gt_value
    overlap_z = max(0., min(source[2] + source[5] / 2, gt[2] + gt[5] / 2)
                    - max(source[2] - source[5] / 2, gt[2] - gt[5] / 2))
    intersection_cap = min(np.prod(source[3:5]), np.prod(gt[3:5])) * overlap_z
    upper = float(intersection_cap / (np.prod(source[3:6]) + np.prod(gt[3:6]) - intersection_cap))
    assert best_value <= upper + 1e-7
    return injected, best, {'grid_points': 720, 'local_refinements': len(selected),
                            'best_observed_iou': best_value, 'GT_yaw_injection_iou': gt_value,
                            'certified_but_loose_volume_height_iou_upper_bound': upper,
                            'best_observed_not_certified_upper_bound': True}


def statistics(rows, arms):
    platforms = ['waymo', 'drone', 'quad']
    summaries = []
    comparisons = []
    scenes = sorted(set((r['platform'], r['scene']) for r in rows))
    for arm in arms:
        values = [r for r in rows if r['arm'] == arm]
        for platform in ['all', *platforms]:
            subset = [r for r in values if platform == 'all' or r['platform'] == platform]
            summaries.append({'arm': arm, 'platform': platform, 'n': len(subset),
                              'mean_iou': float(np.mean([r['iou'] for r in subset])),
                              'acc25_count': sum(r['iou'] > .25 for r in subset),
                              'acc50_count': sum(r['iou'] > .5 for r in subset),
                              'scene_clusters': len(set((r['platform'], r['scene']) for r in subset))})
        grouped = []
        for platform, scene in scenes:
            sr = [r for r in values if (r['platform'], r['scene']) == (platform, scene)]
            grouped.append({'platform': platform, 'scene': scene, 'n': len(sr),
                            'mean_iou': float(np.mean([r['iou'] for r in sr])),
                            'mean_delta': float(np.mean([r['delta_source'] for r in sr]))})
        by_platform = {p: np.array([s['mean_delta'] for s in grouped if s['platform'] == p]) for p in platforms}
        estimate = float(np.mean([v.mean() for v in by_platform.values()]))
        rng = np.random.default_rng(20261030)
        boots = np.mean([rng.choice(v, size=(10000, len(v)), replace=True).mean(1) for v in by_platform.values()], axis=0)
        loo = []
        for p, v in by_platform.items():
            for j in range(len(v)):
                if len(v) > 1:
                    loo.append(float(np.mean([np.delete(v, j).mean() if q == p else w.mean() for q, w in by_platform.items()])))
        d = np.array([r['delta_source'] for r in values])
        comparisons.append({'arm': arm, 'frameweighted_mean_delta': float(d.mean()),
                            'equalplatform_scene_macro_delta': estimate,
                            'platform_stratified_scene_bootstrap_95ci_descriptive': np.percentile(boots, [2.5, 97.5]).tolist(),
                            'leave_one_scene_out_range': [min(loo), max(loo)],
                            'improved': int((d > 1e-10).sum()), 'harmed': int((d < -1e-10).sum()),
                            'equal': int((abs(d) <= 1e-10).sum()), 'scene_groups': grouped,
                            'tails': sorted([{'id': r['id'], 'platform': r['platform'], 'delta': r['delta_source']} for r in values], key=lambda r: r['delta'])})
    return summaries, comparisons


def association(rows, features, arms):
    feature_names = sorted({k for r in features for k, v in r['features'].items()
                            if k not in ['valid_source_box', 'reason'] and (v is None or isinstance(v, (int, float)))})
    by_id = {r['id']: r for r in features}
    output = []
    for arm in arms:
        subset = [r for r in rows if r['arm'] == arm]
        for name in feature_names:
            available = [(float(by_id[r['id']]['features'][name]), r) for r in subset
                         if by_id[r['id']]['features'].get(name) is not None]
            x = np.array([v for v, r in available]); y = np.array([r['delta_source'] for v, r in available])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                rho = spearmanr(x, y).statistic if len(x) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0 else None
            changed = abs(y) > 1e-10; labels = y[changed] > 0; score = x[changed]
            positive = int(labels.sum()); negative = int((~labels).sum())
            auc = None
            if positive and negative:
                auc = float((rankdata(score)[labels].sum() - positive * (positive + 1) / 2) / (positive * negative))
            scene_auc = []
            for platform, scene in sorted(set((r['platform'], r['scene']) for _, r in available)):
                ss = [(v, r['delta_source']) for v, r in available if (r['platform'], r['scene']) == (platform, scene) and abs(r['delta_source']) > 1e-10]
                lab = np.array([d > 0 for _, d in ss], dtype=bool); np_ = int(lab.sum()); nn_ = len(lab) - np_
                val = float((rankdata([v for v, _ in ss])[lab].sum() - np_ * (np_ + 1) / 2) / (np_ * nn_)) if np_ and nn_ else None
                scene_auc.append({'platform': platform, 'scene': scene, 'n_changed': len(ss), 'AUROC': val})
            output.append({'arm': arm, 'feature': name, 'available_n': len(x), 'missing_n': len(subset) - len(x),
                           'positive_n': positive, 'negative_n': negative, 'unchanged_n': int((~changed).sum()),
                           'higher_feature_AUROC_no_direction_flip': auc,
                           'spearman_vs_delta': float(rho) if rho is not None and np.isfinite(rho) else None,
                           'within_scene_AUROC': scene_auc})
    return output


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--predictions', type=Path, required=True); a = ap.parse_args()
    output = R / 'diagnostics'; output.mkdir(exist_ok=False)
    p3seal_path = a.predictions / 'SEALED.json'
    if not p3seal_path.is_file():
        p3seal_path = a.predictions / 'sealed.json'
    seal = json.loads(p3seal_path.read_text())
    p2 = R / 'p2'; fs = json.loads((p2 / 'SEALED.json').read_text())
    assert sha(R / 'PLAN_DIAGNOSTIC_LOCK.json') == (R / 'PLAN_DIAGNOSTIC_LOCK.sha256').read_text().strip()
    for filename, digest in fs['files'].items():
        assert sha(p2 / filename) == digest
    for filename, digest in seal['files'].items():
        assert sha(a.predictions / filename) == digest
    assert fs['n'] == seal['n'] == 54
    source_key = 'source' if 'source' in seal['arms'] else 'source_yaw0'
    arms = ['source' if name == source_key else name for name in seal['arms']]
    assert 'source' in arms
    input_receipt = json.loads((OLD / 'INPUT_PREPARATION_RECEIPT.json').read_text())
    frozen_export = R.parents[1] / 'research/round2/public_export'
    frozen_manifest = json.loads((frozen_export / 'MANIFEST.json').read_text())
    export_entries = {entry['path']: entry for entry in frozen_manifest['files']}
    historical = {}
    for cohort in ['dev', 'confirm']:
        index = OLD / (cohort + '_scoring_index_PRIVATE.json')
        assert sha(index) == input_receipt['cohorts'][cohort]['scoring_index_sha256']
        previous_score = OLD / (cohort + '_scored_double.json')
        frozen_name = 'audits/3EED_' + cohort.upper() + '_V1_INDEPENDENT.json'
        assert sha(frozen_export / frozen_name) == export_entries[frozen_name]['sha256']
        frozen_scores = json.loads((frozen_export / frozen_name).read_text())
        expected = {row['id']: row['iou'] for row in frozen_scores['rows'] if row['arm'] == 'source'}
        old = json.loads(previous_score.read_text())
        assert len(expected) == (18 if cohort == 'dev' else 36)
        assert max(abs(row['iou'] - expected[row['id']]) for row in old['rows'] if row['arm'] == 'source') < 1e-6
        historical[cohort] = {'index_sha256': sha(index), 'score_sha256': sha(previous_score)}
    code_lock = {'scorer_sha256': sha(Path(__file__)), 'polygon_iou_sha256': sha(R.parents[1] / 'tools/audit_round2_3eed.py'),
                 'plan_sha256': sha(R / 'PLAN_DIAGNOSTIC_LOCK.json'), 'p2_seal_sha256': sha(p2 / 'SEALED.json'),
                 'p3_seal_sha256': sha(p3seal_path), 'historical_input_hashes': historical,
                 'before_numeric_target_access': True}
    (output / 'SCORER_LOCK.json').write_text(json.dumps(code_lock, indent=2) + '\n')
    # Only after complete GT-free prediction/feature seals: open numeric target metadata.
    features_by_id = {v['id']: v for v in json.loads((p2 / 'FEATURES_PRIVATE.json').read_text())['rows']}
    rows = []; oracle_boxes = {}; reference_checks = []; occupancy = []; perturbation_rows = []
    for cohort in ['dev', 'confirm']:
        samples = json.loads((OLD / (cohort + '_scoring_index_PRIVATE.json')).read_text())
        inputs = {s['id']: s for s in json.loads((OLD / 'inputs' / (cohort + '_manifest.json')).read_text())}
        old_scores = json.loads((OLD / (cohort + '_scored_double.json')).read_text())
        old_source = {r['id']: r['iou'] for r in old_scores['rows'] if r['arm'] == 'source'}
        for s in samples:
            gt = target(s)
            with np.load(a.predictions / (s['id'] + '.npz'), allow_pickle=False) as z:
                boxes = {('source' if arm == source_key else arm): z[arm].astype(float) for arm in seal['arms']}
            source = boxes['source']; base = metric(source, gt)
            reference_checks.append(abs(base - old_source[s['id']]))
            injected, best, trace = orientation_diagnostics(source, gt, [b[6] for b in boxes.values() if np.array_equal(b[:6], source[:6])])
            boxes.update(GT_yaw_injection=injected, finite_yaw_search_best=best)
            oracle_boxes[s['id']] = {'GT_yaw_injection': injected, 'finite_yaw_search_best': best}
            probe = features_by_id[s['id']]['trace']
            probe_yaw = probe.get('PCA_yaw')
            probe_box = source.copy(); probe_box[6] = probe_yaw if probe_yaw is not None else source[6]
            probe_iou = metric(probe_box, gt)
            for field in ['bootstrap_yaws', 'partial_visibility_yaws', 'expanded_yaws', 'background_yaws', 'shifted_yaws']:
                for j, theta in enumerate(probe.get(field, [])):
                    pb = source.copy(); pb[6] = theta if theta is not None else source[6]
                    value = metric(pb, gt)
                    perturbation_rows.append({'id': s['id'], 'platform': s['platform'], 'scene': s['scene'],
                                              'perturbation': field, 'variant_index': j,
                                              'valid_PCA_yaw': theta is not None, 'sourcefallback_if_invalid': theta is None,
                                              'probe_min_unique_points': 3, 'unperturbed_probe_iou': probe_iou,
                                              'perturbed_iou': value, 'delta_unperturbed_probe': value - probe_iou})
            for arm, box in boxes.items():
                value = metric(box, gt)
                rows.append({'id': s['id'], 'cohort': cohort, 'platform': s['platform'], 'scene': s['scene'],
                             'arm': arm, 'iou': value, 'source_iou': base, 'delta_source': value - base,
                             'orientation_only_capacity': bool(np.array_equal(box[:6], source[:6])),
                             'oracle_GT_used_only_offline': arm in ['GT_yaw_injection', 'finite_yaw_search_best']})
            pc = np.unique(np.load(inputs[s['id']]['point_path'], allow_pickle=False).astype(float), axis=0)
            c, sn = np.cos(gt[6]), np.sin(gt[6]); rot = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]])
            inside = np.all(abs((pc - gt[:3]) @ rot) <= gt[3:6] / 2 + 1e-7, axis=1)
            crop = np.all(abs(pc - source[:3]) <= source[3:6] / 2 + 1e-7, axis=1)
            occupancy.append({'id': s['id'], 'GT_box_point_occupancy_fraction_in_sourcecrop': float(inside[crop].mean()) if crop.any() else None,
                              'GT_box_point_retention_by_sourcecrop': float(crop[inside].mean()) if inside.any() else None,
                              'GT_inside_sampled_point_count': int(inside.sum()), 'sourcecrop_sampled_point_count': int(crop.sum()),
                              'proxy_not_instance_segmentation_or_observability_groundtruth': True, 'search': trace})
    assert len(reference_checks) == 54 and max(reference_checks) < 1e-6
    allarms = [*arms, 'GT_yaw_injection', 'finite_yaw_search_best']
    summary, compared = statistics(rows, allarms)
    f = json.loads((p2 / 'FEATURES_PRIVATE.json').read_text())['rows']
    regular = [arm for arm in arms if arm != 'source' and not arm.startswith('legacy_')]
    stats = association(rows, f, regular)
    for sid, data in oracle_boxes.items():
        np.savez_compressed(output / (sid + '_ORACLE_PRIVATE.npz'), **data)
    report = {'scope': 'Historical54unioncachedparents; descriptive posthoc P1-P3, not sourceonly or newconfirmation',
              'n': 54, 'scorer_lock_sha256': sha(output / 'SCORER_LOCK.json'),
              'source_max_difference_vs_frozen_round2': max(reference_checks), 'arms': allarms,
              'rows': rows, 'summaries': summary, 'comparisons': compared,
              'P2_all_predeclared_associations': stats, 'GT_geometry_proxies_offline_only': occupancy,
              'P2_controlled_perturbation_rows': perturbation_rows,
              'limits': ['GTyaw replacement not IoUmax upperbound', 'finite yawsearch is feasible lowerbound not certified maximum',
                         'Old54labels andscenes exposed; no newheldoutmethodclaim', 'Drone2Quad3scenecluster intervals descriptive',
                         'No selected reliabilityclassifier; no targetlabels in geometric optimization']}
    (output / 'RESULTS_PRIVATE.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (output / 'SEALED.json').write_text(json.dumps({'n': 54, 'files': {p.name: sha(p) for p in output.iterdir() if p.is_file()},
                                                   'numeric_GT_used_only_after_predictionseal': True}, indent=2) + '\n')
    print(json.dumps({'n': 54, 'summaries': [x for x in summary if x['platform'] == 'all'],
                      'source_max_difference': max(reference_checks)}, indent=2))


if __name__ == '__main__':
    main()
