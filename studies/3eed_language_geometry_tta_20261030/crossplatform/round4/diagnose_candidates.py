"""Offline candidate-coverage diagnostic, after all cohort source/PCA seals.

No labels, oracle indices or IoUs are available to the adaptation pipeline.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

R = Path(__file__).resolve().parent
S = R / 'source'
sys.path.insert(0, str(R.parents[1] / 'tools'))
from audit_round2_3eed import iou


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cohort', choices=['dev', 'confirm'], required=True); args = ap.parse_args()
    cache = S / (args.cohort + '_source'); sealpath = cache / 'sealed.json'; seal = json.loads(sealpath.read_text())
    for name, digest in seal['files'].items():
        assert sha(cache / name) == digest
    plan = json.loads((S / 'SOURCE_PLAN.json').read_text())
    assert sha(S / 'SOURCE_PLAN.json') == seal['source_plan_sha256']
    for rel, digest in plan['local_code_sha256'].items():
        assert sha(S / rel) == digest
    index = S / (args.cohort + '_scoring_index_PRIVATE.json')
    receipt = json.loads((S / 'INPUT_RECEIPT.json').read_text())
    assert sha(index) == receipt['cohorts'][args.cohort]['scoring_index_sha256']
    manifest = S / 'inputs' / (args.cohort + '_manifest.json')
    assert sha(manifest) == seal['manifest_sha256'] == receipt['cohorts'][args.cohort]['manifest_sha256']
    if args.cohort == 'confirm':
        # Source predictions alone cannot unlock confirmation numeric labels.
        release = json.loads((R / 'CONFIRMATION_ALL_ARMS_SEALED.json').read_text())
        assert release['all_confirm_arms_sealed_before_numeric_GT']
        assert release['source_seal_sha256'] == sha(sealpath)
    out = R / (args.cohort + '_candidate_diagnostic'); out.mkdir(exist_ok=False)
    lock = {'source_seal_sha256': sha(sealpath), 'source_plan_sha256': sha(S / 'SOURCE_PLAN.json'),
            'scorer_sha256': sha(Path(__file__)), 'polygon_iou_sha256': sha(R.parents[1] / 'tools/audit_round2_3eed.py'),
            'scoring_index_sha256': sha(index), 'all_source_files_verified_before_numeric_GT': True}
    (out / 'SCORER_LOCK.json').write_text(json.dumps(lock, indent=2) + '\n')
    samples = json.loads(index.read_text()); rows = []
    assert len(samples) == seal['n_samples']
    for sample in samples:
        # First access to numeric targets in this process, after every cache seal.
        meta = json.loads(Path(sample['meta_path']).read_text())
        gt = np.asarray(meta['ground_info'][sample['object_index']]['bbox_3d'][:7], dtype=float)
        if sample['platform'] == 'drone':
            gt[2] += 1.8
        if sample['platform'] == 'waymo':
            angle = [0, -45, 45, -90, 90][sample['waymo_view_suffix']]
            gt[:3] = Rotation.from_euler('z', angle, degrees=True).apply(gt[:3]); gt[6] += np.deg2rad(angle)
        with np.load(cache / (sample['id'] + '_v0_full.npz'), allow_pickle=False) as z:
            raw = np.c_[z['boxes'].astype(float), np.zeros(256)]
            pca = z['boxes_pca'].astype(float); score = z['semantic_scores'].astype(float)
        assert np.array_equal(raw[:, :6], pca[:, :6])
        order = np.argsort(-score, kind='stable'); selected = int(order[0])
        values = {'raw_yaw0': np.array([iou(box, gt) for box in raw]),
                  'fixed_PCA': np.array([iou(box, gt) for box in pca])}
        interventions = {}
        for component, slices in [('center', slice(0, 3)), ('size', slice(3, 6)), ('yaw', slice(6, 7))]:
            box = pca[selected].copy(); box[slices] = gt[slices]
            interventions[component] = float(iou(box, gt))
        np.savez_compressed(out / (sample['id'] + '_IOU_PRIVATE.npz'), **values)
        metrics = {}
        for geometry, v in values.items():
            metrics[geometry] = {'selected_iou': float(v[selected]), 'best_candidate_iou': float(v.max()),
                                 'rank_best_IoU_query': int(v.argmax()),
                                 'topk': {str(k): {str(t): bool(np.any(v[order[:k]] > t)) for t in [.25, .5]}
                                          for k in [1, 5, 10, 32, 256]},
                                 'rank_recoverable': {str(t): bool(v[selected] <= t and np.any(v > t)) for t in [.25, .5]},
                                 'coverage_failure': {str(t): bool(not np.any(v > t)) for t in [.25, .5]}}
        rows.append({'id': sample['id'], 'platform': sample['platform'], 'split': sample['split'], 'scene': sample['scene'],
                     'selected_query': selected, 'metrics': metrics, 'fixed_selected_PCA_single_GT_replacements': interventions,
                     'metadata_top_level_keys_not_values': sorted(meta),
                     'exact_wrong_instance_identity': 'not_inferred_from_candidate_overlap'})
    summary = []
    for platform in ['waymo', 'drone', 'quad']:
        subset = [r for r in rows if r['platform'] == platform]
        for geometry in ['raw_yaw0', 'fixed_PCA']:
            summary.append({'platform': platform, 'split': subset[0]['split'], 'geometry': geometry, 'n': len(subset),
                'scene_clusters': len(set(x['scene'] for x in subset)),
                'selected_mean_iou': float(np.mean([x['metrics'][geometry]['selected_iou'] for x in subset])),
                'best_candidate_mean_iou': float(np.mean([x['metrics'][geometry]['best_candidate_iou'] for x in subset])),
                'topk_success_counts': {str(k): {str(t): sum(x['metrics'][geometry]['topk'][str(k)][str(t)] for x in subset)
                                                for t in [.25, .5]} for k in [1, 5, 10, 32, 256]},
                'rank_recoverable_counts': {str(t): sum(x['metrics'][geometry]['rank_recoverable'][str(t)] for x in subset) for t in [.25, .5]},
                'coverage_failure_counts': {str(t): sum(x['metrics'][geometry]['coverage_failure'][str(t)] for x in subset) for t in [.25, .5]},
                'fixed_selected_PCA_single_GT_replacement_means': {c: float(np.mean([x['fixed_selected_PCA_single_GT_replacements'][c] for x in subset]))
                                                                  for c in ['center', 'size', 'yaw']}})
    result = {'cohort': args.cohort, 'n': len(rows), 'source_seal_sha256': sha(sealpath), 'scorer_lock_sha256': sha(out / 'SCORER_LOCK.json'),
              'scope': 'Unionmodel; primary research-new WaymoVAL vs TRAIN-exposed DroneQuad auxiliary reported separately. '
                       'Rank-recoverable is conditional coverage at fixed geometry/IoU, not exact instance identity.',
              'rows': rows, 'summaries': summary}
    (out / 'RESULTS_PRIVATE.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    (out / 'SEALED.json').write_text(json.dumps({'files': {p.name: sha(p) for p in out.iterdir() if p.is_file()},
        'n': len(rows), 'all_source_sealed_before_numeric_GT': True}, indent=2) + '\n')
    print(json.dumps({'cohort': args.cohort, 'summaries': summary}, indent=2))


if __name__ == '__main__':
    main()
