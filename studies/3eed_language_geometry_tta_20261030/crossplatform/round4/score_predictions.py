"""Separate-process CPU task scoring, with cohort-wide prediction seals first."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

R = Path(__file__).resolve().parent; S = R / 'source'
sys.path.insert(0, str(R.parents[1] / 'tools'))
from audit_round2_3eed import iou


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def summarize(rows, arms):
    output = []; comparisons = []
    for platform in ['waymo', 'drone', 'quad']:
        for arm in arms:
            part = [r for r in rows if r['platform'] == platform and r['arm'] == arm]
            scenes = sorted(set(r['scene'] for r in part))
            groups = [{'scene': sc, 'n': sum(r['scene'] == sc for r in part),
                       'mean_delta': float(np.mean([r['delta_frozen_PCA'] for r in part if r['scene'] == sc])),
                       'mean_iou': float(np.mean([r['iou'] for r in part if r['scene'] == sc]))} for sc in scenes]
            rng = np.random.default_rng(20261030)
            means = np.array([g['mean_delta'] for g in groups])
            boot = rng.choice(means, size=(10000, len(means)), replace=True).mean(1)
            output.append({'platform': platform, 'split': part[0]['split'], 'arm': arm, 'n': len(part),
                'scene_clusters': len(scenes), 'mean_iou': float(np.mean([r['iou'] for r in part])),
                'acc25_count': sum(r['iou'] > .25 for r in part), 'acc50_count': sum(r['iou'] > .5 for r in part),
                'invalid_count': sum(r['invalid'] for r in part)})
            delta = np.array([r['delta_frozen_PCA'] for r in part])
            comparisons.append({'platform': platform, 'split': part[0]['split'], 'arm': arm,
                'mean_delta_frozen_PCA': float(delta.mean()), 'scene_macro_delta': float(means.mean()),
                'scene_cluster_bootstrap_95ci_descriptive': np.percentile(boot, [2.5, 97.5]).tolist() if len(means) >= 2 else None,
                'single_cluster_no_population_CI': len(means) < 2,
                'positive': int((delta > 1e-10).sum()), 'negative': int((delta < -1e-10).sum()),
                'same': int((abs(delta) <= 1e-10).sum()),
                'rescue25': sum(r['iou'] > .25 and r['frozen_PCA_iou'] <= .25 for r in part),
                'harm25': sum(r['iou'] <= .25 and r['frozen_PCA_iou'] > .25 for r in part),
                'rescue50': sum(r['iou'] > .5 and r['frozen_PCA_iou'] <= .5 for r in part),
                'harm50': sum(r['iou'] <= .5 and r['frozen_PCA_iou'] > .5 for r in part),
                'scene_groups': groups, 'all_signed_tails': sorted(part, key=lambda r: r['delta_frozen_PCA'])})
    return output, comparisons


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cohort', choices=['dev', 'confirm'], required=True)
    ap.add_argument('--predictions', type=Path, required=True); ap.add_argument('--output', type=Path, required=True); a = ap.parse_args()
    sealpath = a.predictions / 'sealed.json'
    if not sealpath.exists():
        sealpath = a.predictions / 'SEALED.json'
    seal = json.loads(sealpath.read_text())
    for name, digest in seal['files'].items():
        assert sha(a.predictions / name) == digest
    source = S / (a.cohort + '_source'); ss = json.loads((source / 'sealed.json').read_text())
    for name, digest in ss['files'].items():
        assert sha(source / name) == digest
    index = S / (a.cohort + '_scoring_index_PRIVATE.json')
    receipt = json.loads((S / 'INPUT_RECEIPT.json').read_text())
    assert sha(index) == receipt['cohorts'][a.cohort]['scoring_index_sha256']
    if a.cohort == 'confirm':
        master = json.loads((R / 'CONFIRMATION_ALL_ARMS_SEALED.json').read_text())
        assert master['all_confirm_arms_sealed_before_numeric_GT']
        assert master['source_seal_sha256'] == sha(source / 'sealed.json')
        assert sha(sealpath) in master['prediction_seals'].values()
        for path, digest in master['all_sealed_files_to_verify'].items():
            assert sha(Path(path)) == digest
    a.output.mkdir(exist_ok=False)
    (a.output / 'SCORER_LOCK.json').write_text(json.dumps({'scorer_sha256': sha(Path(__file__)),
        'polygon_iou_sha256': sha(R.parents[1] / 'tools/audit_round2_3eed.py'), 'prediction_seal_sha256': sha(sealpath),
        'source_seal_sha256': sha(source / 'sealed.json'), 'scoring_index_sha256': sha(index),
        'all_cohort_source_and_new_prediction_hashes_verified_before_numeric_GT': True}, indent=2) + '\n')
    samples = json.loads(index.read_text()); assert len(samples) == ss['n_samples']
    assert seal.get('n', seal.get('n_samples')) == len(samples)
    rows = []
    for sample in samples:
        meta = json.loads(Path(sample['meta_path']).read_text())
        gt = np.asarray(meta['ground_info'][sample['object_index']]['bbox_3d'][:7], dtype=float)
        if sample['platform'] == 'drone': gt[2] += 1.8
        if sample['platform'] == 'waymo':
            angle = [0, -45, 45, -90, 90][sample['waymo_view_suffix']]
            gt[:3] = Rotation.from_euler('z', angle, degrees=True).apply(gt[:3]); gt[6] += np.deg2rad(angle)
        with np.load(source / (sample['id'] + '_v0_full.npz'), allow_pickle=False) as z:
            q = int(z['semantic_scores'].argmax()); reference = z['boxes_pca'][q].astype(float)
        base = float(iou(reference, gt))
        with np.load(a.predictions / (sample['id'] + '.npz'), allow_pickle=False) as z:
            for arm in seal['arms']:
                box = z[arm].astype(float); assert box.shape == (7,)
                invalid = bool(not np.isfinite(box).all() or np.any(box[3:6] <= 0))
                value = 0. if invalid else float(iou(box, gt))
                rows.append({'id': sample['id'], 'platform': sample['platform'], 'split': sample['split'], 'scene': sample['scene'],
                             'arm': arm, 'iou': value, 'frozen_PCA_iou': base, 'delta_frozen_PCA': value - base, 'invalid': invalid})
    summaries, compared = summarize(rows, seal['arms'])
    result = {'cohort': a.cohort, 'n': len(samples), 'arms': seal['arms'], 'rows': rows, 'summaries': summaries,
              'comparisons_vs_frozen_PCA': compared, 'prediction_seal_sha256': sha(sealpath),
              'scope': 'PrimaryWaymoVAL andTRAIN-exposedDroneQuad separate. No poolheadline. Clusterintervals descriptive.'}
    (a.output / 'RESULTS_PRIVATE.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'cohort': a.cohort, 'summaries': [s for s in summaries if s['platform'] == 'waymo']}, indent=2))


if __name__ == '__main__':
    main()
