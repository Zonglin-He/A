"""Independent SVD yaw recovery and two-oriented-box IoU audit."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from obb_iou import oriented_iou


def sha(path):
    return hashlib.file_digest(Path(path).open('rb'), 'sha256').hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--replication36', action='store_true')
    args = ap.parse_args()
    root = args.root
    run = root / ('replication36/pca' if args.replication36 else 'runs/geometry_pca_v1')
    seal = json.loads((run / 'sealed.json').read_text())
    plan_file = root / 'replication36/PLAN_LOCK.json' if args.replication36 else run / 'registered_plan.json'
    assert sha(plan_file) == seal['plan_sha256']
    for file, expected in seal['files'].items():
        assert sha(run / file) == expected
    manifest_file = root / 'replication36/manifest.json' if args.replication36 else root / 'manifest.json'
    score_file = root / 'replication36/scored.json' if args.replication36 else run / 'scored.json'
    manifest = json.loads(manifest_file.read_text())
    scores = {x['id']: x for x in json.loads(score_file.read_text())['rows']}
    if args.replication36:
        source = root / 'replication36/source'
        assert sha(source/'sealed.json') == seal['source_seal_sha256']
        source_seal = json.loads((source/'sealed.json').read_text())
        for file, expected in source_seal['files'].items(): assert sha(source/file) == expected
    trace = {x['id']: x for x in json.loads((run / 'trace.json').read_text())}
    rows, errors = [], []
    for index, sample in enumerate(manifest):
        sid = sample['id']
        with np.load(run / f'{sid}.npz', allow_pickle=False) as data:
            source, refined = data['source7d'], data['refined7d']
        if args.replication36:
            with np.load(root/'replication36/source'/f'{sid}.npz', allow_pickle=False) as original:
                assert np.array_equal(original['source7d'], source)
        assert np.array_equal(source[:6], refined[:6])
        path = sample['lidar_path']
        raw = np.load(path) if path.endswith('.npy') else np.fromfile(path, np.float32).reshape(-1, 4)
        idx = np.random.default_rng(20261029 + index).choice(len(raw), 16384, replace=len(raw)<16384)
        points = raw[idx,:3].copy()
        angle = 0.
        if sample['platform'] == 'drone':
            points[:,2] += 1.8
        if sample['platform'] == 'waymo':
            view = int(Path(path).parent.name.split('_')[-1])
            angle = [0.,-45.,45.,-90.,90.][view]
            points = points @ Rotation.from_euler('z', angle, degrees=True).as_matrix().T
        points = np.unique(points.astype(np.float32), axis=0)
        crop = points[(abs(points-source[:3]) <= source[3:6]/2 + 1e-7).all(1)]
        expected_yaw = 0.
        if len(crop) >= 8 and source[3]/source[4] >= 1.5:
            centered = crop[:,:2].astype(float) - crop[:,:2].astype(float).mean(0)
            _, singular, axes = np.linalg.svd(centered, full_matrices=False)
            if singular[0]**2/(len(crop)-1) > 1e-12:
                expected_yaw = np.arctan2(axes[0,1], axes[0,0])
        angle_error = abs(((float(refined[6])-expected_yaw+np.pi/2) % np.pi)-np.pi/2)
        assert angle_error < 2e-6, (sid, angle_error)
        assert len(crop) == trace[sid]['predicted_crop_points']
        # Targets are read only after verifying all prediction seals.
        meta = json.loads(Path(sample['meta_path']).read_text())
        gt = np.asarray(meta['ground_info'][sample['obj_index']]['bbox_3d'][:7], float)
        if sample['platform'] == 'drone': gt[2] += 1.8
        if sample['platform'] == 'waymo':
            gt[:3] = Rotation.from_euler('z', angle, degrees=True).as_matrix() @ gt[:3]
            gt[6] += np.deg2rad(angle)
        metrics = {}
        for arm, box in [('source', source), ('refined', refined)]:
            value = oriented_iou(gt[:3], gt[3:6], Rotation.from_euler('z',gt[6]).as_matrix(),
                                 box[:3], box[3:6], Rotation.from_euler('z',float(box[6])).as_matrix())
            errors.append(abs(value-scores[sid][arm+'_iou']))
            metrics[arm+'_iou'] = value
        rows.append({'id':sid,'platform':sample['platform'],'yaw_error_mod_pi':angle_error, **metrics})
    assert max(errors) < 1e-6
    result = {'seal_verified':True,'no_GT_used_in_independent_yaw_recovery':True,
              'center_size_bitexact':True,'SVD_matches_covariance_yaw':True,
              'independent_iou_max_error':max(errors),'n':len(rows),'rows':rows,
              'scope':'Development geometric refinement with an explicit 6D to 7D output extension; not network TTA.'}
    with args.output.open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
