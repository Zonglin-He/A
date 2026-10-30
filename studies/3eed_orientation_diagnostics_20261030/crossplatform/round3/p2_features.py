"""GT-free geometry descriptors and controlled observation perturbations.

Historical round2 cached parents only. No metadata/annotation or score input.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, cKDTree, QhullError
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

R = Path(__file__).resolve().parent
OLD = R.parent / 'round2'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def wrap(a):
    return (a + np.pi / 2) % np.pi - np.pi / 2


def yaw(points, box):
    if len(points) < 3:
        return None
    cov = np.cov(points[:, :2], rowvar=False)
    eig, vec = np.linalg.eigh(cov)
    if eig[-1] <= 1e-12:
        return None
    v = vec[:, -1]
    return float(wrap(np.arctan2(v[1], v[0]) - (np.pi / 2 if box[3] < box[4] else 0)))


def deviation(a, reference):
    return None if a is None or reference is None else float(abs(wrap(a - reference)))


def summarize_angles(values, reference):
    valid = [v for v in values if v is not None]
    delta = [deviation(v, reference) for v in valid] if reference is not None else []
    return {'valid': len(valid), 'total': len(values),
            'mean_abs_radians': float(np.mean(delta)) if delta else None,
            'max_abs_radians': float(max(delta)) if delta else None,
            'cos2_agreement': float(np.mean(np.cos(2 * np.asarray(delta)))) if delta else None}


def features(points, box, seed):
    pc = np.unique(np.asarray(points, dtype=np.float64)[:, :3], axis=0)
    box = np.asarray(box, dtype=np.float64)
    valid_box = bool(np.isfinite(box).all() and np.all(box[3:6] > 0))
    if not valid_box:
        return {'valid_source_box': False}, {}
    relative = pc - box[:3]
    selected = np.all(abs(relative) <= box[3:6] / 2 + 1e-7, axis=1)
    crop = pc[selected]
    n = len(crop)
    f = {'valid_source_box': True, 'unique_input_points': len(pc), 'unique_crop_count': n,
         'log_point_density': float(np.log1p(n / np.prod(box[3:6])))}
    if n < 3:
        f['reason'] = 'fewer_than_three_unique_points_no_yaw_descriptors'
        return f, {'crop_count': n}
    base_yaw = yaw(crop, box)
    ev = np.linalg.eigvalsh(np.cov(crop[:, :2], rowvar=False))
    f['PCA_eigengap'] = float((ev[-1] - ev[0]) / max(float(ev.sum()), 1e-12))
    f['z_span_fraction'] = float(np.ptp(crop[:, 2]) / box[5])
    f['source_crop_boundary_fraction'] = float(np.mean(np.any(abs(crop - box[:3]) >= .45 * box[3:6], axis=1)))
    try:
        f['XY_hull_area_fraction'] = float(ConvexHull(crop[:, :2]).volume / np.prod(box[3:5]))
    except QhullError:
        f['XY_hull_area_fraction'] = 0.
    k = min(8, n)
    distances, index = cKDTree(crop).query(crop, k=k)
    local = crop[index] - crop[index].mean(1, keepdims=True)
    values, vectors = np.linalg.eigh(np.einsum('nki,nkj->nij', local, local) / max(k - 1, 1))
    planarity = (values[:, 1] - values[:, 0]) / np.maximum(values[:, 2], 1e-12)
    normals = vectors[:, :, 0]
    horizontal = np.linalg.norm(normals[:, :2], axis=1) >= .8
    f['horizontal_normal_support'] = float(np.mean(horizontal & (planarity >= .1)))
    f['vertical_normal_planarity'] = float(np.mean(planarity[horizontal])) if horizontal.any() else None
    radius = 3 * max(float(np.median(distances[:, 1])), 1e-8)
    src = np.repeat(np.arange(n), k - 1)
    dst = index[:, 1:].reshape(-1)
    include = distances[:, 1:].reshape(-1) <= radius
    graph = coo_matrix((np.ones(int(include.sum())), (src[include], dst[include])), shape=(n, n)).tocsr()
    components, labels = connected_components(graph, directed=False)
    f['nearest_neighbor_components_fraction'] = float(np.bincount(labels).max() / n)
    f['components_count'] = int(components)
    rng = np.random.default_rng(seed)
    boot = [yaw(crop[rng.choice(n, max(3, int(.75 * n)), replace=False)], box) for _ in range(16)]
    partial = []
    retained = []
    centered = crop[:, :2] - np.median(crop[:, :2], axis=0)
    for angle in np.arange(8) * np.pi / 4:
        projection = centered @ np.array([np.cos(angle), np.sin(angle)])
        subset = crop[projection >= np.median(projection)]
        partial.append(yaw(subset, box)); retained.append(len(subset))
    roof = crop[crop[:, 2] >= np.median(crop[:, 2])]
    partial.append(yaw(roof, box)); retained.append(len(roof))
    expanded = pc[np.all(abs(relative) <= 1.25 * box[3:6] / 2 + 1e-7, axis=1)]
    expansion = [yaw(expanded, box)]
    shell = pc[np.all(abs(relative) <= 1.5 * box[3:6] / 2 + 1e-7, axis=1) & ~selected]
    shell = shell[np.argsort(np.sum(((shell - box[:3]) / box[3:6]) ** 2, axis=1), kind='stable')]
    background = []
    injected = []
    for fraction in [.25, .5]:
        count = min(len(shell), int(np.ceil(fraction * n)))
        injected.append(count)
        background.append(yaw(np.concatenate([crop, shell[:count]]), box) if count else None)
    shifted = []
    shift_counts = []
    for axis in range(3):
        for sign in [-1, 1]:
            center = box[:3].copy(); center[axis] += sign * .1 * box[3 + axis]
            subset = pc[np.all(abs(pc - center) <= box[3:6] / 2 + 1e-7, axis=1)]
            shifted.append(yaw(subset, box)); shift_counts.append(len(subset))
    perturb = {'bootstrap': summarize_angles(boot, base_yaw), 'partial_visibility': summarize_angles(partial, base_yaw),
               'crop_expansion': summarize_angles(expansion, base_yaw), 'background_injection': summarize_angles(background, base_yaw),
               'crop_shift': summarize_angles(shifted, base_yaw)}
    for name, value in perturb.items():
        f[name + '_stability'] = value['cos2_agreement']
    raw = {'PCA_yaw': base_yaw, 'bootstrap_yaws': boot, 'partial_visibility_yaws': partial,
           'partial_retained_counts': retained, 'expanded_yaws': expansion, 'expanded_count': len(expanded),
           'background_yaws': background, 'background_injected_counts': injected, 'annular_available_count': len(shell),
           'shifted_yaws': shifted, 'shifted_crop_counts': shift_counts, 'angle_statistics': perturb}
    return f, raw


def main():
    lock = R / 'PLAN_DIAGNOSTIC_LOCK.json'
    assert sha(lock) == (R / 'PLAN_DIAGNOSTIC_LOCK.sha256').read_text().strip()
    out = R / 'p2'; out.mkdir(exist_ok=False)
    engineering = {'plan_sha256': sha(lock), 'source_sha256': sha(Path(__file__)),
                   'input_manifests': {c: sha(OLD / 'inputs' / (c + '_manifest.json')) for c in ['dev', 'confirm']},
                   'numeric_GT_access': False}
    (out / 'CODE_INPUT_LOCK.json').write_text(json.dumps(engineering, indent=2) + '\n')
    rows = []
    for cohort in ['dev', 'confirm']:
        samples = json.loads((OLD / 'inputs' / (cohort + '_manifest.json')).read_text())
        seal = json.loads((OLD / (cohort + '_source') / 'sealed.json').read_text())
        assert seal['input_manifest_sha256'] == sha(OLD / 'inputs' / (cohort + '_manifest.json'))
        for sample in samples:
            p = OLD / (cohort + '_source') / (sample['id'] + '.npz')
            assert sha(p) == seal['files'][p.name]
            pp = Path(sample['point_path']); assert sha(pp) == sample['point_sha256']
            with np.load(p, allow_pickle=False) as source:
                selected = int(source['semantic_scores'].argmax())
                box = np.r_[source['boxes'][selected].astype(float), 0.]
            f, trace = features(np.load(pp, allow_pickle=False), box, int(sample['id'][:8], 16))
            rows.append({'id': sample['id'], 'cohort': cohort, 'platform': sample['platform'],
                         'scene': sample['scene'], 'selected_query': selected, 'features': f, 'trace': trace})
    result = out / 'FEATURES_PRIVATE.json'
    result.write_text(json.dumps({'scope': 'Historical54 unioncheckpoint descriptive descriptors; no labels or scores accessed',
                                 'rows': rows}, indent=2, allow_nan=False) + '\n')
    sealed = {'n': len(rows), 'numeric_GT_access': False, 'plan_sha256': sha(lock),
              'code_lock_sha256': sha(out / 'CODE_INPUT_LOCK.json'), 'files': {result.name: sha(result)}}
    (out / 'SEALED.json').write_text(json.dumps(sealed, indent=2) + '\n')
    print(json.dumps({'n': len(rows), 'GT_free_features_sealed': True, 'feature_sha256': sha(result)}))


if __name__ == '__main__':
    main()
