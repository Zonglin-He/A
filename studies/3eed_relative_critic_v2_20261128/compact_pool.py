"""Seal compact candidate pools from frozen Qwen responses and physical evidence."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from math import pi
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
import sys

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '3eed_four_paradigms_20261128'))
from prepare_blind import ROOT, digest, read, sealed_inputs, write
from physical_readout import angle, anchored, coordinates, interval

RUN = ROOT / 'runs/relative-critic-v2-20261128'
PRIOR = ROOT / 'runs/four-paradigms-20261128'
BASELINE = '3034cb6c0bde673a6972d3d60b0a5ca0863afd50f6a669c7f977feac4fbaf57e'
FAMILIES = ('pose', 'quantile', 'regional', 'size')
CAPACITY = 24
ORACLE_GATE = 41.
TOL = 1.e-8


@dataclass(frozen=True)
class Region:
    points: np.ndarray
    ground: np.ndarray
    selection: np.ndarray | None


def ranked_regions(sample, terrain, scales, returns, in_window, selection):
    """Replay the published region vote and keep the first three valid ranks."""
    from cray_simple import MIN_RETURNS, geometry, region_catalog

    radii, assigned, births = region_catalog(sample, terrain, scales)
    if not len(births) or not in_window.any():
        return []
    votes = geometry.transfer(sample, returns, radii, assigned, births)
    scene, central = votes[:, :len(in_window)], votes[:, len(in_window):]
    window = scene[:, in_window]
    n = len(births)
    exposures = np.bincount(scene[scene >= 0], minlength=n)
    hits = np.bincount(window[window >= 0], minlength=n)
    centers = np.bincount(central[central >= 0], minlength=n)
    allowed = np.flatnonzero((centers >= MIN_RETURNS) & (hits > 0))
    if not len(allowed):
        return []
    scores = hits[allowed].astype(float)**2 / exposures[allowed]
    top = allowed[np.argsort(-scores, kind='stable')[:3]]
    held = None if selection is None else geometry.transfer(sample, selection, radii, assigned, births)
    return [Region(sample[assigned[births[i]] == i], terrain,
                   None if held is None else held[births[i]] == i) for i in top]


def pca_heading(points, ground, upright):
    from cray_simple import axial, completed_dimensions, geometry

    measured = geometry.regions.measured_box(points, ground)
    spans = completed_dimensions(measured[3:6], upright)
    return axial(measured[6] + (pi / 2 if spans[1] > spans[0] else 0.))


def projected(ref, case, stats):
    from minimal import live
    from replay_factorial import ray_changes

    if not len(case['selection']):
        return ref
    matrix = np.asarray(case['transform'])
    selection, audit = (coordinates(case[k]) for k in ('selection', 'audit'))
    old = live.first_depth(ref, selection, matrix)
    if not (old > 0).any():
        return ref
    target = ref.copy()
    target[3:6] = case['prior']
    fitted = interval(ref, case).project(target)
    for name, previous, rays in (('selection', old, selection),
                                 ('audit', live.first_depth(ref, audit, matrix), audit)):
        counts = ray_changes(previous, live.first_depth(fitted, rays, matrix))
        stats[name][0] += counts['n']
        stats[name][1] += counts['changed']
    return fitted


def box_key(box):
    """Match equivalent upright boxes, including the swapped length/width parameterization."""
    box = np.asarray(box).copy()
    if box[3] < box[4]:
        box[[3, 4]] = box[[4, 3]]
        box[6] += pi / 2
    yaw = box[6]
    if abs(box[3] - box[4]) <= TOL:
        yaw *= 2
    return tuple(np.r_[box[:6], np.cos(2*yaw), np.sin(2*yaw)].round(8))


def make(case, frame, replies):
    import minimal
    from cray_simple import anchor, horizontal_axes

    geometry = minimal.geometry
    stages = {}

    def capture(sample, terrain, scales, returns, in_window, selection=None):
        regions = ranked_regions(sample, terrain, scales, returns, in_window, selection)
        stages['H' if selection is None else 'C_ray'] = regions
        return (regions[0].points, regions[0].selection) if regions else (None, None)

    with patch.object(minimal, 'choose_region', capture):
        baseline, context = minimal.predict(frame, replies, audit=True)
    if baseline != case['base']:
        raise ValueError(f"Minimal-D replay mismatch for {case['uid']}")
    stats = {'selection': [0, 0], 'audit': [0, 0]}
    pool = [dict(box=baseline['box'], sources=['anchor'])]
    record = dict(uid=case['uid'], target=case['target'], branch=baseline['readout'], pool=pool,
                  audit=stats, actual_region_count=0, window=bool(case['first_window']))
    if baseline['box'] is None or not case['first_window']:
        return record

    seen = {box_key(baseline['box']): pool[0]}

    def add(box, family, arm):
        if not np.isfinite(box).all() or (box[3:6] <= 0).any():
            raise ValueError(f'Invalid candidate in {family}/{arm}')
        entry = seen.get(box_key(box))
        if entry is None:
            entry = dict(box=box.tolist(), sources=[])
            seen[box_key(box)] = entry
            pool.append(entry)
        entry['sources'].append(f'{family}/{arm}')

    prior = np.asarray(case['prior'])
    support = coordinates(case['measured'])
    routes = stages.get(baseline['readout'], [])
    record['actual_region_count'] = len(routes)
    frame_id = frame['target'], frame['suffix']
    spec, _, transform = geometry.camera(frame)
    viewer = np.linalg.inv(geometry.model_to_camera_matrix(spec))[:3, 3]
    raw = np.asarray(replies['3d']['value'])
    upright, _ = geometry.to_sensor(raw, transform)
    qwen = upright[6]
    if frame['target'] == 'waymo':
        qwen += np.deg2rad(geometry.WAYMO_ANGLES[frame['suffix']])

    def ref_box(region, yaw, span, quantile):
        if region is None:
            return anchored(support, np.asarray(case['eye']), np.asarray(case['initial']),
                            span, yaw, quantile)
        local = np.c_[region.points[:, :2] @ horizontal_axes(yaw).T, region.points[:, 2]]
        low, high = np.quantile(local, [quantile, 1 - quantile], axis=0)
        box = anchor(low, high, span, yaw, viewer, region.ground if quantile == 0 else None)
        return geometry.sensor_frame_box(box, *frame_id)

    primary = routes[0] if routes else None
    if primary is not None:
        pca = pca_heading(primary.points, primary.ground, upright[3:6])
        reference_span = upright[3:6]
    elif len(support) >= 3:
        centered = support[:, :2] - support[:, :2].mean(0)
        _, vectors = np.linalg.eigh(centered.T @ centered)
        axis = vectors[:, -1]
        pca, qwen = np.arctan2(axis[1], axis[0]), case['yaw']
        reference_span = prior
    else:
        return record

    if context is not None:
        evidence = context[0]
        case = dict(case, measured=evidence.support_sensor, selection=evidence.selection, audit=evidence.audit)
        if not np.allclose([pca, qwen], evidence.headings, atol=TOL, rtol=0):
            raise ValueError('Nominal PCA/Qwen headings disagree with Minimal-D')

    for name, yaw in (('pca', pca), ('pca90', pca + pi / 2), ('qwen', qwen), ('qwen90', qwen + pi / 2)):
        reference = ref_box(primary, yaw, reference_span, 0.)
        if context is not None and name in ('pca', 'qwen'):
            native = context[1][int(name == 'qwen')]
            np.testing.assert_allclose(reference, native[1], rtol=0, atol=TOL)
        fit = projected(reference, case, stats)
        if context is not None and name in ('pca', 'qwen'):
            np.testing.assert_allclose(fit, native[0], rtol=0, atol=TOL)
        add(reference, 'pose', name)
        add(fit, 'pose', f'{name}_projected')
        add(ref_box(primary, yaw, reference_span, .05), 'quantile', name)

    for rank, region in enumerate(routes[1:], 2):
        region_yaw = pca_heading(region.points, region.ground, upright[3:6])
        for name, yaw in (('pca', region_yaw), ('qwen', qwen)):
            add(ref_box(region, yaw, reference_span, 0.), 'regional', f'top{rank}_{name}')

    for name, yaw in (('pca', pca), ('qwen', qwen)):
        for scale in (.85, 1.15):
            add(ref_box(primary, yaw, prior * scale, 0.), 'size', f'{name}_{scale}')
    assert len(pool) <= CAPACITY and stats['selection'][1] == 0
    return record


def main():
    cv2.setNumThreads(1)
    RUN.mkdir(exist_ok=True, parents=True)
    started = perf_counter()
    source, _, _, pins, _ = sealed_inputs()
    assert pins['registration'] == BASELINE
    prior = PRIOR / 'inputs-private.json'
    seal = read(PRIOR / 'input-seal.json')
    assert not seal['labels_read'] and seal['source'] == pins and seal['evidence'] == digest(prior)
    cases = [c for c in read(prior) if c['target'] != 'mono']
    assert len(cases) == 702 and all(c['target'] == source[c['uid']]['target'] for c in cases)
    with ProcessPoolExecutor(8) as workers:
        jobs = ((c, source[c['uid']]['frame'], source[c['uid']]['raw']) for c in cases)
        rows = []
        for row in workers.map(generate, jobs, chunksize=4):
            rows.append(row)
            if len(rows) % 100 == 0 or len(rows) == len(cases):
                print(f'Sealed candidates prepared {len(rows)}/{len(cases)}', flush=True)
    file = RUN / 'candidates-private.json'
    write(file, rows)
    write(RUN / 'candidate-seal.json', dict(payload=digest(file), n=len(rows), labels_read=False,
          evidence=digest(prior), registration=BASELINE, generator=digest(__file__),
          protocol=digest(HERE / 'PROTOCOL_ZH.md'), seconds=perf_counter()-started))


def generate(job):
    return make(*job)


if __name__ == '__main__':
    main()
