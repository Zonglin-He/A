"""Replicate the frozen region completion and capture matched diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from cohort import OUT, PREVIOUS, PRIVATE, PROTOCOL, ROOT, checked_lock, digest, read, read_panel, write
from crossplatform.candidate_frontend import sensor_frame_box
from crossplatform.round7.camera_projection import raw_to_model_points
from followup import aligned_completion
from geometry import WAYMO_ANGLES, axial, camera, cell_points, pixel_rays, raw_returns, rigid_xyz, to_sensor, variants
from inputs import load_panel as development_panel, qwen_predictions
from region_depth import (held_cells, inside, localize, model_sample, model_to_camera_matrix, partition,
                          project_model_points, regions, sample_points, transform_to_front_view, z_buffer)


sys.path.insert(0, str(ROOT / 'crossplatform/round7'))

PRIMARY = ('A', 'H')
HEADING = ('H_yaw_only', 'H_qwen_reanchored')
SUPPORT = ('H_central', 'H_window')
FIT = ('Fit_window', 'Fit_region', 'Fit_mismatch')
ARMS = PRIMARY + HEADING + SUPPORT + FIT


def held_mask(xyz, specification, size):
    projection = project_model_points(xyz, specification, size)
    visible = projection['in_image']
    result = np.zeros(len(xyz), bool)
    result[visible] = held_cells((projection['uv'][visible] // 4).astype(np.int64))
    return result


def fit_sample(row, specification, size):
    points, indices, _ = sample_points(dict(pointpath=row['points'], platform=row['target'], seed=row['seed']))
    path = row['points']
    source = (np.load(path, allow_pickle=False) if path.endswith('.npy')
              else np.fromfile(path, '<f4').reshape(-1, 4))
    exact = raw_to_model_points(source[indices, :3], row['target'], row['suffix'])
    if row['target'] == 'waymo':
        points, _ = transform_to_front_view(points, np.empty((0, 7)), row['waymo_view'])
        points = points.astype(np.float32)
    withheld_raw = held_mask(exact, specification, size)
    withheld_model = held_mask(points, specification, size)
    keep = ~(withheld_raw | withheld_model)
    audit = dict(draws=len(points), retained=int(keep.sum()), withheld=int(withheld_raw.sum()),
                 retained_withheld_raw=int((keep & withheld_raw).sum()),
                 additionally_removed_raw_by_model=int((~withheld_raw & withheld_model).sum()),
                 additionally_removed_model_by_raw=int((withheld_raw & ~withheld_model).sum()))
    assert not audit['retained_withheld_raw']
    return np.unique(points[keep].astype(float), axis=0), audit


def to_raw_support(points, target, suffix):
    from crossplatform.round7.camera_projection import z_rotation

    raw = points.copy()
    if target == 'drone':
        raw[:, 2] -= 1.8
    elif target == 'waymo':
        raw = raw @ z_rotation(WAYMO_ANGLES[suffix])
    return raw


def heading_evidence(support, measured, uid):
    centered = support[:, :2] - support[:, :2].mean(0)
    small, large = np.linalg.eigvalsh(centered.T @ centered)
    gap = float((large - small) / (large + small)) if large + small > 0 else 0.
    offset = support[:, :2] - measured[:2]
    angle = np.sort(np.arctan2(offset[:, 1], offset[:, 0]))
    arc = 1. - float(np.diff(np.r_[angle, angle[0] + 2*np.pi]).max())/(2*np.pi)
    rng = np.random.default_rng(int(uid[:16], 16))
    shifts = []
    for _ in range(24):
        points = centered[rng.choice(len(centered), max(3, int(len(centered)*.75)), replace=False)]
        points -= points.mean(0)
        _, directions = np.linalg.eigh(points.T @ points)
        axis = directions[:, -1]
        shifts.append(abs(axial(np.arctan2(axis[1], axis[0]) - measured[6])))
    return dict(normalized_gap=gap, projected_arc=float(arc),
                jitter_p90_degrees=float(np.rad2deg(np.quantile(shifts, .9))))


def to_long_axis(box):
    result = np.array(box, float)
    if result[4] > result[3]:
        result[[3, 4]] = result[[4, 3]]
        result[6] += np.pi/2
    result[6] = axial(result[6])
    return result


def predict_frame(row, pixel, raw_3d, *, diagnostics=True):
    """Sealed Qwen decoded boxes in, sensor-coordinate upright boxes out."""
    target, view = row['target'], row['suffix']
    result = dict(boxes=dict.fromkeys(ARMS), status='invalid_qwen_3d', selected=False, fit_available=False,
                  fit_bins=0, region_bins=0, region_points=0, support_central=0, support_window=0,
                  matched_support=False, nontrivial_mismatch=False)
    if raw_3d is None:
        return result
    specification, size, transform = camera(row)
    uv, cloud = raw_returns(row, specification, size)
    raw = np.asarray(raw_3d, float)
    original, measured, _ = variants(raw, pixel, specification, size, uv, cloud)
    base = original['ray_depth']
    prior, _ = to_sensor(base, transform)
    dims, yaw = prior[3:6], prior[6]
    boxes = dict.fromkeys(ARMS, prior.tolist())
    result.update(boxes=boxes, fit_bins=measured['fit_bins'], status='missing_qwen_2d')
    if pixel is None:
        return result

    window = np.asarray(pixel, float)*np.tile(size, 2)/1000.
    guide = pixel_rays((window[:2]+window[2:])/2., specification)[0]
    center, cells = z_buffer(uv, cloud, np.flatnonzero(inside(uv, window, factor=.5)))
    central = center[~held_cells(cells)]
    fit, held = cell_points(uv, cloud, window)
    np.testing.assert_array_equal(cloud[central], fit)
    np.testing.assert_array_equal(cloud[center[held_cells(cells)]], held)
    all_ids, all_cells = z_buffer(uv, cloud, np.arange(len(uv)))
    scene_fit = all_ids[~held_cells(all_cells)]
    xy = rigid_xyz(cloud, np.linalg.inv(model_to_camera_matrix(specification)))
    sample = model_sample(row)

    def select(points):
        return localize(scene_fit, xy, points, central, uv, fit[:, 2], window, row['uid'], raw, guide, base)

    whole = select(sample)
    result['status'] = whole['status']
    if whole['status'] == 'used_region':
        node = whole['chosen']
        _, labels, births, _ = partition(sample)
        support = sample[labels[births[node]] == node]
        plane = regions.terrain(sample)
        measured_box = regions.measured_box(support, plane)
        model_yaw = yaw + np.deg2rad(WAYMO_ANGLES[view]) if target == 'waymo' else yaw
        fused = sensor_frame_box(regions.complete(measured_box, dims), target, view)
        direct = sensor_frame_box(aligned_completion(support, plane, model_yaw, dims), target, view)
        yaw_only = to_long_axis(fused)
        yaw_only[6] = yaw
        boxes.update(H=fused.tolist(), H_yaw_only=yaw_only.tolist(), H_qwen_reanchored=direct.tolist(),
                     H_central=fused.tolist(), H_window=fused.tolist())
        result.update(selected=True, node=node, region_bins=len(whole['roi_ids']), region_points=len(support))
        if diagnostics:
            result['heading'] = heading_evidence(support, measured_box, row['uid'])
            result['support_raw'] = to_raw_support(support, target, view).tolist()
            projection = project_model_points(support, specification, size)
            counts = {}
            for arm, factor in (('H_central', .5), ('H_window', 1.)):
                inside_crop = projection['in_image'] & inside(projection['uv'], window, factor)
                counts[arm] = int(inside_crop.sum())
                if counts[arm] >= 5:
                    box = regions.measured_box(support[inside_crop], plane)
                    boxes[arm] = sensor_frame_box(regions.complete(box, dims), target, view).tolist()
            result.update(support_central=counts['H_central'], support_window=counts['H_window'],
                          matched_support=counts['H_central'] >= 5)
    if diagnostics:
        sample_fit, audit = fit_sample(row, specification, size)
        result['isolation'] = audit
        fitted = select(sample_fit)
        result['fit_status'] = fitted['status']
        if fitted['status'] == 'used_region':
            plane = regions.terrain(sample_fit)
            points = xy[central]
            actual, wrong = (np.asarray(fitted[k], int) for k in ('roi_ids', 'mismatch_ids'))
            assert 5 <= len(actual) == len(wrong) <= len(points)
            for arm, subset in zip(FIT, (points, points[actual], points[wrong])):
                box = regions.complete(regions.measured_box(subset, plane), dims)
                boxes[arm] = sensor_frame_box(box, target, view).tolist()
            result.update(fit_available=True, fit_region_bins=len(actual), nontrivial_mismatch=not np.array_equal(actual,wrong))
    return result


def audit_exact_development():
    previous_path = PREVIOUS / 'private/control_boxes.json'
    assert digest(previous_path) == read(PREVIOUS / 'results/controls_prediction_seal.json')['boxes']
    previous = read(previous_path)
    qwen = qwen_predictions()
    maximum = 0.
    for row in development_panel():
        uid = row['uid']
        record = qwen[uid]
        current = predict_frame(row, record['pixel'], record['camera_boxes'].get('raw'), diagnostics=False)
        for arm, expected in (('A',record['sensor_boxes'].get('ray_depth')), ('H',previous[uid]['C_E_Qsize']),
                              ('H_qwen_reanchored',previous[uid]['C_E_Qpose'])):
            box = current['boxes'][arm]
            if expected is None:
                assert box is None
            else:
                np.testing.assert_array_equal(box, expected)
                maximum = max(maximum, float(np.max(np.abs(np.asarray(box)-expected))))
    write(OUT / 'locked_replay.json', dict(physical_frames=len(previous), coordinate_comparisons=3*len(previous),
          maximum_coordinate_difference=maximum, read_development_gt=False, method_lock=checked_lock()))
    print('Locked development outputs exactly reproduced', len(previous), flush=True)


def run():
    panel = read_panel()
    model = read(OUT / 'model_seal.json')
    hashes = read(PRIVATE / 'reply_hashes.json')
    assert model['ordered_reply_hash'] == hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    assert model['pins']['panel'] == digest(PRIVATE / 'panel.json')
    assert model['pins']['protocol'] == digest(PROTOCOL)
    start = perf_counter()
    outputs = {}
    for row in panel:
        replies = {}
        for mode in ('2d', '3d'):
            key = row['uid'] + '_' + mode
            f = PRIVATE / 'replies' / (key + '.json')
            assert digest(f) == hashes[key]
            reply = read(f)
            assert reply['pins'] == model['pins'] and reply['uid'] == row['uid'] and reply['task'] == mode
            replies[mode] = reply
        assert digest(row['points']) == row['point_sha']
        result = predict_frame(row, replies['2d']['value'], replies['3d']['value'])
        result.update(pixel=replies['2d']['value'], qwen_2d=replies['2d']['status'], qwen_3d=replies['3d']['status'])
        outputs[row['uid']] = result
    path = PRIVATE / 'predictions.json'
    write(path, outputs)
    write(OUT / 'prediction_seal.json', dict(predictions=digest(path), panel=digest(PRIVATE / 'panel.json'),
          protocol=digest(PROTOCOL), method_lock=checked_lock(), code=digest(Path(__file__)),
          qwen_seal=digest(OUT/'model_seal.json'), coordinate_replay=digest(OUT/'locked_replay.json'),
          rows=len(outputs), seconds=perf_counter()-start, task_labels_read=False, FINAL_access=False))
    print(len(outputs), 'geometry predictions sealed', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('audit', 'run'))
    {'audit':audit_exact_development, 'run':run}[parser.parse_args().action]()
