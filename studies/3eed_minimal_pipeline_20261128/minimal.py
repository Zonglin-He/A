"""Bind physical evidence and read out visibility-equivalent pose candidates."""
from dataclasses import dataclass

import numpy as np

from cray_simple import (FOUR_SCALES, MIN_RETURNS, a_box, axial, choose_region,
                         complete_h, completed_dimensions, horizontal_axes)
from live_2d import geometry, live, projection_overlap, visible_rectangle
from center_sets import axes
from extent_clip import TOL


SINGLE = (7,)
DUAL = (FOUR_SCALES[0], FOUR_SCALES[-1])
HEADINGS = ('pca', 'qwen')


@dataclass(frozen=True)
class Evidence:
    spec: object
    size: tuple
    transform: np.ndarray
    rectangle: np.ndarray | None
    eye_model: np.ndarray
    support_model: np.ndarray
    support_sensor: np.ndarray
    ground: np.ndarray
    selection: np.ndarray
    audit: np.ndarray
    headings: tuple
    reference_span: np.ndarray
    prior: np.ndarray


def bind(frame, replies, scales, *, audit=False):
    """Use a shared view for dual-pose binding and historical low-evidence readouts."""
    raw = replies['3d']['value']
    pixels = replies['2d']['value']
    result = dict(box=None, route='H', applicable=False, readout='missing_3d',
                  selection_rays=0, heading=None, extent_corrected=False)
    if raw is None:
        return result, None
    raw = np.asarray(raw, float)
    spec, size, transform = geometry.camera(frame)
    upright, rotation = geometry.to_sensor(raw, transform)
    if pixels is None:
        return dict(result, box=upright.tolist(), readout='A'), None

    uv, returns = geometry.raw_returns(frame, spec, size)
    window = np.asarray(pixels, float) * np.tile(size, 2) / 1000.
    central, cells = geometry.z_buffer(uv, returns, np.flatnonzero(geometry.inside(uv, window, .5)))
    central = central[geometry.reserve(cells) == 0]
    if len(central) < MIN_RETURNS:
        box = a_box(raw, window, central, returns, spec, transform)
        return dict(result, box=box.tolist(), readout='A'), None

    ids, cells = geometry.z_buffer(uv, returns, np.arange(len(uv)))
    partition = geometry.reserve(cells)
    in_window = geometry.inside(uv[ids], window)
    fit_ids = ids[partition == 0]
    model_camera = geometry.model_to_camera_matrix(spec)
    inverse_model = np.linalg.inv(model_camera)
    xyz = geometry.rigid_xyz(returns, inverse_model)
    fit_returns = xyz[np.r_[fit_ids, central]]
    reference_span = upright[3:6]
    target, suffix = frame['target'], frame['suffix']

    if live.vehicle_reply(replies['2d']):
        sample, _ = geometry.fit_sample(frame, spec, size)
        if len(sample) >= 9:
            terrain = geometry.regions.terrain(sample)
            pick = ids[(partition == 1) & in_window]
            held = ids[(partition == 2) & in_window] if audit else ids[:0]
            requested = np.r_[pick, held]
            support, associated = choose_region(sample, terrain, scales, fit_returns,
                                               in_window[partition == 0], xyz[requested])
            if support is not None:
                selected = returns[pick[associated[:len(pick)]]]
                result['selection_rays'] = len(selected)
                if len(selected) >= MIN_RETURNS:
                    measured = geometry.regions.measured_box(support, terrain)
                    spans = completed_dimensions(measured[3:6], reference_span)
                    pca = axial(measured[6] + (np.pi / 2 if spans[1] > spans[0] else 0.))
                    qwen = upright[6]
                    if target == 'waymo':
                        qwen += np.deg2rad(geometry.WAYMO_ANGLES[suffix])
                    converted = geometry.rigid_xyz(support, np.linalg.inv(transform) @ model_camera)
                    vertical = int(np.argmax(np.abs(rotation[2])))
                    prior = np.r_[np.sort(np.delete(raw[3:6], vertical))[::-1], raw[3+vertical]]
                    evidence = Evidence(
                        spec, size, transform, visible_rectangle(window, size), inverse_model[:3, 3],
                        support, converted, terrain, selected, returns[held[associated[len(pick):]]],
                        (pca, qwen), reference_span, prior,
                    )
                    return None, evidence

    sample = geometry.model_sample(frame)
    terrain = geometry.regions.terrain(sample)
    support, _ = choose_region(sample, terrain, scales, fit_returns, in_window[partition == 0])
    if support is None:
        box = a_box(raw, window, central, returns, spec, transform)
        return dict(result, box=box.tolist(), readout='A'), None
    box = geometry.sensor_frame_box(complete_h(support, terrain, reference_span), target, suffix)
    return dict(result, box=box.tolist(), readout='H'), None


def axis_shift(mode, current_span, new_span):
    """Center displacement in an anchored interval when its span changes."""
    return mode * (new_span - current_span) / 2


def recover_pose(evidence, yaw, target, suffix):
    """Build a reference interval, then project its span with both witness sets."""
    axes_xy = horizontal_axes(yaw)
    xy = evidence.support_model[:, :2] @ axes_xy.T
    lower, upper = xy.min(0), xy.max(0)
    viewer = evidence.eye_model[:2] @ axes_xy.T
    ref_mode = (viewer < lower).astype(float) - (viewer > upper)
    size = evidence.reference_span
    center = ((lower + upper) / 2 + axis_shift(ref_mode, upper - lower, size[:2])) @ axes_xy
    bottom = evidence.support_model[:, 2].min()
    terrain = evidence.ground
    if np.isfinite(terrain).all():
        bottom = min(bottom, float(center @ terrain[:2] + terrain[2]))
    model_box = np.r_[center, bottom + axis_shift(1., 0., size[2]), size, yaw]
    reference = geometry.sensor_frame_box(model_box, target, suffix)

    depth = live.first_depth(reference, evidence.selection, evidence.transform)
    protected = live.protected_points(reference, evidence.support_sensor, evidence.selection,
                                      depth, evidence.transform)
    eye, measured, entries = protected
    size = reference[3:6]
    half = size / 2
    if not len(entries) or (np.abs(eye) <= half + TOL).all():
        return reference, reference, depth, protected

    points = np.concatenate((measured, entries))
    if not (np.abs(points) <= half + TOL).all():
        raise ValueError('An extent witness lies outside its incumbent')
    low, high = points.min(0), points.max(0)
    mode = np.r_[(eye[:2] < -half[:2]).astype(float) - (eye[:2] > half[:2]), 1.]
    required = np.where(mode > 0, half + high + TOL, np.where(
        mode < 0, half - low + TOL, 2 * (np.maximum(-low, high) + TOL)))
    span = np.minimum(size, np.maximum(evidence.prior, required))
    move = axis_shift(mode, size, span)
    box = reference.copy()
    box[:2] += axes(reference[6])[:2, :2] @ move[:2]
    box[2] += move[2]
    box[3:6] = span
    return (box if np.max(np.abs(box - reference)) > TOL else reference), reference, depth, protected


def select(evidence, candidates):
    scores = (None, None) if evidence.rectangle is None else tuple(
        projection_overlap(candidate[0], evidence.rectangle, evidence.spec, evidence.transform, evidence.size)
        for candidate in candidates)
    pca, qwen = (0. if score is None else score for score in scores)
    if pca != qwen:
        return int(qwen > pca), False

    cap = max(.1, .5 * min(evidence.reference_span[:2]))
    observed = evidence.selection[:, 2]
    costs = [float(np.where(depth > 0, np.minimum(np.abs(depth - observed), cap), cap).sum())
             for _, _, depth, _ in candidates]
    return int(costs[1] < costs[0] - 1.e-12 * len(observed) * cap), True


def predict(frame, replies, *, scales=FOUR_SCALES, audit=False):
    fallback, evidence = bind(frame, replies, scales, audit=audit)
    if evidence is None:
        return (fallback, None) if audit else fallback
    candidates = tuple(recover_pose(evidence, yaw, frame['target'], frame['suffix'])
                       for yaw in evidence.headings)
    winner, ray_default = select(evidence, candidates)
    box, reference, _, _ = candidates[winner]
    result = dict(box=box.tolist(), route='C_ray', applicable=True, readout='C_ray',
                  selection_rays=len(evidence.selection), heading=HEADINGS[winner],
                  extent_corrected=box is not reference)
    return (result, (evidence, candidates, ray_default)) if audit else result
