"""Clip a box to its intrinsic extent while preserving its geometric evidence."""

import numpy as np

from center_sets import axes, local


TOL = 1e-8


def protected_points(box, fit, selection, depth, transform):
    """Return the camera and protected points in incumbent box coordinates."""
    inverse = np.linalg.inv(transform)
    rotation, translation = inverse[:3, :3], inverse[:3, 3]
    eye = local(translation[None], box)[0]
    measured = local(np.concatenate((fit, selection @ rotation.T + translation)), box)
    measured = measured[(np.abs(measured) <= box[3:6] / 2 + TOL).all(axis=1)]
    hit = depth > 0
    lifted = selection[hit] * (depth[hit] / selection[hit, 2])[:, None]
    entries = local(lifted @ rotation.T + translation, box)
    return eye, measured, entries


def clip_extent(box, raw_dimensions, rotation, eye, measured, entries, *, height_only):
    """Return the smallest anchored box allowed by the prior and the witnesses."""
    size = box[3:6]
    half = size / 2
    if not len(entries) or (np.abs(eye) <= half + TOL).all():
        return box
    witnesses = np.concatenate((measured, entries))
    if not (np.abs(witnesses) <= half + TOL).all():
        raise ValueError("An extent witness lies outside its incumbent")

    vertical = int(np.argmax(np.abs(rotation[2])))
    required_height = half[2] + witnesses[:, 2].max() + TOL
    height = min(size[2], max(raw_dimensions[vertical], required_height))
    new = box.copy()
    new[2] += (height - size[2]) / 2
    new[5] = height
    if not height_only:
        prior = np.sort(np.delete(raw_dimensions, vertical))[::-1]
        low = witnesses[:, :2].min(axis=0)
        high = witnesses[:, :2].max(axis=0)
        lower_fixed = eye[:2] < -half[:2]
        upper_fixed = eye[:2] > half[:2]
        required = np.where(
            lower_fixed,
            half[:2] + high + TOL,
            np.where(upper_fixed, half[:2] - low + TOL, 2 * (np.maximum(-low, high) + TOL)),
        )
        horizontal = np.minimum(size[:2], np.maximum(prior, required))
        offset = (size[:2] - horizontal) * (upper_fixed.astype(float) - lower_fixed) / 2
        new[:2] += axes(box[6])[:2, :2] @ offset
        new[3:5] = horizontal
    return new if np.max(np.abs(new - box)) > TOL else box
