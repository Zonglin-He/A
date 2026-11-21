"""Check quartic intersections with independent polynomial roots."""

from itertools import product
import json
from math import comb
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / '3eed_oabc_pose_center_feasibility_20261121'))
from feasible_geometry import CameraRays, continuous_fit, rotation
from shape_proxy import QuarticRays


def scalar_entry(ray, center, origin, half):
    local = (origin - center) / half
    slope = ray / half
    axes = np.abs(ray) > 1.e-12
    if np.any(~axes & (np.abs(local) > 1.)):
        return np.inf
    left, right = (-1. - local[axes]) / slope[axes], (1. - local[axes]) / slope[axes]
    start, end = np.minimum(left, right).max(), np.maximum(left, right).min()
    if end < start or start <= 0.:
        return np.inf
    base = local + start * slope
    if np.sum(base**4) <= 1.:
        return start
    step = (end - start) * slope
    coefficients = [sum(comb(4, k) * a**(4-k) * b**k for a, b in zip(base, step)) for k in range(5)]
    coefficients[0] -= 1.
    roots = np.polynomial.polynomial.polyroots(coefficients)
    valid = [float(r.real) for r in roots if abs(r.imag) < 1.e-10 and 0. <= r.real <= 1.]
    return start + min(valid) * (end - start) if valid else np.inf


def synthetic_fit(camera):
    dims, yaw = np.array([4., 1.9, 1.6]), .41
    center = np.array([1.35, -.30, 9.])
    inverse, basis = np.linalg.inv(camera), rotation(yaw)
    origin = inverse[:3, 3] @ basis
    camera_center = camera[:3, :3] @ (center @ basis.T) + camera[:3, 3]
    slopes = np.array(list(product(np.linspace(-.23, .23, 20), np.linspace(-.17, .17, 15))))
    rays = np.column_stack((slopes + camera_center[:2] / camera_center[2], np.ones(len(slopes))))
    directions = rays @ inverse[:3, :3].T @ basis
    depths = np.asarray([scalar_entry(v, center, origin, dims / 2.) for v in directions])
    hits = np.isfinite(depths)
    returns = rays[hits] * depths[hits, None]
    initial = center + (.4, -.39, .37)
    result = {}
    for surface in (CameraRays, QuarticRays):
        view = surface(returns, camera, yaw, dims, dims[1] / 2.)
        readout, loss, _, _ = continuous_fit(view, initial - dims, initial + dims, initial, 5, 120)
        result[surface.__name__] = dict(center_m=float(np.linalg.norm(readout - center)), mean_residual_m=loss)
    assert len(returns) >= 40
    assert result['QuarticRays']['center_m'] < min(.03, result['CameraRays']['center_m'])
    return len(returns), result


def verify():
    rng = np.random.default_rng(819)
    camera = np.eye(4)
    camera[:3, :3], camera[:3, 3] = rotation(.31), (.25, -.37, .43)
    directions = np.column_stack((rng.uniform(-.4, .4, (247, 2)), np.ones(247)))
    centers = rng.uniform((-.9, -.6, 4.5), (.9, .6, 10.), (20, 3))
    half = np.array([1.45, .87, .75])
    cases, hits, error = 0, 0, 0.
    for yaw in (.0, .6, 1.2):
        view = QuarticRays(directions, camera, yaw, 2. * half, .87)
        depth, hit, _ = view.entries(centers)
        expected = np.array([[scalar_entry(ray, center, view.origin, half) for ray in view.directions]
                             for center in centers])
        np.testing.assert_array_equal(hit, np.isfinite(expected))
        cases += expected.size
        hits += int(hit.sum())
        error = max(error, float(np.max(np.abs(expected[hit] - depth[hit]))))
    assert error < 2.e-7
    example = np.array([[0., 0., 1.], [.095, .095, 1.]])
    target, extent = [0., 0., 10.], np.full(3, 2.)
    view = QuarticRays(example, np.eye(4), 0., extent, 1.)
    depth, valid, _ = view.entries(target)
    assert valid.tolist() == [[True, False]] and abs(depth[0, 0] - 9.) < 1.e-7
    assert CameraRays(example, np.eye(4), 0., extent, 1.).entries(target)[1][0, 1]
    tangent, valid, _ = view.entries([1., 0., 10.])
    assert valid[0, 0] and abs(tangent[0, 0] - 10.) < 2.e-4
    n, synthetic = synthetic_fit(camera)
    return dict(independent_root_cases=cases, matching_hits=hits, max_entry_error_m=error,
                central_corner_tangent=True, synthetic_rays=n, synthetic_shape_control=synthetic)


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
