"""Exact angular co-inclusion scatter for the horizontal axis of a fixed 3D box."""
import numpy as np


ARMS = ("v1", "shared", "independent", "disk", "crossed", "recrop_once", "crop_45")
PI = np.pi


def angular_limits(radius, length, width):
    outer = np.full(len(radius), PI / 2)
    inner = np.zeros(len(radius))
    outside_width = radius > width / 2
    outside_length = radius > length / 2
    outer[outside_width] = np.arcsin((width / 2) / radius[outside_width])
    inner[outside_length] = np.arccos((length / 2) / radius[outside_length])
    return inner, outer


def moments(points):
    x, y = points.T
    return np.stack((np.ones_like(x), x, y, x*x, x*y, y*y), axis=-1)


def pair_scatter(left, right):
    """Half the sum over point pairs drawn from the two indicated support sets."""
    n, x, y, xx, xy, yy = left.T
    k, u, v, uu, uv, vv = right.T
    return .5 * np.stack((k*xx+n*uu-2*x*u, k*xy+n*uv-x*v-y*u, k*yy+n*vv-2*y*v), axis=-1)


def averaged_scatters(xy, length, width):
    """Return (xx,xy,yy) and pair mass for four predeclared distributions."""
    f = moments(xy)
    radius = np.hypot(*xy.T)
    phi = np.arctan2(xy[:, 1], xy[:, 0])
    inner, outer = angular_limits(radius, length, width)
    arc = np.maximum(0., outer-inner)
    probability = 2*arc/PI
    full = radius <= width/2
    moving = (~full) & (arc > 0)
    # Each point is visible over two arcs on the axial circle. Store both ends.
    start = np.concatenate((phi[moving]-outer[moving], phi[moving]+inner[moving])) % PI
    end = (start + np.tile(arc[moving], 2)) % PI
    values = np.tile(f[moving], (2, 1))
    initial = f[full].sum(0) + values[end < start].sum(0)
    events = np.r_[start, end]
    order = np.argsort(events, kind="stable")
    event_angles = events[order]
    deltas = np.concatenate((values, -values), axis=0)
    states = np.vstack((initial, initial + np.cumsum(deltas[order], axis=0)))
    boundaries = np.unique(np.r_[0., PI, event_angles, (event_angles-PI/2) % PI])
    weights = np.diff(boundaries)/PI
    mid = (boundaries[1:]+boundaries[:-1])/2
    a = states[np.searchsorted(event_angles, mid, side="right")]
    b = states[np.searchsorted(event_angles, (mid+PI/2) % PI, side="right")]

    common = weights @ pair_scatter(a, a)
    common_mass = .5*(weights @ (a[:, 0]*(a[:, 0]-1)))
    cross = weights @ pair_scatter(a, b)
    # A point present in both orthogonal rectangles lies in their common square.
    cin, cout = angular_limits(radius, width, width)
    self_mass = np.maximum(0., cout-cin).sum()*2/PI
    cross_mass = .5*(weights @ (a[:, 0]*b[:, 0])-self_mass)
    weighted = probability @ f
    ind = pair_scatter(weighted, weighted)
    ind_mass = .5*(weighted[0]**2 - probability @ probability)
    disk_stats = f.sum(0)
    return {"shared": (common, common_mass), "independent": (ind, ind_mass),
            "crossed": (cross, cross_mass),
            "disk": (pair_scatter(disk_stats, disk_stats), len(f)*(len(f)-1)/2)}


def principal(scatter, mass, default):
    if mass <= 1e-12:
        return default
    xx, xy, yy = scatter/mass
    if np.hypot(xx-yy, 2*xy) <= 1e-12 * max(1., xx+yy):
        return default
    return float(.5*np.arctan2(2*xy, xx-yy))


def one_crop(xy, length, width, angle, default):
    co, si = np.cos(angle), np.sin(angle)
    inside = (np.abs(xy[:, 0]*co+xy[:, 1]*si) <= length/2+1e-7) & (
        np.abs(xy[:, 1]*co-xy[:, 0]*si) <= width/2+1e-7)
    count = int(inside.sum())
    if count < 8:
        return default
    f = moments(xy[inside]).sum(0)
    return principal(pair_scatter(f, f), count*(count-1)/2, default)


def estimate(points, native, native_yaw):
    """All coordinates are in the original model frame; retain native eligibility."""
    native = np.asarray(native, dtype=np.float32)
    center, size = native[:3], native[3:6]
    result = np.full(len(ARMS), native_yaw, dtype=float)
    half = size/2 + np.float32(1e-7)
    delta = points-center
    crop = np.unique(points[np.all(np.abs(delta) <= half, axis=1)], axis=0)
    eligible = len(crop) >= 8 and size[0]/size[1] >= 1.5
    reproduced = 0.
    if eligible:
        vals, vectors = np.linalg.eigh(np.cov(crop[:, :2].astype(float), rowvar=False))
        eligible = vals[-1] > 1e-12
        if eligible:
            x, y = vectors[:, -1]
            reproduced = float(np.float32((np.arctan2(y,x) + PI/2) % PI - PI/2))
    residual = abs((reproduced-native_yaw+PI/2) % PI - PI/2)
    if residual > 1e-6:
        raise ValueError(f"Native crop failed to reproduce frozen yaw: {residual:g}")
    if not eligible:
        return result, (0, len(crop), len(crop)), residual

    length, width = map(float, size[:2])
    # Bounding rectangle prefilter, then the exact union disk and native z plane.
    r = np.hypot(length, width)/2
    slab = (np.abs(delta[:, 2]) <= half[2]) & (np.max(np.abs(delta[:, :2]), axis=1) <= r+1e-6)
    xyz = np.unique(points[slab], axis=0)
    xy = xyz[:, :2].astype(float)-center[:2].astype(float)
    xy = xy[np.hypot(*xy.T) <= r+1e-12]
    for name, (scatter, mass) in averaged_scatters(xy, length, width).items():
        result[ARMS.index(name)] = principal(scatter, mass, native_yaw)
    result[ARMS.index("recrop_once")] = one_crop(xy, length, width, native_yaw, native_yaw)
    result[ARMS.index("crop_45")] = one_crop(xy, length, width, PI/4, native_yaw)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite output heading")
    return result, (1, len(crop), len(xy)), residual
