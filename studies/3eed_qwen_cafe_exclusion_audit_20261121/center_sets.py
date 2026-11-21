"""Measure admissible-center sets without selecting any new box."""
import numpy as np


TOL = 1.e-9


def axes(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array(((c, -s, 0.), (s, c, 0.), (0., 0., 1.)))


def local(points, box):
    return (points-box[:3]) @ axes(box[6])


def interior(points, box):
    return (np.abs(local(points, box)) < box[3:6]/2-TOL).all(axis=1)


def gaps(lower, upper, starts, ends):
    """Subtract open intervals from a closed interval, retaining isolated endpoints."""
    free = []
    cursor = lower
    for index in np.argsort(starts):
        start, end = starts[index], ends[index]
        if end <= lower or start >= upper:
            continue
        if start >= cursor:
            free.append((cursor, start))
        cursor = max(cursor, end)
        if cursor > upper:
            break
    if cursor <= upper:
        free.append((cursor, upper))
    return free


def free_envelope(lower, upper, starts, ends):
    """Integrate allowed y-length across x-slabs and retain feasible boundary lines."""
    overlaps = ((starts < upper) & (ends > lower)).all(axis=1)
    starts, ends = starts[overlaps], ends[overlaps]
    if not len(starts):
        return lower, upper, float(np.prod(upper-lower))
    knots = np.unique(np.r_[lower[0], upper[0], np.clip(starts[:, 0], lower[0], upper[0]),
                            np.clip(ends[:, 0], lower[0], upper[0])])
    # A slab uses its open interior for occupancy and its closure for coordinate bounds.
    slabs = list(zip(knots[:-1], knots[1:])) + [(x, x) for x in knots]
    min_xy, max_xy = np.full(2, np.inf), np.full(2, -np.inf)
    area = 0.
    for left, right in slabs:
        x = left + (right-left)/2
        live = (starts[:, 0] < x) & (x < ends[:, 0])
        free = gaps(lower[1], upper[1], starts[live, 1], ends[live, 1])
        if free:
            min_xy = np.minimum(min_xy, (left, free[0][0]))
            max_xy = np.maximum(max_xy, (right, free[-1][1]))
            area += (right-left)*sum(hi-lo for lo, hi in free)
    if not np.isfinite(min_xy).all():
        return None, None, 0.
    return min_xy, max_xy, float(area)


def lost_volume_fraction(lower, upper, dimensions):
    """Bound the fraction of the original box that any feasible translation replaces."""
    farthest = np.maximum(np.abs(lower), np.abs(upper))
    return float(1-np.clip(1-farthest/dimensions, 0., 1.).prod())


def measure(box, target, witnesses):
    support = local(target, box)
    half = box[3:6]/2
    lo = support[:, :2].max(axis=0)-half[:2]
    hi = support[:, :2].min(axis=0)+half[:2]
    compatible_xy = bool((lo <= hi+TOL).all())
    compatible_z = bool((np.abs(support[:, 2]) <= half[2]+TOL).all())
    result = dict(compatible_xy=compatible_xy, compatible_z=compatible_z, base_supports_all=False,
                  joint_possible=False, positive_area=None, joint_area=None,
                  positive_volume_change_bound=None, joint_volume_change_bound=None)
    if not (compatible_xy and compatible_z):
        return result
    middle = (lo+hi)/2
    lo, hi = np.minimum(lo, middle), np.maximum(hi, middle)
    negative = local(witnesses, box)
    negative = negative[np.abs(negative[:, 2]) < half[2]-TOL, :2]
    starts, ends = negative-half[:2]+TOL, negative+half[:2]-TOL
    valid = (starts < ends).all(axis=1)
    free_lo, free_hi, area = free_envelope(lo, hi, starts[valid], ends[valid])
    return dict(
        compatible_xy=True, compatible_z=True,
        base_supports_all=bool(((lo <= TOL) & (hi >= -TOL)).all()),
        joint_possible=free_lo is not None, positive_area=float(np.prod(hi-lo)), joint_area=area,
        positive_volume_change_bound=lost_volume_fraction(lo, hi, box[3:5]),
        joint_volume_change_bound=(None if free_lo is None else lost_volume_fraction(free_lo, free_hi, box[3:5])),
    )
