"""Independent CPU IoU for boxes with explicit rotation matrices.

No dataset-specific Euler ordering is assumed. Rotation columns are box axes in
world coordinates; extents are full side lengths. Callers must audit conversions.
"""
import itertools
import numpy as np
from scipy.spatial import ConvexHull

SIGNS = np.array(list(itertools.product((-1., 1.), repeat=3)))
EDGES = np.array([(i, j) for i in range(8) for j in range(i + 1, 8)
                  if np.count_nonzero(SIGNS[i] != SIGNS[j]) == 1])


def _box(center, extent, rotation):
    c, e, r = (np.asarray(x, dtype=np.float64) for x in (center, extent, rotation))
    if c.shape != (3,) or e.shape != (3,) or r.shape != (3, 3):
        raise ValueError("Expected center[3], extent[3], rotation[3,3]")
    if not all(np.isfinite(x).all() for x in (c, e, r)) or (e <= 0).any():
        raise ValueError("Box must be finite and have positive side lengths")
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-7) or not np.isclose(np.linalg.det(r), 1, atol=1e-7):
        raise ValueError("Expected a proper orthonormal rotation")
    corners = (SIGNS * e / 2) @ r.T + c
    normals = np.concatenate([r.T, -r.T])
    offset = normals @ c + np.tile(e / 2, 2)
    return corners, normals, offset


def _edge_plane_vertices(corners, normals, offsets, tol):
    start, end = corners[EDGES[:, 0]], corners[EDGES[:, 1]]
    diff = end - start
    numerator = offsets[None, :] - start @ normals.T
    denominator = diff @ normals.T
    valid = abs(denominator) > tol
    t = np.zeros_like(numerator)
    np.divide(numerator, denominator, out=t, where=valid)
    valid &= (t >= -tol) & (t <= 1 + tol)
    points = start[:, None, :] + t[..., None] * diff[:, None, :]
    return points[valid]


def oriented_iou(center_a, extent_a, rotation_a, center_b, extent_b, rotation_b):
    # Shift to a local origin to avoid loss of precision in large world coords.
    ca, cb = np.asarray(center_a, float), np.asarray(center_b, float)
    origin = (ca + cb) / 2
    a, na, da = _box(ca - origin, extent_a, rotation_a)
    b, nb, db = _box(cb - origin, extent_b, rotation_b)
    tolerance = 1e-9 * max(1., float(np.max(extent_a)), float(np.max(extent_b)))
    points = np.concatenate([a, b, _edge_plane_vertices(a, nb, db, tolerance),
                             _edge_plane_vertices(b, na, da, tolerance)])
    inside = ((points @ na.T <= da + tolerance).all(1) &
              (points @ nb.T <= db + tolerance).all(1))
    points = points[inside]
    if len(points) < 4 or np.linalg.matrix_rank(points - points[0], tol=tolerance) < 3:
        return 0.
    intersection = float(ConvexHull(points).volume)
    va, vb = float(np.prod(extent_a)), float(np.prod(extent_b))
    intersection = min(max(intersection, 0.), va, vb)
    return intersection / (va + vb - intersection)


if __name__ == "__main__":
    from scipy.spatial.transform import Rotation
    r = np.eye(3)
    assert abs(oriented_iou([0,0,0], [2,2,2], r, [0,0,0], [2,2,2], r) - 1) < 1e-12
    assert oriented_iou([0,0,0], [2,2,2], r, [3,0,0], [2,2,2], r) == 0
    assert oriented_iou([0,0,0], [2,2,2], r, [2,0,0], [2,2,2], r) == 0
    assert abs(oriented_iou([0,0,0], [2,2,2], r, [0,0,0], [1,1,1], r) - .125) < 1e-12
    # Known axis-aligned overlap: intersection=4, union=12.
    assert abs(oriented_iou([0,0,0], [2,2,2], r, [1,0,0], [2,2,2], r) - 1/3) < 1e-12
    ra = Rotation.from_euler('xyz', [19,32,14], degrees=True).as_matrix()
    rb = Rotation.from_euler('xyz', [43,16,-27], degrees=True).as_matrix()
    q = Rotation.from_euler('xyz', [-15,42,58], degrees=True).as_matrix()
    a, b, ea, eb = np.array([.2,.8,.4]), np.array([.5,.6,.9]), [2,3,4], [1,3,2]
    x = oriented_iou(a,ea,ra,b,eb,rb)
    assert abs(x - oriented_iou(b,eb,rb,a,ea,ra)) < 1e-12
    assert abs(x - oriented_iou(q@a,ea,q@ra,q@b,eb,q@rb)) < 1e-12
    print("Independent OBB IoU checks passed: identity, disjoint, contact, containment, analytic overlap, symmetry, rigid-transform invariance.")
