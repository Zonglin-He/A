"""Camera-only projection bridge. No task annotations, model or GT consumer.

The missing-distortion policy must be explicit: zero is an assumption, not a
statement that the distributed JPEGs are physically undistorted.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

CAMERA_KEYS = frozenset({'image_intrinsic', 'image_extrinsic', 'camera_distortion'})
WAYMO_ANGLES = (0., -45., 45., -90., 90.)


def read_camera_allowlist(path):
    import ijson
    from ijson.common import ObjectBuilder
    builders = {}
    with Path(path).open('rb') as stream:
        for prefix, event, value in ijson.parse(stream):
            first = prefix.split('.', 1)[0]
            if first in CAMERA_KEYS:
                builders.setdefault(first, ObjectBuilder()).event(event, value)
    result = {key: value.value for key, value in builders.items()}
    if not {'image_intrinsic', 'image_extrinsic'} <= result.keys():
        raise ValueError('Camera intrinsics/extrinsics are missing')
    return result


def z_rotation(degrees):
    angle = np.deg2rad(degrees)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


@dataclass(frozen=True)
class CameraSpec:
    platform: str
    intrinsic: np.ndarray
    extrinsic: np.ndarray
    distortion: object = None
    waymo_view_suffix: int = 0
    missing_distortion_policy: str = 'reject'

    @classmethod
    def from_camera_fields(cls, platform, fields, **options):
        if set(fields) - CAMERA_KEYS:
            raise ValueError('Only camera whitelist fields are accepted')
        return cls(platform, np.asarray(fields['image_intrinsic'], np.float64),
                   np.asarray(fields['image_extrinsic'], np.float64),
                   fields.get('camera_distortion'), **options)

    def validated(self):
        k = np.asarray(self.intrinsic, np.float64)
        e = np.asarray(self.extrinsic, np.float64)
        if k.shape == (3, 4):
            if not np.array_equal(k[:, 3], np.zeros(3)):
                raise ValueError('Nonzero fourth intrinsic column needs another camera model')
            k = k[:, :3]
        if k.shape != (3, 3) or e.shape != (4, 4):
            raise ValueError('Invalid camera matrix shapes')
        if not (np.isfinite(k).all() and np.isfinite(e).all()):
            raise ValueError('Nonfinite camera matrix')
        if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1], atol=1e-10):
            raise ValueError('Unsupported pinhole intrinsic')
        if not np.allclose(e[3], [0, 0, 0, 1], atol=1e-8):
            raise ValueError('Invalid homogeneous extrinsic')
        if not np.allclose(e[:3, :3].T @ e[:3, :3], np.eye(3), atol=1e-6) or abs(np.linalg.det(e[:3, :3])-1) > 1e-6:
            raise ValueError('Extrinsic rotation is not rigid')
        if self.distortion is None:
            if self.missing_distortion_policy != 'assume_zero_declared':
                raise ValueError('Missing distortion; an explicit zero-assumption is required')
            d, assumed = np.zeros(5), True
        else:
            d = np.asarray(self.distortion, np.float64).reshape(-1)
            if len(d) not in (4, 5, 8) or not np.isfinite(d).all():
                raise ValueError('Unsupported distortion coefficients')
            assumed = False
        return k.copy(), e.copy(), d.copy(), assumed


def raw_to_model_points(points, platform, waymo_view_suffix=0):
    p = np.array(points, dtype=np.float64, copy=True)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError('Finite Nx3 points required')
    if platform == 'waymo':
        if waymo_view_suffix not in range(5):
            raise ValueError('Unknown Waymo camera suffix')
        p = p @ z_rotation(WAYMO_ANGLES[waymo_view_suffix]).T
    elif platform == 'drone':
        p[:, 2] += 1.8
    elif platform != 'quad':
        raise ValueError('Unknown platform')
    return p


def model_to_camera_matrix(spec):
    _, e, _, _ = spec.validated()
    undo = np.eye(4)
    if spec.platform == 'waymo':
        if spec.waymo_view_suffix not in range(5):
            raise ValueError('Unknown Waymo camera suffix')
        undo[:3, :3] = z_rotation(-WAYMO_ANGLES[spec.waymo_view_suffix])
        axis = np.eye(4)
        axis[:3, :3] = [[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]]
        return axis @ np.linalg.inv(e) @ undo
    if spec.platform == 'drone':
        undo[2, 3] = -1.8
    elif spec.platform != 'quad':
        raise ValueError('Unknown platform')
    return e @ undo


def project_model_points(points, spec, image_size_wh, min_depth=.1):
    p = np.asarray(points, np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError('Finite model-coordinate Nx3 points required')
    w, h = map(int, image_size_wh)
    if w <= 0 or h <= 0 or min_depth <= 0:
        raise ValueError('Invalid image/depth configuration')
    k, _, d, assumed = spec.validated()
    camera = (np.c_[p, np.ones(len(p))] @ model_to_camera_matrix(spec).T)[:, :3]
    depth = camera[:, 2].copy()
    uv = np.full((len(p), 2), np.nan)
    front = depth > min_depth
    if front.any():
        if np.any(d):
            import cv2
            uv[front] = cv2.projectPoints(camera[front], np.zeros(3), np.zeros(3), k, d)[0][:, 0]
        else:
            homogeneous = camera[front] @ k.T
            uv[front] = homogeneous[:, :2] / homogeneous[:, 2:3]
    valid = front & np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return {'uv': uv, 'depth': depth, 'in_image': valid,
            'diagnostics': {'platform': spec.platform,
                            'missing_distortion_assumed_zero': assumed,
                            'image_size_wh': [w, h],
                            'positive_depth_count': int(front.sum()),
                            'in_image_count': int(valid.sum()),
                            'no_pose_or_unused_translation_vector_added': True,
                            'camera_convention': 'Waymo A@inv(E)@undo_front; Drone E@undo_z; Quad E'}}
