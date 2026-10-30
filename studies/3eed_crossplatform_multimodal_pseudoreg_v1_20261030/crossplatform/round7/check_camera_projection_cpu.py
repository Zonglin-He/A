"""Synthetic calibration contracts; no real media/annotations/model/GPU."""
from pathlib import Path
import hashlib, json, tempfile
import numpy as np
from camera_projection import CameraSpec, raw_to_model_points, project_model_points, read_camera_allowlist


def main():
    root = Path(__file__).resolve().parent
    k = np.array([[100., 0, 100], [0, 100., 75], [0, 0, 1.]])
    raw = np.array([[8., -.5, .2], [12., 1., -.3], [10., 0, 1.], [-2., 0, 0]])
    e = np.eye(4)
    e[:3, 3] = [.2, -.1, .05]
    errors = []
    for view in range(5):
        model = raw_to_model_points(raw, 'waymo', view)
        spec = CameraSpec('waymo', np.c_[k, np.zeros(3)], e, waymo_view_suffix=view,
                          missing_distortion_policy='assume_zero_declared')
        out = project_model_points(model, spec, (200, 150))
        # Independent analytic camera axis convention, no use of bridge matrix.
        vehicle_camera = raw - e[:3, 3]
        camera = np.stack([-vehicle_camera[:, 1], -vehicle_camera[:, 2], vehicle_camera[:, 0]], 1)
        expected = camera[:3, :2] / camera[:3, 2:] * [100., 100.] + [100., 75.]
        errors.append(float(np.max(np.abs(out['uv'][:3] - expected))))
        assert np.allclose(out['uv'][:3], expected, atol=1e-12)
        assert not out['in_image'][-1] and np.isnan(out['uv'][-1]).all()
        perm = np.array([2, 0, 3, 1])
        changed = project_model_points(model[perm], spec, (200, 150))
        assert np.allclose(changed['uv'][np.argsort(perm)], out['uv'], equal_nan=True)
    raw_other = np.array([[0., 0., 10.], [1., 1., 5.], [0., 0., -2.]])
    for platform in ('drone', 'quad'):
        spec = CameraSpec(platform, k, np.eye(4), distortion=np.zeros(5))
        out = project_model_points(raw_to_model_points(raw_other, platform), spec, (200, 150))
        expected = np.array([[100., 75.], [120., 95.]])
        assert np.allclose(out['uv'][:2], expected, atol=1e-12)
        assert not out['in_image'][-1]
    rejected = []
    for name, call in [
        ('missing_distortion_not_silently_zero', lambda: CameraSpec('quad', k, np.eye(4)).validated()),
        ('unknown_GT_fields_rejected', lambda: CameraSpec.from_camera_fields('quad', {'image_intrinsic': k, 'image_extrinsic': e, 'ground_info': 'forbidden'})),
        ('nonzero_K_fourth_column_rejected', lambda: CameraSpec('quad', np.c_[k, np.ones(3)], e, np.zeros(5)).validated()),
        ('nonrigid_extrinsic_rejected', lambda: CameraSpec('quad', k, e * 2, np.zeros(5)).validated()),
    ]:
        try:
            call()
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError(name)
    with tempfile.TemporaryDirectory() as temp:
        file = Path(temp) / 'camera.json'
        file.write_text(json.dumps({'image_intrinsic': k.tolist(), 'image_extrinsic': e.tolist(),
                                   'ground_info': {'not_consumed': [123, 456]}, 'other': 'ignored'}))
        fields = read_camera_allowlist(file)
        assert set(fields) == {'image_intrinsic', 'image_extrinsic'}
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    result = {'state': 'synthetic_camera_contracts_passed', 'program_sha256': sha(Path(__file__)),
              'bridge_sha256': sha(root/'camera_projection.py'), 'all5_Waymo_views_inverse_verified': True,
              'Drone_z_offset_and_Quad_identity_verified': True, 'point_permutation_invariant': True,
              'max_Waymo_pixel_error': max(errors), 'negative_depth_rejected': True,
              'strict_invalid_contracts_rejected': rejected, 'metadata_camera_whitelist_only': True,
              'no_real_camera_alignment_or_method_accuracy_claim': True}
    (root/'CPU_CAMERA_FIXTURE.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
