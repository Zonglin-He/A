"""Retained counterexample and boundary regression; synthetic pixels only."""
from pathlib import Path
import hashlib, importlib.util, json, sys
import numpy as np
from multimodal_pseudo import visible_points


def main():
    root = Path(__file__).resolve().parent
    old = root/'history/before_FOV_fix/multimodal_pseudo.py'
    spec = importlib.util.spec_from_file_location('r7_pseudo_before_FOV_fix', old)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    p = np.array([[0., 0., 1.]])
    image = np.full((2, 2, 3), 255, np.uint8)
    depth = np.ones(1)
    left = np.array([[-.4, 0.]])
    old_count = len(module.visible_points(p, image, left, depth)[0])
    new_count = len(visible_points(p, image, left, depth)[0])
    assert old_count == 1 and new_count == 0
    cases = [([-1e-8, 0.], 0), ([0., -1e-8], 0), ([0., 0.], 1),
             ([1.9, 1.9], 1), ([2., 1.], 0), ([1., 2.], 0), ([float('nan'), 0], 0)]
    for uv, expected in cases:
        assert len(visible_points(p, image, np.array([uv]), depth)[0]) == expected
    sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
    receipt = {'state': 'FOV_counterexample_retained_and_repair_passed',
               'original_source_sha256': sha(old), 'fixed_source_sha256': sha(root/'multimodal_pseudo.py'),
               'negative_float_ray_old_accepted_count': old_count,
               'negative_float_ray_fixed_accepted_count': new_count,
               'all7_continuous_edge_nan_cases_passed': True,
               'sampling_convention': 'continuous_half_open_FOV_then_floor_pixel_cell',
               'no_real_data_model_GT_GPU_access': True}
    (root/'CPU_FOV_REPAIR_FIXTURE.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
