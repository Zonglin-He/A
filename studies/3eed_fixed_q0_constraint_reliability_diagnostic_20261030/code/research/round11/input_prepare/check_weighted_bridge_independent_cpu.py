"""Independent synthetic allone/allzero/RGBzero bridge against frozen R10 G."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
B=Path(__file__).resolve().parents[3];D=Path(__file__).resolve().parent
sys.path[:0]=[str(B/'crossplatform/round11'),str(B/'crossplatform/round10'),str(B/'crossplatform/round7')]
from visibility_geometry import generate_geometry_targets
from weighted_geometry import generate_weighted_geometry_targets
from camera_projection import CameraSpec
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=np.linspace(-.8,.8,17);x,y=np.meshgrid(a,a);p=np.c_[x.ravel(),y.ravel(),np.full(x.size,4.08)].astype(np.float32)
camera=CameraSpec('quad',np.array([[100,0,60],[0,100,50],[0,0,1]],float),np.eye(4),np.zeros(5));source=np.array([0,0,5,2,2,2],np.float32);kw=dict(points_fit=p,raw_indices=np.arange(len(p),dtype=np.int64),source_box6=source,q0=0,camera_spec=camera,rgb_mask2d=np.ones((100,120),bool),segmentation_valid=True)
base=generate_geometry_targets(**kw);faces=[f'{j}:{s}' for j in range(3) for s in (-1,1)]
one=generate_weighted_geometry_targets(**kw,face_weights=dict.fromkeys(faces,1.),RGB_weight=1.)
zero=generate_weighted_geometry_targets(**kw,face_weights=dict.fromkeys(faces,0.),RGB_weight=0.)
rgbzero=generate_weighted_geometry_targets(**kw,face_weights=dict.fromkeys(faces,1.),RGB_weight=0.)
assert base['acceptance_mask6'][1].any()
for key in ('arm_boxes_raw6','pre_constraint_geometry6','acceptance_mask6','point_support_sha256'):assert np.array_equal(base[key],one[key]),key
assert np.array_equal(zero['arm_boxes_raw6'],np.tile(source,(4,1))) and np.array_equal(zero['acceptance_mask6'],base['acceptance_mask6'])
assert np.array_equal(rgbzero['arm_boxes_raw6'][1],base['arm_boxes_raw6'][2])
assert np.array_equal(one['arm_boxes_raw6'][1:,3:],np.tile(source[3:],(3,1)))
report={'state':'independent_synthetic_R11_weighted_bridge_pass_not_task_result','program_sha256':sha(__file__),'weighted_geometry_sha256':sha(B/'crossplatform/round11/weighted_geometry.py'),'R10_G_sha256':sha(B/'crossplatform/round10/visibility_geometry.py'),'allone_raw_preconstraint_acceptance_support_bitexact':True,'allzero_exact_source_same_acceptance':True,'RGBzero_allonefaces_equals_original_noRGB_coordinate':True,'unobserved_size_axes_exact_source':True,'no_actual_target_RGB_points_GT_caches_models_GPU_or_scores':True};q=D/'WEIGHTED_BRIDGE_INDEPENDENT_CPU_FIXTURE.json';assert not q.exists();q.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
