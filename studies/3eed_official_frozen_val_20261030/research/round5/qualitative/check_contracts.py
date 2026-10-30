"""Synthetic projection, matching and fixed-stratum engineering contracts only."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R))
from render_cases import source_to_camera,project_camera,near_clip,corners
from diagnose import correspond,aabb_pair_iou,stratum
m={'image_intrinsic':[[500,0,320],[0,500,200],[0,0,1]],'image_extrinsic':np.eye(4).tolist()}
for suffix,a in enumerate([0,-45,45,-90,90]):
 t=np.deg2rad(a);c,s=np.cos(t),np.sin(t);source=np.array([[c,-s,0],[s,c,0],[0,0,1.]])@np.array([5.,1.,0.]);k,d,M=source_to_camera(m,{'platform':'waymo','waymo_view_suffix':suffix});cam=(M@np.r_[source,1])[:3];assert np.allclose(project_camera(cam[None],k,d)[0],[220,200])
k,d,M=source_to_camera(m,{'platform':'drone'});assert np.allclose(M@np.array([0,0,6.8,1]),[0,0,5,1]);assert near_clip([0,0,-1],[1,0,-2]) is None;assert np.allclose(near_clip([0,0,-1],[1,0,1])[:,2],[.1,1])
a=np.array([[0,0,0,2,2,2],[5,0,0,2,2,2],[10,0,0,2,2,2]],float);b=a[[2,0,1]];cc,cost=correspond(a,b);assert np.array_equal(cc,[1,2,0]);assert np.array_equal(aabb_pair_iou(a,b[cc]),np.ones(3))
assert [stratum(*x) for x in [(1,1,0),(0,1,1),(.4,1,0),(0,.4,0),(0,1,0)]]==['success','annotated_sameclass_other_instance','localization_boundary','candidate_coverage_failure50','other_failure']
assert stratum(.5,.5,0)=='candidate_coverage_failure50';assert stratum(.25,1,1)=='annotated_sameclass_other_instance'
(R/'ENGINEERING_FIXTURES.json').write_text(json.dumps({'state':'synthetic_pass','camera_waymo_all5suffix_to_same_physical_pixels':True,'camera_y_left_projects_to_image_left':True,'drone_zundo':True,'near_plane_crossing_clipped_and_bothbehind_rejected':True,'Hungarian_permutation_identity_recovery':True,'pair_AABB_iou_ones':True,'fixed_stratum_rules_and_strictthresholds_pass':True,'native_GT_data_pixels_or_pointcloud_read':False,'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'diagnose.py',R/'render_cases.py',Path(__file__)]}},indent=2));print('SYNTHETIC_CONTRACTS_PASS')
