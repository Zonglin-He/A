"""Necessary synthetic visibility/partial-prior/paired-control checks only."""
from pathlib import Path
from dataclasses import asdict
import hashlib,json,sys
import numpy as np
D=Path(__file__).resolve().parent;sys.path.insert(0,str(D.parent/'round7'))
from camera_projection import CameraSpec,project_model_points
from visibility_geometry import generate_geometry_targets,DEFAULT_CONFIG

def main():
    k=np.array([[50.,0,64],[0,50.,64],[0,0,1.]])
    e=np.eye(4);e[:3,:3]=np.diag([1.,-1.,-1.]);e[:3,3]=[0,0,10]
    camera=CameraSpec('quad',k,e,np.zeros(5));source=np.array([0,0,5,2,1,2.])
    plane=np.array([(x,y,6.2) for x in np.linspace(-1,1,11) for y in np.linspace(-.5,.5,11)],np.float32);ids=np.arange(len(plane),dtype=np.int64)
    mask=np.zeros((128,128),bool);mask[56:73,49:80]=True
    def call(points=plane,raw=ids,rgb=mask,valid=True,cam=camera):return generate_geometry_targets(points_fit=points,raw_indices=raw,source_box6=source,q0=7,camera_spec=cam,rgb_mask2d=rgb,segmentation_valid=valid)
    out=call();assert out['acceptance_mask6'][1,2] and not out['acceptance_mask6'][1,3:].any()
    assert np.array_equal(out['acceptance_mask6'][1],out['acceptance_mask6'][2])
    assert out['point_support_sha256'][1]==out['point_support_sha256'][2] and np.array_equal(out['support_ids_by_arm'][1],out['support_ids_by_arm'][2])
    assert np.all(out['arm_query_index']==7) and np.array_equal(out['arm_boxes_raw6'][1:,3:],np.tile(source[3:],(3,1)))
    assert out['arm_boxes_raw6'][2,2]>source[2] and out['arm_boxes_raw6'][2,2]<5.2
    assert np.linalg.norm(out['arm_boxes_raw6'][1]-out['arm_boxes_raw6'][2])>0
    # A small visible fragment must NEVER shrink unobserved full dimensions.
    fragment=np.zeros_like(mask);fragment[60:69,60:69]=True;frag=call(rgb=fragment)
    assert np.array_equal(frag['arm_boxes_raw6'][1:,3:],np.tile(source[3:],(3,1)))
    # Ground on the away half is not a camera-facing roof, even with a wide mask.
    ground=plane.copy();ground[:,2]=3.9;g=call(points=ground)
    assert not g['acceptance_mask6'][1:].any() and np.array_equal(g['arm_boxes_raw6'],np.tile(source,(4,1)))
    # Replacement rows cannot manufacture point/normal/face support.
    repeated=call(points=np.tile(plane,(4,1)),raw=np.tile(ids,4))
    assert np.array_equal(out['arm_boxes_raw6'],repeated['arm_boxes_raw6']) and np.array_equal(out['point_support_sha256'],repeated['point_support_sha256'])
    perm=np.random.default_rng(7).permutation(len(plane));permuted=call(points=plane[perm],raw=ids[perm]);assert np.array_equal(out['arm_boxes_raw6'],permuted['arm_boxes_raw6'])
    empty=call(rgb=np.zeros_like(mask),valid=False);assert np.array_equal(empty['arm_boxes_raw6'],np.tile(source,(4,1)))
    # Camera inside a source-axis slab does not create near-face observability.
    inside_e=np.eye(4);inside=CameraSpec('quad',k,inside_e,np.zeros(5));side=np.array([(6.,y,z) for y in np.linspace(-.5,.5,11) for z in np.linspace(4,6,11)],np.float32)
    # The above side lies outside source crop; no false observed axis can appear.
    invalid=call(points=side,cam=inside);assert not invalid['acceptance_mask6'][1:].any()
    assert 'point_plane_plus_source_size_prior' in out['provenance'][2,2]
    r={'state':'synthetic_CPU_visibility_geometry_contract_pass_not_real_G_result','raw_index_replacement_dedup_invariant':True,'point_permutation_invariant':True,'same_main_RGBsupport_and_acceptance_exact':True,'one_camera_facing_plane_center_conditional_on_source_size':True,'partial_fragment_never_sets_full_size':True,'away_ground_not_near_roof':True,'source_prior_size_preserved_without_pair':True,'RGB_one_sided_constraint_changes_geometry_solution_without_fullbbox_equality':True,'no_new_targetquery_or_candidate_fusion':True,'config':asdict(DEFAULT_CONFIG),'geometry_only_center_delta_m':(out['arm_boxes_raw6'][2,:3]-source[:3]).tolist(),'full_minus_geometry_only_center_m':(out['arm_boxes_raw6'][1,:3]-out['arm_boxes_raw6'][2,:3]).tolist(),'observed_axes':out['acceptance_mask6'][1].tolist(),'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (D/'visibility_geometry.py',Path(__file__))},'no_GPU_GT_real_points_RGB_or_annotations_used':True}
    (D/'CPU_VISIBILITY_GEOMETRY_FIXTURE.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':main()
