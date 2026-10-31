"""Independent synthetic B invariants, no sensor data/labels/IoU/models."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
B=Path(__file__).resolve().parents[3];sys.path[:0]=[str(B/'crossplatform/round13'),str(B/'crossplatform/round7')]
from geometry_observation import generate_observation_targets,DEFAULT_CONFIG,config_sha256
from camera_projection import CameraSpec
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=np.array([0,0,5,2,2,2],np.float32);mask=np.zeros((160,160),bool);mask[73:87,73:87]=True
camera=CameraSpec('quad',np.array([[100.,0,80],[0,100.,80],[0,0,1]]),np.eye(4),missing_distortion_policy='assume_zero_declared')
x,y=np.meshgrid(np.linspace(-.3,.3,4),np.linspace(-.3,.3,4));inside=np.c_[x.ravel(),y.ravel(),np.full(x.size,5)].astype(np.float32)
def run(p,m=mask,s=source):return generate_observation_targets(points_support_model=p,mask_bool=m,camera_spec=camera,source6=s,q0=17)
checks={};r=run(inside)
assert np.array_equal(r['arm_boxes_raw6'],np.tile(source,(4,1))) and not r['active_mask6'].any();checks['strictly_inside_observations_do_not_pull_center_or_shrink_to_surface']=True
outside=inside.copy();outside[:,0]+=1.5;t=run(outside);assert t['actual_changed_axes'][2].any();assert np.all(t['arm_boxes_raw6'][:,3:]>=source[3:]);assert np.all(t['arm_boxes_raw6'][:,3:]<=source[3:]*np.float32(1.2)+1e-6);checks['outside_lower_bound_allows_only_bounded_expansion_and_shift']=True
assert np.array_equal(t['arm_boxes_raw6'][3],source+np.float32(.5)*(t['arm_boxes_raw6'][1]-source));assert np.array_equal(t['active_mask6'][1],t['active_mask6'][2]);assert len(set(t['point_support_sha256']))==1 and len(set(t['mask_sha256_by_arm']))==1;assert np.array_equal(t['arm_query_index'],[17]*4);checks['fixedq_same_support_same_domain_half_exact']=True
u=run(np.repeat(outside[::-1],3,axis=0));assert np.array_equal(u['arm_boxes_raw6'],t['arm_boxes_raw6']) and np.array_equal(u['point_support_sha256'],t['point_support_sha256']);checks['duplicate_permutation_does_not_reweight_support']=True
shiftmask=np.zeros_like(mask);shiftmask[76:84,108:116]=True;s=run(inside,shiftmask);assert s['actual_changed_axes'][1].any() and np.array_equal(s['arm_boxes_raw6'][2],source);checks['RGB_only_violation_offterm_source_without_changing_shared_domain']=True
bad=source.copy();bad[4]=-1;v=run(inside,s=bad);assert np.array_equal(v['arm_boxes_raw6'],np.tile(bad,(4,1))) and v['diagnostics']['gate_reason']=='nonpositive_source_size';checks['invalid_source_not_repaired']=True
out={'state':'PASS_R13_B_INDEPENDENT_SYNTHETIC_ARRAY_CHECKS','checks':checks,'program_sha256':sha(B/'crossplatform/round13/geometry_observation.py'),'auditor_sha256':sha(Path(__file__)),'config_sha256':config_sha256(),'runtime_scope':'pure synthetic CPU, no targetlabels/sensorfiles/newIoU/model/GPU','actual_utility_and_target_identity_not_tested':True}
p=B/'research/round13/evaluation/B_INDEPENDENT_CPU_FIXTURE.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'state':out['state'],'n':len(checks),'sha256':sha(p)}))
