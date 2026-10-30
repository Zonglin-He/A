"""Syntheticgrammar/camera physics only; no GT/caption outcome selection."""
from pathlib import Path
import json,hashlib,numpy as np
from caption_regions import *
from spatial_teacher import *
R=Path(__file__).resolve().parent
cases=[('A car located left of center in the view. There is a car to its right.','global_left_half'),('The person wearing a red top and holding a phone in the right hand is located left of center in the frame.','global_left_half'),('The car with its left side visible is located on the right middle side of the view.','global_right_half'),('Object type: pedestrian; Position information: middle-right side of the view; Relationship with surroundings: a person ontheleft.','global_right_half'),('The car is located at the center of the view.','explicit_global_horizontal_center'),('The person is located slightly below the center of the view.',None),('A person facing left. There is a car located right of center in the view.',None),('A car located to the left of another car in the view.',None),('A car located both left and right in the view.',None),('A car to the right of a person.',None)]
for text,expected in cases:
 region,diag=parse_target_horizontal_region(text);assert (region.relation if region else None)==expected,(text,region,diag)
# WaymoCamera->vehicle theninverse anddocumentedaxispermutation.
E=np.eye(4);E[:3,3]=[1.5,0,1.2];A=np.array([[0,-1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]],float);T=A@np.linalg.inv(E);xyz=np.array([[11.5,2,1.2],[11.5,-2,1.2],[-8.5,0,1.2]]);cam=np.c_[xyz,np.ones(3)]@T.T;K=np.array([[100,0,100],[0,100,50],[0,0,1]],float);uvw=cam[:,:3]@K.T;uv=uvw[:,:2]/uvw[:,2:];assert uv[0,0]<100 and uv[1,0]>100 and cam[2,2]<0
region,_=parse_target_horizontal_region('The car is located onthe right side of the view.')
# Separatewordfixture here because actualparser requireswordboundaries.
region,_=parse_target_horizontal_region('The car is located on the right side of the view.');prior=np.array([.8,.15,.05]);valid=np.array([True,True,False]);target,d=spatial_posterior(prior,uv,valid,[200,100],region);assert target[1]>prior[1] and target[0]<prior[0] and d['penalty'][2]==0
# Intervalinteriorhaszero penalty, not anattraction toone mean.
assert np.array_equal(interval_violation(np.array([.51,.75,.99]),.5,1),np.zeros(3))
none,nd=spatial_posterior(prior,uv,valid,[200,100],None);assert np.array_equal(none,prior)
out={'state':'passed','scope':'syntheticparserandcameraengineering only','grammar_cases':len(cases),'ignores_otherobjects_bodyorientation_and_verticalonly':True,'Waymo_A_invE_left_right_depth_sign':True,'interval_interior_zero_penalty':True,'unsupported_penalty_zero_normalization_disclosed':True,'no_language_teacher_exactprior':True,'researchGTor_realprojection_read':False,'code_sha256':{p.name:hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in [R/'caption_regions.py',R/'spatial_teacher.py']}}
(R/'REGION_FIXTURES.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
