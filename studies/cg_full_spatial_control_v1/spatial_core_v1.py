"""Predeclared gamma.5/u.5/link.25; original HC beta.5/context1.5 and finalDP."""
from spatial_common_v1 import *
import numpy as np,torch
from stvg.initialization import initialize
from stvg.predictor import output_writeback
from stvg.evidence import SourceEvidence
from stvg.spatial import SpatialGeometry
from stvg.prototype import one_step
from stvg.current_shared_roi import map_interval

def derive(cap,teacher):
 base.legacy();from studies.cgstvg_full_v1.core import solve
 from cgstvg_interfaces_v9 import pool_single_grid
 from cgstvg_runtime_v9 import clip_native_boxes
 ids=cap['frame_ids'];pos=cap['positions8'];ref=np.asarray(teacher['boxes'],np.float64)
 assert ref.shape==(8,4)and teacher['nested_mask_all_false']and teacher['native_same_track_all8']
 calibration=fixed(R/'outputs/paper_followup_v1/cg_calibrations.json')['hc2']['Transferred']
 shared,_=initialize(cap['Source'],pos,ref,calibration);E=output_writeback(cap['Source'],shared,PROFILE['gamma']);grid=cap['grids']
 ev=SourceEvidence.from_native_masks(grid['spatial'],grid['spatial_valid'],ids,pos,teacher['native_mask_logits'],teacher['input_shape'][-2:],grid['shape'],selected_query=teacher['selected_query'])
 geometry=SpatialGeometry(ev,E,ref);features=pool_single_grid(grid['projected'],grid['valid'],E)
 q,_=one_step(features,cap['q0'],kappa=0,beta=.5);iv=map_interval(ids,q)
 c=dict(dataset='hc2',Source=cap['Source'].copy(),native_map=cap['native_map'],q0=cap['q0'].copy(),logits=cap['logits'].clone(),frame_ids=ids,positions8=pos,E=E,native_boxes=ref,geometry=geometry.state_dict(),features=features,posterior=q,interval=np.asarray(iv),profile=PROFILE,first_aux_DP=False,final_DP=True,GT_read=False)
 a=dict(Source=c['Source'].copy(),Source_interval=np.asarray(c['native_map']),SourceJointMAP=c['Source'].copy(),SourceJointMAP_interval=np.asarray(map_interval(ids,c['q0'])),FullSpatial=clip_native_boxes(solve(geometry,iv,PROFILE)),FullSpatial_interval=np.asarray(iv),frame_ids=np.asarray(ids),positions8=np.asarray(pos))
 return a,c

def replay(c):
 base.legacy();from studies.cgstvg_full_v1.core import solve
 from cgstvg_runtime_v9 import clip_native_boxes
 assert c['profile']==PROFILE and c['first_aux_DP']is False and not c['GT_read']
 q,_=one_step(c['features'],c['q0'],kappa=0,beta=.5);iv=map_interval(c['frame_ids'],q)
 assert np.array_equal(q,c['posterior'])and np.array_equal(iv,c['interval'])
 g=SpatialGeometry.from_cached_geometry(c['geometry'])
 return dict(Source=c['Source'].copy(),Source_interval=np.asarray(c['native_map']),SourceJointMAP=c['Source'].copy(),SourceJointMAP_interval=np.asarray(map_interval(c['frame_ids'],c['q0'])),Conservative=c['conservative_boxes'].copy(),Conservative_interval=c['conservative_interval'].copy(),FullSpatial=clip_native_boxes(solve(g,iv,PROFILE)),FullSpatial_interval=np.asarray(iv),frame_ids=np.asarray(c['frame_ids']),positions8=np.asarray(c['positions8']))

def build(cap,teacher,key):
 tick=time.perf_counter();before,bc,bp,cost=conservative.build(cap,teacher,'hc2',key)
 ce=fixed(base.D/'hc2_sealed.json')['files'][key];assert sha(base.D/ce['path'])==ce['sha256']and sha(base.D/ce['receipt_path'])==ce['receipt_sha256']
 with np.load(base.D/ce['path'])as z:assert all(np.array_equal(v,z[k])for k,v in before.items())
 compatibility=time.perf_counter()-tick;tick=time.perf_counter();a,c=derive(cap,teacher);a['Conservative']=before['FullMAP'].copy();a['Conservative_interval']=before['FullMAP_interval'].copy();c.update(conservative_boxes=a['Conservative'],conservative_interval=a['Conservative_interval']);cold=replay(c);assert set(a)==set(cold)and all(np.array_equal(v,cold[k])for k,v in a.items())
 proof=dict(current_conservative_prediction_sha256=ce['sha256'],current_conservative_receipt_sha256=ce['receipt_sha256'],original_compatibility=bp,original_Source_logits_q0_native_interval_exact=True,current_conservative_all_arrays_exact=True,new_E_ROI_actual_pool=True,new_mask_geometry=True,first_aux_DP=False,final_DP=True,profile=PROFILE)
 return a,c,proof,dict(compatibility_elapsed_wall_seconds=compatibility,selected_fullspatial_elapsed_wall_seconds=time.perf_counter()-tick)
