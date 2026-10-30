"""Synthetic array/trace reduction fixture only. No research outputs are opened."""
import importlib.util,json,hashlib
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('mechanism_reduce',D/'reduce_mechanism.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
b=np.tile([0,0,0,2,1,1],(4,1)).astype(np.float32);b[1,0]=.1;b[2,0]=.05;b[3,0]=.2
accept=np.zeros((4,6),bool);accept[1:,0]=True
z={'sample_id':np.array('synthetic'),'q_source':np.array(4),'arm_names':np.array(m.ARMS),'arm_query_index':np.full(4,4),'arm_boxes_raw6':b,'pre_constraint_geometry6':np.tile(b[2],(4,1)),'acceptance_mask6':accept,'actual_changed_axes':b!=b[0],'point_support_sha256':np.array(['0'*64,'1'*64,'1'*64,'2'*64]),'arm_status':np.array(['source','converged_bounded_result','finite_nonconverged_bounded_result','numerical_invalid_source_fallback'])}
# Numerical fallback must equal source even if its frozen acceptance remains open.
z['arm_boxes_raw6'][3]=b[0];z['actual_changed_axes']=b!=b[0]
terms={'prior':.1,'observed_surface':.2,'RGB_one_sided_containment':.3,'total':.6}
def s(status):return {'success':status=='converged_bounded_result','status':status,'numerical_invalid':status=='numerical_invalid_source_fallback','iterations':2,'initial_terms':terms,'final_terms':terms}
t={'id':'synthetic','platform':'synthetic_platform','model_input_rows':16,'unique_raw_indices':8,'geometry_unique_XYZ_rows':7,'RGB':{'state':'point_seeded_grabcut_component','reason':'synthetic'},'geometry':{'diagnostics':{'global_gate_reason':'shared_input_acceptance','geometric_support_count':6,'RGB_support_count':3,'source_face_evidence_RGB':{},'solver':{name:s(str(z['arm_status'][a])) for a,name in enumerate(m.ARMS) if a},'RGB_boundary_sample_count':10}}}
r=m.one(t,z);v=m.aggregate([r]);assert v['n']==1 and v['segmentation_valid']==1 and v['accepted_axis_counts6']==[1,0,0,0,0,0]
assert v['full_noRGB_different_float32']==1 and v['RGB_over_geometric_support']['mean']==.5
assert v['arms']['no_rgb_coordinate']['finite_nonconverged']==1
assert v['arms']['rgb_gated_pointmask_ablation']['numerical_invalid']==1
assert v['arms']['rgb_gated_pointmask_ablation']['actually_changed_samples']==0
json.dumps(v,allow_nan=False)
# All-source fallback still stays in every denominator and has missing solver terms.
z['arm_boxes_raw6'][:]=z['arm_boxes_raw6'][0];z['actual_changed_axes'][:]=False;z['acceptance_mask6'][:]=False
t['RGB']={'state':'segmentation_rejected','reason':'synthetic_missing_support'};t['geometry']={'diagnostics':{'global_gate_reason':'invalid_or_empty_RGB_support'}}
r0=m.one(t,z);out=m.aggregate([r,r0]);assert out['n']==2 and out['segmentation_valid']==1 and out['arms']['full_rgb']['solver_executed']==1
assert out['full_noRGB_identical_float32']==1 and out['full_noRGB_different_float32']==1
json.dumps(out,allow_nan=False)
receipt={'state':'PASS_synthetic_reducer_only_not_real_mechanism_result','preserves_all_source_and_missing_solver_denominators':True,'finite_nonconvergence_and_numerical_fallback_distinct':True,'accepted_vs_changed_distinct':True,'RGB_support_ratio_and_main_coordinate_difference':True,'no_research_data_GPU_GT_or_media_opened':True,'program_sha256':hashlib.sha256((D/'reduce_mechanism.py').read_bytes()).hexdigest()}
(D/'CPU_FIXTURE.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
