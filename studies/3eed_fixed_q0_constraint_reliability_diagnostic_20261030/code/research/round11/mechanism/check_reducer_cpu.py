"""Synthetic dictionaries/arrays only; no actual sealed experiment outputs opened."""
import importlib.util,json,hashlib,copy
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('reduce_r11',D/'reduce_mechanism.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
faces=['0:-1','0:1','1:-1','1:1','2:-1','2:1']
def fixture(ident='synthetic',status='association_supported'):
    supported=status=='association_supported';defined=status!='association_undefined';case=.8 if supported else 0. if defined else None
    scores=dict(zip(m.CONTROLS,[case,None,-.1,-.2,8.,.9]))
    rr={'face_weights':dict.fromkeys(faces,.8 if supported else 0.),'RGB_weight':.7 if supported else 0.,'RGB_weight_defined':defined,'case_reliability':case,'association_status':status,'axis_reliability6':[.8 if supported else 0.,0,0,0,0,0],'axis_reliability_defined6':[defined,False,False,False,False,False],'control_scores':scores,'active_face_keys':['0:1'],'raw_face_weights':dict.fromkeys(faces,.8),'raw_face_weight_defined':dict.fromkeys(faces,True),'raw_RGB_weight':.7,'raw_RGB_weight_defined':True,'raw_axis_reliability6':[.8,0,0,0,0,0],'raw_axis_reliability_defined6':[True,False,False,False,False,False],'raw_case_reliability':.8,'features':{'faces':{'0:1':{'defined':True,'reason':'fixture','counts_base_A_B':[8,4,4],'physical_support_count_control':8,'final_plane_residual_m':.2}},'mask_stability':{'score':.9,'IoU':.95},'component_ambiguity':{'conflict_observed':status=='association_conflicted','substantial_component_count':1,'fragmentation_purity':1.,'ambiguity_factor':1.},'crossheldout_foreground_support':1.,'partition_segmentation':[{'valid':True,'reason':'synthetic'}]*2},'partition_hashes':{'partition_unique_XYZ_counts':[20,20],'disjoint_physical_observations':True,'same_XYZ_same_partition':True,'normal_graphs_must_be_rebuilt_per_partition':True},'undefined_reasons':[] if defined else ['synthetic_undefined']}
    b=np.tile([0,0,0,1,1,1.],(11,1)).astype(np.float32);b[1,0]=.1;b[2,0]=.11
    if supported:b[3,0]=.05;b[4,0]=.04
    keep=np.ones(6,bool)
    for j in range(5,11):b[j]=b[1]
    vals=list(scores.values());active=np.array([True,False,False,False,False,False])
    z={'sample_id':np.array(ident),'q_source':np.array(7),'arm_names':np.array(m.ARMS),'arm_query_index':np.full(11,7),'arm_boxes_raw6':b,'atomic_active6':active,'base_acceptance_mask6':active,'coordinate_reliability6':np.array(rr['axis_reliability6']),'coordinate_defined6':np.array(rr['axis_reliability_defined6']),'RGB_reliability':np.array(rr['RGB_weight']),'RGB_defined':np.array(defined),'case_reliability':np.array(case if case is not None else 0),'case_defined':np.array(case is not None),'control_score_names':np.array(m.CONTROLS),'control_scores':np.array([v if v is not None else 0 for v in vals]),'control_defined':np.array([v is not None for v in vals]),'gate_selection':keep,'association_status':np.array(status),'face_keys':np.array(faces),'face_weights':np.array([rr['face_weights'][k] for k in faces])}
    solver={'solver':{'full_rgb':{'success':True,'status':'converged_bounded_result' if supported else 'all_constraints_zero_source_prior_exact','iterations':3 if supported else 0,'numerical_invalid':False}}}
    t={'id':ident,'platform':'synthetic_platform','base_actual_change':True,'reliability':rr,'gates':dict.fromkeys(m.CONTROLS,True),'geometry_unique_XYZ':40,'RGB':{'state':'point_seeded_grabcut_component','reason':'synthetic'},'base_geometry':{'diagnostics':solver},'weighted_solver':solver,'uniform_solver':solver,'uniform_means':{}}
    return m.one(t,z)
rows=[fixture(),fixture('undefined','association_undefined'),fixture('conflict','association_conflicted')]
a=m.aggregate(rows)
assert a['n']==a['baseC']==3 and a['case_effective_in_C']['NA']==1 and a['case_effective_in_C']['zero_count']==1
assert a['raw_positive_face_effective_zero_count']==2 and a['raw_positive_RGB_effective_zero_cases']==2
assert a['face_effective_defined']==2 and a['RGB_effective_defined_in_C']==2
assert a['arms']['weighted_constraints']['actual_nonzero_samples']==1
assert a['controls']['reliability']['selected_score_NA']==1 and a['controls']['reliability']['selected_case_defined_zero']==1
assert a['solvers']['weighted_constraints']['statuses']['all_constraints_zero_source_prior_exact']==2
assert a['case_raw_in_C']['largest_exact_tie']==3
json.dumps(a,allow_nan=False)
d={'state':'PASS_synthetic_R11_reducer_only','raw_effective_gate_distinction':True,'defined_conflict_zero_vs_undefined_NA':True,'case_ties_and_selected_NA_retained':True,'actual_nonzero_distinct_from_accepted':True,'uniform_denominator_flags_preserved':True,'zero_constraint_analytic_solver_distinct':True,'no_actual_data_GT_media_models_or_GPU_used':True,'reducer_sha256':hashlib.sha256((D/'reduce_mechanism.py').read_bytes()).hexdigest()}
(D/'CPU_FIXTURE.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))
