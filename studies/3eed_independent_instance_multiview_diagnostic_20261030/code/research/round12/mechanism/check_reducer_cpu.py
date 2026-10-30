"""Synthetic R12 census schema only; no actual experiment outputs opened."""
from pathlib import Path
import hashlib,json,importlib.util,copy
import numpy as np
D=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('reduce_r12',D/'reduce_mechanism.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
def sol():return {'diagnostics':{'solver':{'full_rgb':{'success':True,'status':'converged_bounded_result','iterations':2,'numerical_invalid':False}}}}
def attempt(alias,mode):
    success=mode!='reg_reject';v={'frame_alias':alias,'status':mode,'registration_backend':'one_depth_sensor_PnP_v1','registration':{'accepted':success,'status':'synthetic','observation_diagnostics':{'background_only_qualification':success},'n_fit':12,'n_holdout':6,'holdout_residuals':{'n':25}},'verified':mode=='verified','contradiction':mode=='contradiction'}
    if success:v.update(neighbor_mask_index=0,target_depthbound_unique_pairs=6,static_support={'accepted':mode=='verified','residuals':{'n':6}})
    return v
def example(ident,mode):
    b=np.tile([0,0,0,1,1,1],(8,1)).astype(np.float32);acc=np.zeros((8,6),bool)
    roles=[{'role':'previous','frame_alias':'previous'},{'role':'next','frame_alias':'next'}]
    d={'target_noun':{'scope':'supported_family','COCO_class_family':[3,6,8]},'anchor_association':{'status':'unique_hypothesis','candidates':[0],'candidate_tests':[{'instance_index':0,'visible_unique_foreground_in_source_ROI':10}]},'selected_anchor_mask_index':0,'grabcut':{'geometry':sol()},'single_geometry':sol()}
    b[1,0]=.1;b[2,0]=.2;acc[[1,2,5],0]=True
    if mode=='main':
        b[3,0]=.15;b[7,0]=.25;acc[[3,6,7],0]=True
        d['neighbor_attempts']=[attempt('previous','verified'),attempt('next','reg_reject')]
        g=sol();g.update(arm_query_index=[7]*4,rgb_mask_sha256='1'*64);g['diagnostics']['canonical_allowed_model_mapping']={'applied':True,'same_raw_to_model_then_unique_return_inverse':True,'allowed_canonical_count':45,'model_unique_count':1040,'allowed_model_XYZ_sha256':'2'*64}
        d['multiview_geometry']=g;d['unfiltered_multiview_geometry']=copy.deepcopy(g)
        d['multiview']={'verified_neighbor_n':1,'contradiction_veto':False,'merged_raw_unique_n':1040,'unfiltered_merged_raw_unique_n':1070,'new_raw_unique_count':40,'unfiltered_new_raw_unique_count':70,'allowed_corroborated_raw_XYZ_count':45,'neighbors':[{'visible_instance_points_before_transform':100,'anchor_FOV':90,'anchor_mask':80,'anchor_mask_and_source_ROI':70,'pixel_within3px':60,'pixel_and_depth_within_threshold':50,'fully_corroborated_neighbor_points':40,'matched_anchor_raw_ids_unique':[0,1,2,3,4]}]}
    elif mode=='veto':
        d['neighbor_attempts']=[attempt('previous','verified'),attempt('next','contradiction')];d['multiview']={'contradiction_veto':True,'verified_neighbor_n':1,'neighbors':[]}
    else:
        d={'target_noun':{'scope':'NA_unsupported','COCO_class_family':[]}}
        roles=[{'role':'previous','frame_alias':'previous'},{'role':'next','status':'missing_prelocked_no_replacement'}];b[:]=b[0];acc[:]=False
    b[5]=b[0]+np.float32(.5)*(b[2]-b[0]);b[6]=b[0]+np.float32(.5)*(b[3]-b[0])
    if mode!='unsupported':d['wrongview']={'attempt':attempt('wrong','reg_reject')}
    t={'id':ident,'platform':'synthetic','q_source':7,'roles':roles,'wrong_frame_alias':'wrong','anchor_raw_full_points':1000,'diagnostics':d}
    z={'sample_id':np.array(ident),'q_source':np.array(7),'arm_names':np.array(r.ARMS),'arm_query_index':np.full(8,7),'arm_status':np.array(['synthetic']*8),'arm_boxes_raw6':b,'acceptance_mask6':acc,'actual_changed_axes':b!=b[0],'point_support_sha256':np.array(['0'*64]*8)}
    return r.one(t,z)
rows=[example('m','main'),example('v','veto'),example('u','unsupported')];d=r.aggregate(rows)
assert d['n']==3 and d['real_views']['retrieved_preselected']==5 and d['real_views']['actually_attempted']==4 and d['real_views']['not_attempted_anchor_or_noun']==1
assert d['wrong_views']['retrieved_preselected']==3 and d['wrong_views']['actually_attempted']==2
assert d['real_views']['global_PnP_pass']==3 and d['real_views']['unique_mask_LK_track']==3 and d['real_views']['depth_pairs_at_least6']==3 and d['real_views']['static_pass']==2 and d['real_views']['static_contradiction']==1
assert d['MV_admitted_anchors']==1 and d['static_contradiction_veto_anchors']==1 and d['main_unfiltered_different_float32']==1
assert d['arms']['single_instance_direct']['changed_samples']==2 and d['arms']['multiview_instance_direct']['changed_samples']==1
assert d['arms']['single_instance_half']['solver_objects']==0 and d['arms']['single_instance_half']['half_no_new_solver']
assert d['corroborating_anchor_raw_unique_union_n']['sum']==5 and d['anchor_crossview_corroborated_fraction']['mean']==.5
json.dumps(d,allow_nan=False)
receipt={'state':'PASS_synthetic_R12_mechanism_schema_not_real_observation_or_quality_result','retrieved_not_attempted_registration_failure_denominators_distinct':True,'PnP_track_depth_static_funnel_verified':True,'veto_single_preservation_and_gate_identity':True,'corroborated_points_anchor_union_and_allowed_support':True,'main_unfiltered_differences_no_quality_claim':True,'half_no_extra_solver':True,'source_q0_fixed_and_no_GT_used':True,'reducer_sha256':hashlib.sha256((D/'reduce_mechanism.py').read_bytes()).hexdigest(),'actual_outputs_or_media_points_features_not_read':True}
(D/'CPU_FIXTURE.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
