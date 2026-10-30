"""Synthetic orchestration/G tests only; temporal SIFT/LK has separate fixtures."""
from pathlib import Path
from dataclasses import asdict
import json,hashlib,sys,copy
import numpy as np
D=Path(__file__).resolve().parent;sys.path[:0]=[str(D),str(D.parent/'round10')]
import multiview_feedback as f
import geometry_feedback as g
import visibility_geometry as old
from camera_projection import CameraSpec

def main():
    K=np.array([[50.,0,64],[0,50.,64],[0,0,1.]]);E=np.eye(4);E[:3,:3]=np.diag([1.,-1.,-1.]);E[:3,3]=[0,0,10]
    camera=CameraSpec('quad',K,E,np.zeros(5));source=np.array([0,0,5,2,1,2.],np.float32)
    P=np.array([(x,y,6.2) for x in np.linspace(-1,1,11) for y in np.linspace(-.5,.5,11)],np.float32)
    mask=np.zeros((128,128),bool);mask[56:73,49:80]=True;rgb=np.zeros((128,128,3),np.uint8);rgb[mask]=[220,70,20]
    inst={'masks_bool':mask[None],'boxes_xyxy':np.array([[49,56,80,73]],np.float32),'scores':np.array([.9],np.float32),'class_ids':np.array([3]),'instance_indices':np.array([0])}
    A={'frame_alias':'anchor','rgb':rgb,'raw_points':P,'camera':camera,'instances':inst}
    # No allowed filter must be the exact original objective, not a new solver.
    kw={'points_fit':P,'raw_indices':np.arange(len(P)),'source_box6':source,'q0':7,'camera_spec':camera,'rgb_mask2d':mask,'segmentation_valid':True}
    oldout=old.generate_geometry_targets(**kw);newout=g.generate_geometry_targets(**kw)
    assert np.array_equal(oldout['arm_boxes_raw6'],newout['arm_boxes_raw6'])
    none=g.generate_geometry_targets(**kw,allowed_raw_ids=np.empty(0,np.int64));assert np.array_equal(none['arm_boxes_raw6'],np.tile(source,(4,1)))
    assert f.parse_target_family('A truck next to a person')[1]==(3,6,8)
    assert f.parse_target_family('A biker next to a red car')[1]==()
    # Distinct source-independent masks must remain ambiguous; never choose by score.
    amb=copy.deepcopy(A);amb['instances']={k:np.repeat(v,2,axis=0) for k,v in inst.items()};amb['instances']['scores']=np.array([.9,.8],np.float32)
    ai,ad=f._unique_target(amb,source,(3,6,8),f.DEFAULT_CONFIG);assert ai is None and len(ad['candidates'])==2
    # The actual routing hook is the fixed PnP function, never old SIFT/pose fallback.
    oldpnp=f.sensor_pnp.estimate_pair_registration;old_sift=f.temporal.estimate_pair_registration;called=[]
    def pnp_spy(**kwargs):
        assert not any('pose' in k for k in kwargs)
        assert kwargs['excluded_mask_A'].dtype==bool and np.array_equal(kwargs['excluded_mask_A'],mask)
        called.append(True)
        return {'accepted':False,'status':'synthetic_PnP_rejection','T_raw_B_to_A':None,'observation_diagnostics':{'background_only_qualification':False}}
    def wrong_backend(**kwargs):raise AssertionError('old registration backend must not run')
    f.sensor_pnp.estimate_pair_registration=pnp_spy;f.temporal.estimate_pair_registration=wrong_backend
    try:
        routed=f._attempt_view(A,{**A,'frame_alias':'PnP_neighbor'},0,5)
        assert len(called)==1 and not routed['verified'] and routed['registration_backend']=='one_depth_sensor_PnP_v1'
    finally:f.sensor_pnp.estimate_pair_registration=oldpnp;f.temporal.estimate_pair_registration=old_sift
    try:
        f.build_geometry_feedback(anchor=A,neighbors=[],wrong_neighbor=None,source_box6=source,q0=7,caption='car',config=f.FeedbackConfig(registration_backend='provided_pose'))
        raise AssertionError('unregistered backend accepted')
    except ValueError:pass
    original=f._attempt_view
    def fixed_attempt(anchor,neighbor,target_index,seed):
        alias=neighbor['frame_alias'];base={'frame_alias':alias,'verified':False,'contradiction':False}
        if alias=='static_contradiction':return {**base,'status':'contradiction_static_target_support','contradiction':True,'target_depthbound_unique_pairs':6}
        if alias=='insufficient':return {**base,'status':'NA_insufficient_depthbound_track_pairs_not_proven_motion','target_depthbound_unique_pairs':2}
        if alias=='wrong':return {**base,'status':'NA_sensor_registration_or_background_fit_protocol'}
        added=P.copy();added[:,0]+=.03
        if alias=='uncorroborated':added[:,2]+=.4
        return {**base,'status':'verified_static_multiview_hypothesis_not_literal_identity','verified':True,'registration':{'T_raw_B_to_A':np.eye(4)},'neighbor_visible_foreground_raw':added,'target_depthbound_unique_pairs':8}
    f._attempt_view=fixed_attempt
    try:
        def frame(alias):return {**A,'frame_alias':alias}
        out=f.build_geometry_feedback(anchor=A,neighbors=[frame('valid'),frame('insufficient')],wrong_neighbor=frame('wrong'),source_box6=source,q0=7,caption='The car near a person')
        assert tuple(out['arm_names'])==f.ARMS and np.all(out['arm_query_index']==7)
        assert out['diagnostics']['multiview']['verified_neighbor_n']==1 and not out['diagnostics']['multiview']['contradiction_veto']
        assert out['diagnostics']['multiview']['unfiltered_ablation_has_same_instance_static_gate_and_views']
        assert out['arm_status'][7]=='unfiltered_trackqualified_maskpoint_aggregation_ablation'
        assert np.array_equal(out['arm_boxes_raw6'][5],source+np.float32(.5)*(out['arm_boxes_raw6'][2]-source))
        assert np.array_equal(out['arm_boxes_raw6'][6],source+np.float32(.5)*(out['arm_boxes_raw6'][3]-source))
        assert np.array_equal(out['arm_boxes_raw6'][4],source)
        veto=f.build_geometry_feedback(anchor=A,neighbors=[frame('valid'),frame('static_contradiction')],wrong_neighbor=None,source_box6=source,q0=7,caption='car')
        assert np.array_equal(veto['arm_boxes_raw6'][[3,6,7]],np.tile(source,(3,1))) and not np.array_equal(veto['arm_boxes_raw6'][2],source)
        assert veto['diagnostics']['multiview']['contradiction_veto']
        unsupported_points=f.build_geometry_feedback(anchor=A,neighbors=[frame('uncorroborated')],wrong_neighbor=None,source_box6=source,q0=7,caption='car')
        assert unsupported_points['diagnostics']['multiview']['allowed_corroborated_raw_XYZ_count']==0
        assert np.array_equal(unsupported_points['arm_boxes_raw6'][3],source)
        assert unsupported_points['diagnostics']['multiview_geometry']['diagnostics']['canonical_allowed_model_mapping']['allowed_canonical_count']==0
        assert unsupported_points['diagnostics']['multiview']['unfiltered_new_raw_unique_count']>0
        # Unfiltered ablation is not substituted when corroborated main is empty.
        assert not np.array_equal(unsupported_points['arm_boxes_raw6'][7],source)
        unsupported=f.build_geometry_feedback(anchor=A,neighbors=[],wrong_neighbor=None,source_box6=source,q0=7,caption='biker')
        assert np.array_equal(unsupported['arm_boxes_raw6'][2:],np.tile(source,(6,1)))
        multi=f.build_geometry_feedback(anchor=amb,neighbors=[],wrong_neighbor=None,source_box6=source,q0=7,caption='car')
        assert np.array_equal(multi['arm_boxes_raw6'][2:],np.tile(source,(6,1)))
        # Foreground additions must survive anchor FOV/mask/sourceROI; all background is not merged.
        inp=out['diagnostics']['multiview'];assert inp['no_whole_background_neighbor_aggregated'] and inp['new_raw_unique_count']>0
    finally:f._attempt_view=original
    receipt={'state':'PASS_synthetic_G_orchestration_not_real_mask_registration_or_task_result','actual_registration_routing_PnP_only_no_oldSIFT_or_pose_fallback':True,'geometry_no_filter_bitexact_R10':True,'independent_samefamily_multicandidate_NA_no_rerank':True,'unsupported_composite_noun_NA_no_GTfallback':True,'fixed_q0_eight_arms':True,'low_depth_pairs_only_neighbor_NA_not_motion_veto':True,'verified_static_contradiction_vetoes_all_MV_but_preserves_single':True,'legacy_named_maskmatch_arm_is_unfiltered_samegate_aggregation_ablation':True,'uncorroborated_points_do_not_enter_main_or_trigger_unfiltered_fallback':True,'allowed_support_ids_mapped_through_canonical_model_XYZ':True,'fixed_half_displacements_exact':True,'wrongview_no_replacement_sourcefallback':True,'foreground_only_merged_filtered_anchor_FOV_mask_ROI':True,'config':asdict(f.DEFAULT_CONFIG),'code_sha256':{n:hashlib.sha256((D/n).read_bytes()).hexdigest() for n in ('multiview_feedback.py','geometry_feedback.py','check_feedback_cpu.py','run_feedback.py')},'orchestration_fixture_uses_stub_verified_attempts_temporal_engineering_tested_separately':True,'no_real_data_GT_model_GPU_used':True}
    (D/'FEEDBACK_CPU_FIXTURE.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
