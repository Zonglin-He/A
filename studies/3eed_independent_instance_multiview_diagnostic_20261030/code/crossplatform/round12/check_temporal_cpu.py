"""Synthetic SE3/flow/depth engineering fixture. No real files, models or GT."""
from pathlib import Path
from dataclasses import asdict
import hashlib,json,sys,copy
import numpy as np
import cv2
D=Path(__file__).resolve().parent;sys.path.insert(0,str(D))
import temporal_association as t
from camera_projection import CameraSpec,raw_to_model_points

def main():
    cv2.setNumThreads(1)
    rng=np.random.default_rng(12);b=rng.uniform(-3,3,(240,3));b[:,2]+=10
    ax=.12;c,s=np.cos(ax),np.sin(ax);T=np.eye(4);T[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]];T[:3,3]=[.3,-.2,.1]
    a=t.transform_points(b,T)+rng.normal(0,.001,b.shape)
    hold=t.spatial_split(b,7);fit=np.flatnonzero(~hold);a[fit[:24]]+=rng.uniform(1,3,(24,3))
    out=t.estimate_from_correspondences(points_A=a,points_B=b,seed=7)
    assert out['accepted'] and out['provided_pose_used_for_fit_or_selection'] is False
    assert np.max(np.abs(out['T_raw_B_to_A']-T))<.001
    wrong=np.eye(4);wrong[:3,3]=[40,-20,30]
    poses=t.estimate_from_correspondences(points_A=a,points_B=b,seed=7,provided_pose_A=wrong,provided_pose_B=np.eye(4))
    assert np.array_equal(out['T_raw_B_to_A'],poses['T_raw_B_to_A']) and poses['accepted']==out['accepted']
    assert poses['provided_pose_diagnostic']['hypotheses']['inv_pose_A_at_pose_B']['holdout']['median_m']>20
    # Holdout changes never alter fit estimator, only the acceptance measurement.
    bad=a.copy();bad[hold]+=10
    failed=t.estimate_from_correspondences(points_A=bad,points_B=b,seed=7)
    assert not failed['accepted'] and np.array_equal(failed['T_raw_B_to_A'],out['T_raw_B_to_A'])
    perm=rng.permutation(len(a));per=t.estimate_from_correspondences(points_A=a[perm],points_B=b[perm],seed=7)
    assert np.array_equal(per['T_raw_B_to_A'],out['T_raw_B_to_A'])
    inv=t.inverse_se3(T);assert np.max(abs(inv@T-np.eye(4)))<1e-12
    try:t.validate_se3(np.diag([-1,1,1,1]));raise AssertionError('reflection accepted')
    except ValueError:pass
    # Correct raw/model conjugation, especially Drone z offset under rotation.
    k=np.array([[100.,0,64],[0,100.,64],[0,0,1.]])
    ca=CameraSpec('drone',k,np.eye(4),np.zeros(5));cb=CameraSpec('drone',k,np.eye(4),np.zeros(5))
    Tx=T.copy();Tx[:3,:3]=[[1,0,0],[0,c,-s],[0,s,c]]
    tm=t.raw_pair_to_model_transform(Tx,ca,cb)
    expected=raw_to_model_points(t.transform_points(b,Tx),'drone')
    got=t.transform_points(raw_to_model_points(b,'drone'),tm)
    assert np.max(abs(expected-got))<1e-12
    assert np.max(abs(t.transform_points(raw_to_model_points(b,'drone'),Tx)-expected))>.01
    # Actual SIFT+RGBD API on artificial full-frame texture and static planar depth.
    imB=rng.integers(0,256,(128,128,3),np.uint8);imA=cv2.warpAffine(imB,np.array([[1.,0,2.],[0,1.,0]]),(128,128))
    uv=np.array([(x,y) for y in range(4,124,2) for x in range(4,124,2)],float)
    pB=np.c_[(uv[:,0]-64)/10,(uv[:,1]-64)/10,np.full(len(uv),10.)];pA=pB+[.2,0,0]
    camera=CameraSpec('quad',k,np.eye(4),np.zeros(5))
    pair=t.estimate_pair_registration(rgb_A=imA,raw_points_A=pA,camera_A=camera,rgb_B=imB,raw_points_B=pB,camera_B=camera,seed=5)
    assert pair['accepted'] and np.max(abs(pair['T_raw_B_to_A'][:3,3]-[.2,0,0]))<1e-8
    assert not pair['observation_diagnostics']['background_only_qualification']
    # Same appearance does not bypass static3D support, and time never implies velocity.
    static=t.check_static_object_support(points_A=pA[:20],points_B=pB[:20],T_raw_B_to_A=pair['T_raw_B_to_A']);assert static['accepted']
    moving=t.check_static_object_support(points_A=pA[:20]+[1,0,0],points_B=pB[:20],T_raw_B_to_A=pair['T_raw_B_to_A']);assert not moving['accepted']
    ma=np.zeros((1,128,128),bool);mb=ma.copy();ma[0,30:95,22:82]=True;mb[0,30:95,20:80]=True
    flow=t.match_instance_masks(rgb_A=imA,rgb_B=imB,masks_A=ma,masks_B=mb,class_ids_A=np.array([1]),class_ids_B=np.array([1]))
    assert len(flow['matches'])==1 and flow['not_literal_caption_identity_verified'] is True
    # Duplicating an equally plausible sameclass mask must reject rather than nearest-select.
    amb=t.match_instance_masks(rgb_A=imA,rgb_B=imB,masks_A=ma,masks_B=np.repeat(mb,2,axis=0),class_ids_A=np.array([1]),class_ids_B=np.array([1,1]));assert not amb['matches']
    f=flow['flow_points_A_to_B'][0];bound=t.bind_raw_depth_to_pixel_pairs(pixel_A=f['start_xy'],pixel_B=f['end_xy'],raw_points_A=pA,camera_A=camera,raw_points_B=pB,camera_B=camera,image_hw_A=(128,128),image_hw_B=(128,128),include_mask_A=ma[0],include_mask_B=mb[0]);assert bound['n_unique_physical_pairs']>=6
    empty=t.estimate_pair_registration(rgb_A=np.zeros_like(imA),raw_points_A=pA,camera_A=camera,rgb_B=np.zeros_like(imB),raw_points_B=pB,camera_B=camera);assert not empty['accepted'] and empty['T_raw_B_to_A'] is None
    lineB=np.r_[np.c_[np.linspace(0,10,31),np.zeros((31,2))],np.array([[5,1,0],[3,-1,0],[8,1,0]])]
    lineA=lineB.copy();lineA[-3:,1]*=1.7
    degenerate=t.estimate_from_correspondences(points_A=lineA,points_B=lineB,seed=0)
    assert not degenerate['accepted'] and degenerate['T_raw_B_to_A'] is None and degenerate['status']=='degenerate_fit_consensus'
    assert degenerate['n_fit']==27 and degenerate['n_holdout']==7
    receipt={'state':'PASS_CPU_SYNTHETIC_ONLY_not_real_observation_correspondence','winning_noncollinear_hypothesis_degenerate_final_consensus_returns_NA':True,'sensor_SE3_recovers_known_rigid_transform_with_fit_outliers':True,'provided_pose_never_affects_estimate_or_acceptance':True,'independent_spatial_holdout_can_reject_without_changing_fit':True,'permutation_invariant_correspondence_estimation':True,'reflection_rejected_and_inverse_correct':True,'Drone_model_shift_correctly_conjugated':True,'synthetic_fullimage_SIFT_and_nearest3px_depth_pipeline_pass':True,'independent_mask_optical_flow_and_cycle_unique_match':True,'sameclass_equal_mask_ambiguity_rejected':True,'moving_or_wrong3D_support_rejected_without_motion_refit':True,'empty_texture_source_NA_no_velocity_claim':True,'synthetic_SIFT_pair_counts':{'fit':pair['n_fit'],'holdout':pair['n_holdout'],'RGBD_unique':pair['n_pairs']},'config':asdict(t.DEFAULT_CONFIG),'code_sha256':{n:hashlib.sha256((D/n).read_bytes()).hexdigest() for n in ('temporal_association.py','check_temporal_cpu.py')},'no_real_data_models_GPU_GT_or_assets_used':True}
    (D/'CPU_FIXTURE.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
