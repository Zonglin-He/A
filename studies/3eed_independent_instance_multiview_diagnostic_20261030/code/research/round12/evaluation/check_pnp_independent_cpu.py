"""Synthetic PnP transform/heldout review. RGB correspondence front-end stub only.
No actual images/P/GT/model. Tests the one-depth estimator, not SIFT quality.
"""
from pathlib import Path
import sys,hashlib,json
import numpy as np,cv2
B=Path(__file__).resolve().parents[3];D=B/'crossplatform/round12';O=Path(__file__).resolve().parent;sys.path.insert(0,str(D));import sensor_pnp as p
import temporal_association as t
from camera_projection import CameraSpec,raw_to_model_points,project_model_points

def sha(x):return hashlib.sha256(Path(x).read_bytes()).hexdigest()
def main():
 cv2.setNumThreads(1);rng=np.random.default_rng(12312);A=rng.uniform([-4,-3,8],[4,3,12],(400,3));K=np.array([[120.,0,128],[0,120.,128],[0,0,1.]]);EA=np.eye(4);EB=np.eye(4);EA[:3,3]=[.04,-.01,.02];EB[:3,3]=[-.02,.03,0]
 ca=CameraSpec('drone',K,EA,np.zeros(5));cb=CameraSpec('drone',K,EB,np.zeros(5));angle=.025;c,s=np.cos(angle),np.sin(angle);T=np.eye(4);T[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]];T[:3,3]=[.1,-.07,.03];rawB=t.transform_points(A,t.inverse_se3(T));image=np.zeros((256,256,3),np.uint8);mask=np.zeros((256,256),bool)
 xyz,xa,_=t._visible_raw(A,ca,image.shape[:2],mask,t.DEFAULT_CONFIG);correspondingB=t.transform_points(xyz,t.inverse_se3(T));pr=project_model_points(raw_to_model_points(correspondingB,'drone'),cb,(256,256));xb=pr['uv'];assert pr['in_image'].all()
 original=p.rgb_matches;p.rgb_matches=lambda *args:(xa.copy(),xb.copy())
 try:
  args=dict(rgb_A=image,raw_points_A=A,camera_A=ca,rgb_B=image,raw_points_B=rawB,camera_B=cb,excluded_mask_A=mask,excluded_mask_B=mask,seed=37)
  good=p.estimate_pair_registration(**args);assert good['accepted'] and np.max(abs(good['T_raw_B_to_A']-T))<1e-6
  wrong=p.estimate_pair_registration(**{**args,'raw_points_B':rawB+[5,0,0]});assert not wrong['accepted'] and np.array_equal(wrong['T_raw_B_to_A'],good['T_raw_B_to_A'])
  hold=t.spatial_split(xyz,37,t.DEFAULT_CONFIG);bad=xb.copy();bad[hold]+=[20,-17];p.rgb_matches=lambda *args:(xa.copy(),bad.copy());badheld=p.estimate_pair_registration(**args);assert not badheld['accepted'] and np.array_equal(badheld['T_raw_B_to_A'],good['T_raw_B_to_A'])
  p.rgb_matches=lambda *args:(xa.copy(),xb.copy());small=rawB[~t.spatial_split(rawB,37,t.DEFAULT_CONFIG)][:1];empty=p.estimate_pair_registration(**{**args,'raw_points_B':small});assert not empty['accepted'] and empty['status']=='PnP_insufficient_B_cloud_holdout' and empty['holdout_residuals']['n']==0 and np.array_equal(empty['T_raw_B_to_A'],good['T_raw_B_to_A'])
 finally:p.rgb_matches=original
 out={'state':'PASS_R12_INDEPENDENT_SYNTHETIC_ONE_DEPTH_PNP_REVIEW','module_sha256':sha(D/'sensor_pnp.py'),'fixture_sha256':sha(Path(__file__)),'rawB_to_rawA_direction_with_distinct_camera_extrinsics_and_Drone_shift':True,'known_transform_max_abs_error':float(np.max(abs(good['T_raw_B_to_A']-T))),'heldout_A_image_correspondences_only_reject_same_fit_T':True,'wrong_B_cloud_only_rejects_same_fit_T':True,'empty_B_cloud_holdout_returns_explicit_NA_same_fit_T_no_nan_quantile':True,'cheirality_positive_fraction_correct_case':good['RGB_validation']['positive_camera_depth_fraction'],'fit_holdout_pairs':[good['n_fit'],good['n_holdout']],'SIFT_frontend_stubbed_perfect_measurements_not_an_image_quality_test':True,'no_actual_data_GT_model_GPU_or_taskscore':True};(O/'PNP_INDEPENDENT_CPU_FIXTURE.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
