"""Classical one-depth RGB-LiDAR odometry: 3D(A) -> image(B) PnP.
Separate heldout image rays and B-cloud validate the fixed fit. No pose/q0/GT.
"""
from dataclasses import dataclass,asdict
import numpy as np
from scipy.spatial import cKDTree
from temporal_association import _visible_raw,_rgb,_exclusion,spatial_split,DEFAULT_CONFIG as TC,raw_to_model_transform,transform_points,array_sha,validate_se3
from sensor_icp import rgb_matches
from camera_projection import model_to_camera_matrix
@dataclass(frozen=True)
class PnPConfig:
 binding_px:float=3.
 reprojection_px:float=3.
 iterations:int=1000
 confidence:float=.999
 minimum_fit:int=10
 minimum_holdout:int=6
 minimum_pixel_inlier_fraction:float=.6
 maximum_pixel_median:float=3.
 minimum_cloud_holdout:int=20
 cloud_inlier_m:float=.25
 maximum_cloud_median_m:float=.15
 minimum_cloud_inlier_fraction:float=.6
DEFAULT_CONFIG=PnPConfig()

def estimate_pair_registration(*,rgb_A,raw_points_A,camera_A,rgb_B,raw_points_B,camera_B,excluded_mask_A=None,excluded_mask_B=None,seed=0,config=DEFAULT_CONFIG,**kwargs):
 import cv2
 assert not kwargs
 A=_rgb(rgb_A);B=_rgb(rgb_B);ea,ha=_exclusion(excluded_mask_A,A.shape[:2]);eb,hb=_exclusion(excluded_mask_B,B.shape[:2]);pa,ua,da=_visible_raw(raw_points_A,camera_A,A.shape[:2],ea,TC);pb,ub,db=_visible_raw(raw_points_B,camera_B,B.shape[:2],eb,TC)
 out={'accepted':False,'status':'PnP_insufficient_correspondences','T_raw_B_to_A':None,'config':asdict(config),'provided_pose_used_for_fit_or_selection':False,'clock_status':'unknown_counter_no_velocity','primary_uses_sensor_correspondences_only':True,'observation_diagnostics':{'foreground_exclusion_A_B_available':[ha,hb],'background_only_qualification':False,'single_depth_only_avoids_two_nearest_depth_correspondences':True}}
 if not len(pa) or not len(pb):return out
 xa,xb=rgb_matches(A,B,ea,eb);tree=cKDTree(ua);ds,ids=tree.query(xa);used=set();keep=[]
 for k in np.argsort(ds,kind='stable'):
  if ds[k]<=config.binding_px and int(ids[k]) not in used:used.add(int(ids[k]));keep.append(int(k))
 ids=ids[keep];xp=xb[keep];xyz=pa[ids];held=spatial_split(xyz,seed,TC);fit=np.flatnonzero(~held);vh=np.flatnonzero(held);out.update(n_pairs=len(xyz),n_fit=len(fit),n_holdout=len(vh),fit_pair_sha256=array_sha(np.c_[xyz[fit],xp[fit]]),holdout_pair_sha256=array_sha(np.c_[xyz[vh],xp[vh]]),spatial_holdout_A_never_used_for_fit_or_refine=True)
 if len(fit)<config.minimum_fit or len(vh)<config.minimum_holdout:return out
 MA=model_to_camera_matrix(camera_A)@raw_to_model_transform(camera_A);MB=model_to_camera_matrix(camera_B)@raw_to_model_transform(camera_B);X=transform_points(xyz,MA);KB,_,dB,_=camera_B.validated();cv2.setRNGSeed(int(seed)%2147483647)
 try:
  ok,r,t,inl=cv2.solvePnPRansac(X[fit],xp[fit],KB,dB,iterationsCount=config.iterations,reprojectionError=config.reprojection_px,confidence=config.confidence,flags=cv2.SOLVEPNP_EPNP)
  if not ok or inl is None or len(inl)<config.minimum_fit:out['status']='PnP_no_RANSAC_consensus';return out
  selected=fit[inl[:,0]];ok,r,t=cv2.solvePnP(X[selected],xp[selected],KB,dB,r,t,True,flags=cv2.SOLVEPNP_ITERATIVE)
  if not ok or not np.isfinite(r).all() or not np.isfinite(t).all():out['status']='PnP_nonfinite_refit';return out
  C=np.eye(4);C[:3,:3]=cv2.Rodrigues(r)[0];C[:3,3]=t[:,0];T=validate_se3(np.linalg.inv(MA)@np.linalg.inv(C)@MB)
  positive=float(np.mean(transform_points(X[vh],C)[:,2]>.1))
  if positive<.9:out['status']='PnP_heldout_cheirality_failed';return out
  pr=cv2.projectPoints(X[vh],r,t,KB,dB)[0][:,0];err=np.linalg.norm(pr-xp[vh],axis=1)
 except (cv2.error,ValueError,np.linalg.LinAlgError):out['status']='PnP_invalid_or_degenerate';return out
 pix={'n':len(err),'positive_camera_depth_fraction':positive,'median_px':float(np.median(err)),'p90_px':float(np.quantile(err,.9)),'inlier_fraction':float(np.mean(err<=config.reprojection_px)),'RANSAC_fit_inliers':len(inl),'p90_reported_not_gate_due_sparse_depth_measurement_outliers':True}
 bh=pb[spatial_split(pb,seed,TC)]
 if len(bh)<config.minimum_cloud_holdout:
  out.update(status='PnP_insufficient_B_cloud_holdout',T_raw_B_to_A=T,RGB_validation=pix,holdout_residuals={'n':len(bh),'not_evaluated_insufficient_depth_returns':True});return out
 d,_=cKDTree(pa).query(transform_points(bh,T));cloud={'n':len(d),'median_m':float(np.median(d)),'p90_m':float(np.quantile(d,.9)),'inlier_fraction':float(np.mean(d<=config.cloud_inlier_m)),'B_cloud_never_used_to_fit_or_initialize_pose':True,'A_full_background_fixed_reference_not_independent_A_holdout':True}
 good=ha and hb and pix['median_px']<=config.maximum_pixel_median and pix['inlier_fraction']>=config.minimum_pixel_inlier_fraction and len(d)>=config.minimum_cloud_holdout and cloud['median_m']<=config.maximum_cloud_median_m and cloud['inlier_fraction']>=config.minimum_cloud_inlier_fraction
 out.update(accepted=bool(good),status='sensor_PnP_heldout_rays_and_cloud_supported' if good else 'sensor_PnP_failed_heldout_rays_or_cloud',T_raw_B_to_A=T,RGB_validation=pix,holdout_residuals=cloud,estimated_translation_norm_m=float(np.linalg.norm(T[:3,3])));out['observation_diagnostics']['background_only_qualification']=bool(good);return out
