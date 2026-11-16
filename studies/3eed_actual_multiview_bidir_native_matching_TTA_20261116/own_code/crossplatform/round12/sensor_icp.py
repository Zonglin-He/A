"""Classical sensor-only trimmed ICP, checked by withheld returns and independent RGB matches.
No poses/q0/captions/GT. Complement to sparse SIFT-nearest-depth correspondence.
"""
from dataclasses import dataclass,asdict
import numpy as np
from scipy.spatial import cKDTree
from temporal_association import _visible_raw,_rgb,_exclusion,spatial_split,rigid_fit_B_to_A,transform_points,DEFAULT_CONFIG as TC,array_sha,raw_to_model_transform
from camera_projection import model_to_camera_matrix
@dataclass(frozen=True)
class ICPConfig:
 voxel_m:float=.10
 iterations:int=30
 max_correspondence_m:float=1.
 trim_fraction:float=.70
 minimum_fit_points:int=30
 minimum_holdout_points:int=20
 holdout_inlier_m:float=.25
 minimum_holdout_inlier_fraction:float=.60
 max_holdout_median_m:float=.15
 max_holdout_p90_m:float=.5
 minimum_RGB_matches:int=12
 max_RGB_median_epipolar_px:float=2.
 RGB_inlier_px:float=3.
 minimum_RGB_inlier_fraction:float=.60
DEFAULT_CONFIG=ICPConfig()

def voxel(p,width):
 p=np.unique(np.asarray(p,float),axis=0);_,ix=np.unique(np.floor(p/width).astype(np.int64),axis=0,return_index=True);return p[np.sort(ix)]
def rgb_matches(A,B,ea,eb):
 import cv2
 sf=cv2.SIFT_create(nfeatures=TC.sift_features);ka,fa=sf.detectAndCompute(cv2.cvtColor(A,cv2.COLOR_RGB2GRAY),(~ea).astype(np.uint8)*255);kb,fb=sf.detectAndCompute(cv2.cvtColor(B,cv2.COLOR_RGB2GRAY),(~eb).astype(np.uint8)*255)
 if fa is None or fb is None or min(len(fa),len(fb))<2:return np.empty((0,2)),np.empty((0,2))
 bf=cv2.BFMatcher(cv2.NORM_L2)
 def ratio(a,b):
  out={}
  for ms in bf.knnMatch(a,b,k=2):
   if len(ms)==2 and ms[0].distance<TC.sift_ratio*ms[1].distance:out[ms[0].queryIdx]=ms[0].trainIdx
  return out
 ab=ratio(fa,fb);ba=ratio(fb,fa);pairs=sorted((i,j) for i,j in ab.items() if ba.get(j)==i)
 return np.array([ka[i].pt for i,j in pairs],float).reshape(-1,2),np.array([kb[j].pt for i,j in pairs],float).reshape(-1,2)
def epipolar(Auv,Buv,Traw,ca,cb,cfg):
 if not len(Auv):return {'n':0,'median_px':None,'inlier_fraction':None,'mode':'no_matches'}
 KA=ca.validated()[0];KB=cb.validated()[0];MA=model_to_camera_matrix(ca)@raw_to_model_transform(ca);MB=model_to_camera_matrix(cb)@raw_to_model_transform(cb);C=MA@Traw@np.linalg.inv(MB);R=C[:3,:3];t=C[:3,3]
 a=np.c_[Auv,np.ones(len(Auv))];b=np.c_[Buv,np.ones(len(Buv))]
 if np.linalg.norm(t)<1e-5:
  pred=(KA@R@np.linalg.inv(KB)@b.T).T;err=np.linalg.norm(pred[:,:2]/pred[:,2:3]-Auv,axis=1);mode='rotation_only_nearzero_translation'
 else:
  tx=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]]);F=np.linalg.inv(KA).T@tx@R@np.linalg.inv(KB);ab=(F@b.T).T;ba=(F.T@a.T).T;err=np.abs((a*ab).sum(1))/np.sqrt((ab[:,:2]**2).sum(1)+(ba[:,:2]**2).sum(1)+1e-30);mode='symmetric_Sampson_firstorder'
 return {'n':len(err),'median_px':float(np.median(err)),'p90_px':float(np.quantile(err,.9)),'inlier_fraction':float(np.mean(err<=cfg.RGB_inlier_px)),'mode':mode,'RGB_correspondences_not_used_for_ICP_fit_or_initialization':True}

def estimate_pair_registration(*,rgb_A,raw_points_A,camera_A,rgb_B,raw_points_B,camera_B,excluded_mask_A=None,excluded_mask_B=None,seed=0,config=DEFAULT_CONFIG,**kwargs):
 # No provided pose accepted, including as an ignored fallback.
 assert not kwargs
 A=_rgb(rgb_A);B=_rgb(rgb_B);ea,ha=_exclusion(excluded_mask_A,A.shape[:2]);eb,hb=_exclusion(excluded_mask_B,B.shape[:2]);pa,_,da=_visible_raw(raw_points_A,camera_A,A.shape[:2],ea,TC);pb,_,db=_visible_raw(raw_points_B,camera_B,B.shape[:2],eb,TC);pa=voxel(pa,config.voxel_m);pb=voxel(pb,config.voxel_m);held=spatial_split(pb,seed,TC);fit=pb[~held];ho=pb[held];T=np.eye(4)
 out={'accepted':False,'status':'ICP_insufficient_observations','T_raw_B_to_A':None,'config':asdict(config),'n_fit':len(fit),'n_holdout':len(ho),'fit_point_sha256':array_sha(fit),'holdout_point_sha256':array_sha(ho),'holdout_B_not_used_to_fit_initialize_or_select_transform':True,'provided_pose_used_for_fit_or_selection':False,'clock_status':'unknown_counter_no_velocity','primary_uses_sensor_correspondences_only':True,'observation_diagnostics':{'foreground_exclusion_A_B_available':[ha,hb],'background_only_qualification':False,'raw_visible_voxel_points_A_B':[len(pa),len(pb)],'background_is_fallible_instance_exclusion_not_semanticGT':True}}
 if len(pa)<config.minimum_fit_points or len(fit)<config.minimum_fit_points or len(ho)<config.minimum_holdout_points:return out
 tree=cKDTree(pa);steps=[]
 for it in range(config.iterations):
  now=transform_points(fit,T);dist,ix=tree.query(now);eligible=np.flatnonzero(dist<=config.max_correspondence_m)
  if len(eligible)<config.minimum_fit_points:out['status']='ICP_insufficient_bounded_correspondences';return out
  n=max(config.minimum_fit_points,int(np.floor(len(eligible)*config.trim_fraction)));keep=eligible[np.argsort(dist[eligible],kind='stable')[:n]]
  try:new=rigid_fit_B_to_A(fit[keep],pa[ix[keep]])
  except ValueError:out['status']='ICP_degenerate_consensus';return out
  delta=float(np.linalg.norm(new-T));T=new;steps.append({'iteration':it,'kept_n':len(keep),'trimmed_median_m':float(np.median(dist[keep])),'matrix_delta':delta})
  if delta<1e-7:break
 # Holdout and RGB are used only after the fixed fit; they never choose/restart T.
 d,_=tree.query(transform_points(ho,T));st={'n':len(d),'median_m':float(np.median(d)),'p90_m':float(np.quantile(d,.9)),'inlier_fraction':float(np.mean(d<=config.holdout_inlier_m)),'reference_A_full_background_is_fixed_not_independent_A_holdout':True};ua,ub=rgb_matches(A,B,ea,eb);im=epipolar(ua,ub,T,camera_A,camera_B,config)
 ok=ha and hb and st['median_m']<=config.max_holdout_median_m and st['p90_m']<=config.max_holdout_p90_m and st['inlier_fraction']>=config.minimum_holdout_inlier_fraction and im['n']>=config.minimum_RGB_matches and im['median_px']<=config.max_RGB_median_epipolar_px and im['inlier_fraction']>=config.minimum_RGB_inlier_fraction
 out.update(accepted=bool(ok),status='sensor_ICP_holdout_and_RGB_supported' if ok else 'sensor_ICP_failed_holdout_or_RGB',T_raw_B_to_A=T,fit_steps=steps,holdout_residuals=st,RGB_validation=im,estimated_translation_norm_m=float(np.linalg.norm(T[:3,3])));out['observation_diagnostics']['background_only_qualification']=bool(ok);return out
