"""R12 sensor-only pair registration and independent-mask correspondence.
Pure array CPU API; no file/GT/model access. Primary mapping is raw B -> raw A.
Provided poses are diagnostic only and never initialize/select the estimator.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import sys,hashlib
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
D=Path(__file__).resolve().parent;sys.path.insert(0,str(D.parent/'round7'))
from camera_projection import raw_to_model_points,project_model_points,z_rotation

@dataclass(frozen=True)
class TemporalConfig:
    sift_features:int=2000
    sift_ratio:float=.75
    maximum_depth_binding_distance_px:float=3.
    minimum_depth_m:float=.1
    split_spatial_voxel_m:float=.5
    split_holdout_modulus:int=4
    split_holdout_remainder:int=0
    ransac_trials:int=512
    ransac_inlier_m:float=.25
    minimum_fit_pairs:int=6
    minimum_holdout_pairs:int=3
    minimum_fit_inlier_fraction:float=.5
    minimum_holdout_inlier_fraction:float=.6
    maximum_holdout_median_m:float=.15
    maximum_holdout_p90_m:float=.5
    minimum_triangle_cross_norm_m2:float=1e-6
    minimum_second_singular_value:float=1e-10
    mask_flow_features:int=64
    flow_quality_level:float=.01
    flow_min_distance_px:float=3.
    flow_window_px:int=21
    flow_pyramid_levels:int=3
    flow_iterations:int=30
    flow_epsilon:float=.01
    flow_cycle_max_px:float=1.5
    minimum_cycle_tracks:int=6
    mask_match_minimum_score:float=.5
    mask_match_minimum_margin:float=.15
    split_salt:str='R12_RGBD_spatial_fit_holdout_v1'
DEFAULT_CONFIG=TemporalConfig()

def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def validate_se3(T):
    a=np.asarray(T,np.float64)
    if a.shape!=(4,4) or not np.isfinite(a).all():raise ValueError('finite 4x4 transform required')
    if not np.allclose(a[3],[0,0,0,1],atol=1e-8,rtol=0):raise ValueError('invalid homogeneous row')
    if not np.allclose(a[:3,:3].T@a[:3,:3],np.eye(3),atol=1e-6,rtol=0) or abs(np.linalg.det(a[:3,:3])-1)>1e-6:raise ValueError('nonrigid rotation')
    return a.copy()
def inverse_se3(T):
    a=validate_se3(T);out=np.eye(4);out[:3,:3]=a[:3,:3].T;out[:3,3]=-out[:3,:3]@a[:3,3];return out

def raw_to_model_transform(camera):
    F=np.eye(4)
    if camera.platform=='drone':F[2,3]=1.8
    elif camera.platform=='waymo':
        if camera.waymo_view_suffix not in range(5):raise ValueError('unknown Waymo camera suffix')
        F[:3,:3]=z_rotation((0.,-45.,45.,-90.,90.)[camera.waymo_view_suffix])
    elif camera.platform!='quad':raise ValueError('unknown platform')
    return F

def raw_pair_to_model_transform(T_raw_B_to_A,camera_A,camera_B):
    return raw_to_model_transform(camera_A)@validate_se3(T_raw_B_to_A)@inverse_se3(raw_to_model_transform(camera_B))

def transform_points(points,T):
    p=np.asarray(points,np.float64);t=validate_se3(T)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all():raise ValueError('finite Nx3 points required')
    return p@t[:3,:3].T+t[:3,3]

def rigid_fit_B_to_A(points_B,points_A,config=DEFAULT_CONFIG):
    b=np.asarray(points_B,float);a=np.asarray(points_A,float)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=3 or len(a)<3 or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('at least3 finite 3D pairs required')
    cb=b.mean(0);ca=a.mean(0);u,s,vt=np.linalg.svd((b-cb).T@(a-ca))
    if s[1]<=config.minimum_second_singular_value:raise ValueError('collinear or degenerate correspondences')
    R=vt.T@u.T
    if np.linalg.det(R)<0:vt[-1]*=-1;R=vt.T@u.T
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=ca-R@cb;return validate_se3(T)

def spatial_split(points_B,seed=0,config=DEFAULT_CONFIG):
    p=np.asarray(points_B,np.float64);cells=np.floor(p/config.split_spatial_voxel_m).astype('<i8')
    u,inv=np.unique(cells,axis=0,return_inverse=True);prefix=(config.split_salt+'|'+str(int(seed))+'|').encode()
    h=np.array([int.from_bytes(hashlib.sha256(prefix+x.tobytes()).digest()[:8],'little')%config.split_holdout_modulus for x in u],np.int64)[inv]
    return h==config.split_holdout_remainder

def residual_stats(a,b,T,threshold):
    if len(a)==0:return {'n':0,'median_m':None,'p90_m':None,'mean_m':None,'inlier_fraction':None}
    e=np.linalg.norm(transform_points(b,T)-a,axis=1)
    return {'n':len(e),'median_m':float(np.median(e)),'p90_m':float(np.quantile(e,.9)),'mean_m':float(e.mean()),'inlier_fraction':float(np.mean(e<=threshold))}

def provided_pose_diagnostics(points_A,points_B,holdout,pose_A=None,pose_B=None,config=DEFAULT_CONFIG):
    """Diagnostic hypotheses only; does not return a chosen pose/estimator input."""
    candidates={'identity':np.eye(4)};reason='provided_poses_unavailable'
    if pose_A is not None and pose_B is not None:
        try:
            A=validate_se3(pose_A);B=validate_se3(pose_B)
            candidates.update(inv_pose_A_at_pose_B=inverse_se3(A)@B,inv_pose_B_at_pose_A=inverse_se3(B)@A);reason='directions_compared_not_provenance_verified'
        except ValueError as e:reason='provided_pose_invalid: '+str(e)
    return {'status':reason,'poses_used_for_primary_estimation':False,'hypotheses':{k:{'T_raw_B_to_A_assumed':v.tolist(),'fit':residual_stats(points_A[~holdout],points_B[~holdout],v,config.ransac_inlier_m),'holdout':residual_stats(points_A[holdout],points_B[holdout],v,config.ransac_inlier_m)} for k,v in candidates.items()}}

def estimate_from_correspondences(*,points_A,points_B,seed=0,provided_pose_A=None,provided_pose_B=None,config=DEFAULT_CONFIG):
    a=np.asarray(points_A,float);b=np.asarray(points_B,float)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=3 or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('finite matched 3D arrays required')
    # Canonical order avoids input permutation changing RANSAC/holdout membership.
    order=np.lexsort(tuple(np.c_[b,a][:,j] for j in range(5,-1,-1)));a=a[order];b=b[order]
    holdout=spatial_split(b,seed,config);fit=np.flatnonzero(~holdout)
    out={'status':'insufficient_correspondences','accepted':False,'T_raw_B_to_A':None,'n_pairs':len(a),'n_fit':len(fit),'n_holdout':int(holdout.sum()),'fit_pair_sha256':array_sha(np.c_[b[~holdout],a[~holdout]]),'holdout_pair_sha256':array_sha(np.c_[b[holdout],a[holdout]]),'holdout_disjoint_spatial_cells':True,'time_unit_unknown_no_velocity_estimate':True,'clock_status':'unknown_or_unverified_counter_not_used','observation_scope':'candidate overlapping raw views; not certified continuous video','kinematic_speed_estimated':False,'primary_uses_sensor_correspondences_only':True,'provided_pose_used_for_fit_or_selection':False,'config':asdict(config)}
    if len(fit)>=config.minimum_fit_pairs and holdout.sum()>=config.minimum_holdout_pairs:
        rng=np.random.default_rng(seed);best=None
        for trial in range(config.ransac_trials):
            ids=rng.choice(fit,3,replace=False)
            if np.linalg.norm(np.cross(b[ids[1]]-b[ids[0]],b[ids[2]]-b[ids[0]]))<config.minimum_triangle_cross_norm_m2:continue
            try:T=rigid_fit_B_to_A(b[ids],a[ids],config)
            except ValueError:continue
            err=np.linalg.norm(transform_points(b[fit],T)-a[fit],axis=1);inside=err<=config.ransac_inlier_m
            key=(int(inside.sum()),-float(np.median(err[inside])) if inside.any() else -float('inf'))
            if best is None or key>best[0]:best=(key,T,fit[inside])
        if best is not None and len(best[2])>=config.minimum_fit_pairs:
            # One fixed inlier refit; holdout never fits/selects a transform.
            try:T=rigid_fit_B_to_A(b[best[2]],a[best[2]],config)
            except ValueError as error:
                out.update(status='degenerate_fit_consensus',accepted=False,T_raw_B_to_A=None,degenerate_fit_reason=str(error))
            else:
                fs=residual_stats(a[~holdout],b[~holdout],T,config.ransac_inlier_m);hs=residual_stats(a[holdout],b[holdout],T,config.ransac_inlier_m)
                ok=fs['inlier_fraction']>=config.minimum_fit_inlier_fraction and hs['inlier_fraction']>=config.minimum_holdout_inlier_fraction and hs['median_m']<=config.maximum_holdout_median_m and hs['p90_m']<=config.maximum_holdout_p90_m
                out.update(status='sensor_SE3_supported_by_holdout' if ok else 'sensor_SE3_failed_holdout_quality',accepted=bool(ok),T_raw_B_to_A=T,fit_residuals=fs,holdout_residuals=hs,ransac_fit_inliers=len(best[2]),estimated_translation_norm_m=float(np.linalg.norm(T[:3,3])),estimated_rotation_angle_rad=float(np.arccos(np.clip((np.trace(T[:3,:3])-1)/2,-1,1))))
        else:out['status']='no_noncollinear_RANSAC_consensus'
    # This runs strictly after all estimator operations; cannot affect its RNG/result.
    out['provided_pose_diagnostic']=provided_pose_diagnostics(a,b,holdout,provided_pose_A,provided_pose_B,config)
    return out

def _rgb(rgb):
    x=np.asarray(rgb)
    if x.ndim!=3 or x.shape[2]!=3 or x.dtype!=np.uint8:raise ValueError('uint8 full RGB image required')
    return x

def _exclusion(mask,shape):
    if mask is None:return np.zeros(shape,bool),False
    m=np.asarray(mask)
    if m.dtype!=bool or m.shape!=shape:raise ValueError('original-resolution bool exclusion mask required')
    return m,True

def _visible_raw(points,camera,shape,excluded,config):
    p=np.asarray(points,float)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all():raise ValueError('finite raw point coordinates required')
    # Canonical unique XYZ prevents repeat scan rows inflating matches.
    p=np.unique(p,axis=0);H,W=shape
    model=raw_to_model_points(p,camera.platform,camera.waymo_view_suffix)
    pr=project_model_points(model,camera,(W,H),min_depth=config.minimum_depth_m);valid=pr['in_image'].copy()
    ids=np.flatnonzero(valid);pix=np.floor(pr['uv'][ids]).astype(np.int64)
    ids=ids[~excluded[pix[:,1],pix[:,0]]];pix=np.floor(pr['uv'][ids]).astype(np.int64)
    key=pix[:,1]*W+pix[:,0];v=p[ids];order=np.lexsort((v[:,2],v[:,1],v[:,0],pr['depth'][ids],key));ids=ids[order];key=key[order]
    ids=ids[np.r_[True,key[1:]!=key[:-1]]] if len(ids) else ids
    return p[ids],pr['uv'][ids],{'raw_unique_XYZ':len(p),'visible_zbuffer_unique':len(ids),'camera_diagnostics':pr['diagnostics']}

def estimate_pair_registration(*,rgb_A,raw_points_A,camera_A,rgb_B,raw_points_B,camera_B,excluded_mask_A=None,excluded_mask_B=None,provided_pose_A=None,provided_pose_B=None,seed=0,config=DEFAULT_CONFIG):
    import cv2
    A=_rgb(rgb_A);B=_rgb(rgb_B);ea,haveA=_exclusion(excluded_mask_A,A.shape[:2]);eb,haveB=_exclusion(excluded_mask_B,B.shape[:2])
    pa,ua,da=_visible_raw(raw_points_A,camera_A,A.shape[:2],ea,config);pb,ub,db=_visible_raw(raw_points_B,camera_B,B.shape[:2],eb,config)
    sift=cv2.SIFT_create(nfeatures=config.sift_features)
    ka,fa=sift.detectAndCompute(cv2.cvtColor(A,cv2.COLOR_RGB2GRAY),(~ea).astype(np.uint8)*255)
    kb,fb=sift.detectAndCompute(cv2.cvtColor(B,cv2.COLOR_RGB2GRAY),(~eb).astype(np.uint8)*255)
    pairs=[]
    if fa is not None and fb is not None and len(fa)>=2 and len(fb)>=2:
        bf=cv2.BFMatcher(cv2.NORM_L2)
        def ratio(x,y):
            out={}
            for group in bf.knnMatch(x,y,k=2):
                if len(group)==2 and group[0].distance<config.sift_ratio*group[1].distance:out[group[0].queryIdx]=group[0]
            return out
        ab=ratio(fa,fb);ba=ratio(fb,fa)
        for i,m in ab.items():
            if m.trainIdx in ba and ba[m.trainIdx].trainIdx==i:pairs.append((i,m.trainIdx,float(m.distance)))
    bound=[]
    if len(pa) and len(pb):
        ta,tb=cKDTree(ua),cKDTree(ub)
        for i,j,d in pairs:
            ad,ai=ta.query(ka[i].pt);bd,bi=tb.query(kb[j].pt)
            if ad<=config.maximum_depth_binding_distance_px and bd<=config.maximum_depth_binding_distance_px:bound.append((d,int(ai),int(bi),float(ad),float(bd)))
    # Unique physical depth observations on both sides, independent of SIFT multiplicity.
    usedA=set();usedB=set();kept=[]
    for x in sorted(bound):
        if x[1] not in usedA and x[2] not in usedB:kept.append(x);usedA.add(x[1]);usedB.add(x[2])
    aa=np.array([pa[x[1]] for x in kept],float).reshape(-1,3);bb=np.array([pb[x[2]] for x in kept],float).reshape(-1,3)
    out=estimate_from_correspondences(points_A=aa,points_B=bb,seed=seed,provided_pose_A=provided_pose_A,provided_pose_B=provided_pose_B,config=config)
    out['observation_diagnostics']={'SIFT_keypoints_A_B':[len(ka),len(kb)],'mutual_ratio_matches':len(pairs),'RGBD_matches_before_unique_return_binding':len(bound),'RGBD_unique_physical_pairs':len(kept),'maximum_binding_distance_px':config.maximum_depth_binding_distance_px,'foreground_exclusion_A_B_available':[haveA,haveB],'foreground_exclusion_is_fullimage_external_masks_not_q0':True,'background_only_qualification':bool(haveA and haveB and out['accepted']),'foreground_exclusion_does_not_certify_all_fit_points_background':True,'without_masks_is_preflight_only_not_certified_background_registration':not(haveA and haveB),'camera_A':da,'camera_B':db,'RGB_A_B_sha256':[array_sha(A),array_sha(B)]}
    return out

def _track_mask(grayA,grayB,mask,config):
    import cv2
    start=cv2.goodFeaturesToTrack(grayA,config.mask_flow_features,config.flow_quality_level,config.flow_min_distance_px,mask=mask.astype(np.uint8)*255)
    if start is None:return np.empty((0,2)),np.empty((0,2)),{'seed_points':0,'cycle_survivors':0}
    args=dict(winSize=(config.flow_window_px,config.flow_window_px),maxLevel=config.flow_pyramid_levels,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,config.flow_iterations,config.flow_epsilon))
    forward,st,_=cv2.calcOpticalFlowPyrLK(grayA,grayB,start,None,**args)
    if forward is None:return np.empty((0,2)),np.empty((0,2)),{'seed_points':len(start),'cycle_survivors':0}
    back,sb,_=cv2.calcOpticalFlowPyrLK(grayB,grayA,forward,None,**args)
    if back is None:return np.empty((0,2)),np.empty((0,2)),{'seed_points':len(start),'cycle_survivors':0}
    x=start[:,0];y=forward[:,0];err=np.linalg.norm(back[:,0]-x,axis=1);H,W=grayB.shape
    good=(st[:,0]>0)&(sb[:,0]>0)&np.isfinite(y).all(1)&np.isfinite(err)&(err<=config.flow_cycle_max_px)&(y[:,0]>=0)&(y[:,0]<W)&(y[:,1]>=0)&(y[:,1]<H)
    return x[good],y[good],{'seed_points':len(start),'cycle_survivors':int(good.sum()),'cycle_median_px':float(np.median(err[good])) if good.any() else None}

def match_instance_masks(*,rgb_A,rgb_B,masks_A,masks_B,class_ids_A,class_ids_B,config=DEFAULT_CONFIG):
    """Independent masks only. No q0/caption/GT. Two-frame optical-flow cycle,
    class-consistent one-to-one association; physical depth/motion is checked later.
    """
    import cv2
    A=_rgb(rgb_A);B=_rgb(rgb_B)
    if A.shape!=B.shape:raise ValueError('same original sensor image size required for LK')
    ma=np.asarray(masks_A);mb=np.asarray(masks_B);ca=np.asarray(class_ids_A);cb=np.asarray(class_ids_B)
    if ma.dtype!=bool or mb.dtype!=bool or ma.shape!=(len(ca),*A.shape[:2]) or mb.shape!=(len(cb),*B.shape[:2]):raise ValueError('full-resolution instance masks required')
    ga=cv2.cvtColor(A,cv2.COLOR_RGB2GRAY);gb=cv2.cvtColor(B,cv2.COLOR_RGB2GRAY)
    tracksA=[_track_mask(ga,gb,m,config) for m in ma];tracksB=[_track_mask(gb,ga,m,config) for m in mb]
    score=np.zeros((len(ma),len(mb)));valid=np.zeros_like(score,bool)
    for i,(_,endA,da) in enumerate(tracksA):
        for j,(_,endB,db) in enumerate(tracksB):
            if ca[i]!=cb[j] or min(len(endA),len(endB))<config.minimum_cycle_tracks:continue
            pixA=np.floor(endA).astype(int);pixB=np.floor(endB).astype(int)
            f=float(mb[j,pixA[:,1],pixA[:,0]].mean());r=float(ma[i,pixB[:,1],pixB[:,0]].mean());score[i,j]=np.sqrt(f*r);valid[i,j]=True
    allowed=np.zeros_like(valid);reasons=[]
    for i in range(len(ma)):
        for j in range(len(mb)):
            if not valid[i,j]:continue
            row=np.delete(score[i],j);col=np.delete(score[:,j],i);row_next=float(row.max()) if len(row) else 0.;col_next=float(col.max()) if len(col) else 0.
            allowed[i,j]=score[i,j]>=config.mask_match_minimum_score and score[i,j]-row_next>=config.mask_match_minimum_margin and score[i,j]-col_next>=config.mask_match_minimum_margin
    matches=[]
    if len(ma) and len(mb):
        ii,jj=linear_sum_assignment(-np.where(allowed,score,0.))
        for i,j in zip(ii,jj):
            if allowed[i,j]:matches.append({'instance_A':int(i),'instance_B':int(j),'score':float(score[i,j]),'class_id':int(ca[i]),'status':'flow_cycle_unique_candidate_not_verified_referring_identity','tracks_A_B':[tracksA[i][2],tracksB[j][2]]})
    matchedA={m['instance_A'] for m in matches};matchedB={m['instance_B'] for m in matches}
    return {'matches':matches,'unmatched_A':[i for i in range(len(ma)) if i not in matchedA],'unmatched_B':[j for j in range(len(mb)) if j not in matchedB],'score_matrix':score,'eligible_matrix':valid,'unique_margin_matrix':allowed,'tracks_A':[x[2] for x in tracksA],'tracks_B':[x[2] for x in tracksB],'not_literal_caption_identity_verified':True,'not_3D_or_object_motion_verified':True,'two_frame_forward_backward_cycle_only':True,'three_frame_track_composition_requires_additional_cycle_check':True,'flow_points_A_to_B':[{'start_xy':x[0],'end_xy':x[1]} for x in tracksA],'flow_points_B_to_A':[{'start_xy':x[0],'end_xy':x[1]} for x in tracksB],'clock_status':'unknown_or_unverified_counter_not_used','config':asdict(config)}

def bind_raw_depth_to_pixel_pairs(*,pixel_A,pixel_B,raw_points_A,camera_A,raw_points_B,camera_B,image_hw_A,image_hw_B,include_mask_A=None,include_mask_B=None,config=DEFAULT_CONFIG):
    """Optional object masks restrict depth to the independently proposed instance.
    Nearest3px is a sensor association hypothesis; masks/edges may still be wrong.
    """
    aa=np.asarray(pixel_A,float);bb=np.asarray(pixel_B,float)
    if aa.shape!=bb.shape or aa.ndim!=2 or aa.shape[1]!=2:raise ValueError('matching Nx2 pixels required')
    ia=np.ones(image_hw_A,bool) if include_mask_A is None else np.asarray(include_mask_A,bool)
    ib=np.ones(image_hw_B,bool) if include_mask_B is None else np.asarray(include_mask_B,bool)
    if ia.shape!=tuple(image_hw_A) or ib.shape!=tuple(image_hw_B):raise ValueError('original image-sized support masks required')
    pa,ua,_=_visible_raw(raw_points_A,camera_A,image_hw_A,~ia,config);pb,ub,_=_visible_raw(raw_points_B,camera_B,image_hw_B,~ib,config)
    matches=[]
    if len(pa) and len(pb):
        ta,tb=cKDTree(ua),cKDTree(ub)
        for i,(x,y) in enumerate(zip(aa,bb)):
            if not np.isfinite(x).all() or not np.isfinite(y).all():continue
            ad,ai=ta.query(x);bd,bi=tb.query(y)
            if max(ad,bd)<=config.maximum_depth_binding_distance_px:matches.append((float(ad+bd),int(ai),int(bi),i))
    seenA=set();seenB=set();kept=[]
    for pair in sorted(matches):
        if pair[1] not in seenA and pair[2] not in seenB:kept.append(pair);seenA.add(pair[1]);seenB.add(pair[2])
    return {'points_A':np.array([pa[m[1]] for m in kept],float).reshape(-1,3),'points_B':np.array([pb[m[2]] for m in kept],float).reshape(-1,3),'pixel_pair_indices':np.array([m[3] for m in kept],np.int64),'n_unique_physical_pairs':len(kept),'independent_instance_masks_used_A_B':[include_mask_A is not None,include_mask_B is not None],'not_complete_object_centers_or_extents':True}

def check_static_object_support(*,points_A,points_B,T_raw_B_to_A,config=DEFAULT_CONFIG):
    """Reject unsupported moving/nonrigid/wrong matches; never infer speed or fit
    separate object motion to make this static branch pass its own residual test.
    """
    a=np.asarray(points_A,float);b=np.asarray(points_B,float)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=3 or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('finite corresponding raw3D support required')
    stats=residual_stats(a,b,T_raw_B_to_A,config.ransac_inlier_m)
    ok=len(a)>=config.minimum_cycle_tracks and stats['inlier_fraction']>=config.minimum_holdout_inlier_fraction and stats['median_m']<=config.maximum_holdout_median_m and stats['p90_m']<=config.maximum_holdout_p90_m
    return {'status':'static_compatible_correspondence_not_verified_identity' if ok else 'NA_static_support_failed_motion_occlusion_or_correspondence_ambiguous','accepted':bool(ok),'residuals':stats,'no_object_motion_fitted_to_force_acceptance':True,'clock_status':'unverified_no_speed','literal_caption_identity_verified':False}
