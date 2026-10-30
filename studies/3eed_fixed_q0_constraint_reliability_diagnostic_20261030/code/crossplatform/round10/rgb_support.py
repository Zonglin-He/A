"""R10 point-seeded GrabCut RGB support, pure supplied-array/camera functions.
No files, annotations, captions, classes, GT, scores, teachers or model calls.
Foreground is a fallible q0-conditioned observation mask, not object identity.
Official algorithm/API: docs.opencv.org/4.11.0/d3/d47/group__imgproc__segmentation.html
and Rother/Kolmogorov/Blake, SIGGRAPH2004 GrabCut (Microsoft Research).
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib,json,sys
import numpy as np

@dataclass(frozen=True)
class RGBSupportConfig:
    context_fraction: float=.25
    context_min_pixels: int=4
    outer_background_border_pixels: int=2
    min_roi_side_pixels: int=8
    min_unique_foreground_seed_pixels: int=5
    min_background_seed_pixels: int=5
    grabcut_iterations: int=5
    min_depth: float=.1
    connected_component_connectivity: int=8
DEFAULT_CONFIG=RGBSupportConfig()

def config_sha256(config=DEFAULT_CONFIG):
    return hashlib.sha256(json.dumps(asdict(config),sort_keys=True,separators=(',',':')).encode()).hexdigest()

def frame_split_seed(private_case_id):
    return int.from_bytes(hashlib.sha256(('R10_visible_point_split_v1|'+str(private_case_id)).encode()).digest()[:4],'little')&0x7fffffff

def split_point_observations(points_xyz,seed):
    """Canonicalfloat32 uniqueXYZ hash parity; duplicates stay on the same side.
    Only a scaffold for future within-frame checks, never extra scenes/views.
    No forced balancing: short/empty partitions remain actual and explicit.
    """
    p=np.array(points_xyz,dtype='<f4',copy=True)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all():raise ValueError('FiniteNx3 points required')
    if not isinstance(seed,(int,np.integer)) or not 0<=int(seed)<2**63:raise ValueError('Unsigned63bit splitseed required')
    p[p==0]=np.float32(0);unique,inverse=np.unique(p,axis=0,return_inverse=True);prefix=int(seed).to_bytes(8,'little')
    parity=np.array([hashlib.sha256(prefix+row.astype('<f4',copy=False).tobytes()).digest()[0]&1 for row in unique],np.uint8)
    fit=parity[inverse]==0;ev=~fit
    return {'fit_mask':fit,'eval_mask':ev,'diagnostics':{'seed':int(seed),'point_count':len(p),'unique_XYZ_count':len(unique),'fit_points':int(fit.sum()),'eval_points':int(ev.sum()),'fit_unique_XYZ':int((parity==0).sum()),'eval_unique_XYZ':int((parity==1).sum()),'duplicate_observations_never_cross_roles':True,'fit_mask_sha256':hashlib.sha256(fit.tobytes()).hexdigest(),'eval_mask_sha256':hashlib.sha256(ev.tobytes()).hexdigest(),'not_independent_views_scenes_or_backbone_inputs':True}}

def _pixels(uv,depth,image_size_wh,min_depth):
    u=np.asarray(uv,float);d=np.asarray(depth,float);w,h=map(int,image_size_wh)
    if u.ndim!=2 or u.shape[1]!=2 or d.shape!=(len(u),):raise ValueError('uvNx2/depthN required')
    valid=np.isfinite(u).all(1)&np.isfinite(d)&(d>min_depth)&(u[:,0]>=0)&(u[:,0]<w)&(u[:,1]>=0)&(u[:,1]<h)
    px=np.zeros((len(u),2),np.int64);px[valid]=np.floor(u[valid]).astype(np.int64)
    return px,valid

def visible_point_indices(points_xyz,uv,depth,image_size_wh,min_depth=.1):
    """ContinuousFOV thenfloor nearestdepth zbuffer; XYZ tie-break; no eval data.
    Caller must construct this separately from fitpoints if a heldoutclaim is made.
    """
    p=np.asarray(points_xyz,float);u=np.asarray(uv,float);d=np.asarray(depth,float)
    if p.ndim!=2 or p.shape!=(len(u),3) or not np.isfinite(p).all():raise ValueError('Finite matchingXYZ required')
    px,valid=_pixels(u,d,image_size_wh,min_depth);ids=np.flatnonzero(valid)
    if not len(ids):return ids
    key=px[ids,1]*int(image_size_wh[0])+px[ids,0];x=p[ids];order=np.lexsort((x[:,2],x[:,1],x[:,0],d[ids],key));ids=ids[order];key=key[order]
    return ids[np.r_[True,key[1:]!=key[:-1]]]

def _box_corners(box6):
    b=np.asarray(box6,float)
    if b.shape!=(6,) or not np.isfinite(b).all() or (b[3:]<=0).any():raise ValueError('Finite positive-size sourcebox6 required')
    s=np.array([[-1 if i&4==0 else 1,-1 if i&2==0 else 1,-1 if i&1==0 else 1] for i in range(8)])
    return b[:3]+s*b[3:]/2

def segment_q0_support(rgb_uint8,*,box_corners_uv,box_corner_depth,fit_uv,fit_depth,fit_source_crop_mask,seed,config=DEFAULT_CONFIG):
    """Point-backed mask initialization and seeded-component GrabCut, no fileIO.
    fit arrays must contain ONLY allowed observation points. No eval array exists
    in this interface. Whole-observation callers explicitly pass their whole P.
    """
    import cv2
    rgb=np.asarray(rgb_uint8)
    if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:raise ValueError('uint8 HxWx3 RGB required')
    h,w=rgb.shape[:2];c=np.asarray(box_corners_uv,float);cd=np.asarray(box_corner_depth,float);u=np.asarray(fit_uv,float);d=np.asarray(fit_depth,float);crop=np.asarray(fit_source_crop_mask,bool)
    if c.shape!=(8,2) or cd.shape!=(8,) or u.ndim!=2 or u.shape[1]!=2 or d.shape!=(len(u),) or crop.shape!=(len(u),):raise ValueError('Invalid supplied projection shapes')
    if config.grabcut_iterations<1 or config.connected_component_connectivity not in (4,8):raise ValueError('InvalidGrabCut configuration')
    cfg=asdict(config);diag={'opencv_version':cv2.__version__,'config':cfg,'config_sha256':config_sha256(config),'cv2_rng_seed':int(seed)&0x7fffffff,'fit_observation_count':len(u),'no_eval_points_or_GT_or_color_rules_used':True,'no_teacher_or_language_encoder':True,'q0_fixed':True,'mask_not_calibrated_object_or_identity_probability':True,'fake_all_foreground_fallback':False}
    empty=np.zeros((h,w),bool)
    def fail(reason,**extra):return {'mask2d':empty.copy(),'point_mask_fit':np.zeros(len(u),bool),'valid':False,'diagnostics':{**diag,**extra,'state':'segmentation_rejected','reason':reason,'mask_sha256':hashlib.sha256(empty.tobytes()).hexdigest()}}
    if h<config.min_roi_side_pixels or w<config.min_roi_side_pixels:return fail('image_too_small')
    if not np.isfinite(c).all() or not np.isfinite(cd).all() or not(cd>config.min_depth).all():return fail('sourcebox_crosses_or_behind_near_plane')
    raw_lo=c.min(0);raw_hi=c.max(0);span=raw_hi-raw_lo
    if (span<=0).any():return fail('degenerate_projected_sourcebox')
    extra=np.maximum(span*config.context_fraction,config.context_min_pixels)
    raw_context=np.r_[raw_lo-extra,raw_hi+extra]
    x0,y0=np.maximum(0,np.floor(raw_context[:2])).astype(int);x1,y1=np.minimum([w,h],np.ceil(raw_context[2:])).astype(int)
    if x1-x0<config.min_roi_side_pixels or y1-y0<config.min_roi_side_pixels:return fail('source_roi_outside_or_too_small')
    rx0,ry0=np.maximum([x0,y0],np.floor(raw_lo)).astype(int);rx1,ry1=np.minimum([x1,y1],np.ceil(raw_hi)).astype(int)
    if rx1<=rx0 or ry1<=ry0:return fail('source_tight_roi_outside')
    diag.update(roi_xyxy=[int(x0),int(y0),int(x1),int(y1)],source_projected_tight_xyxy=[int(rx0),int(ry0),int(rx1),int(ry1)],context_clipped_by_image=bool((raw_context[:2]<0).any() or (raw_context[2:]>[w,h]).any()),roi_area_pixels=int((x1-x0)*(y1-y0)))
    px,valid=_pixels(u,d,(w,h),config.min_depth);seedids=np.flatnonzero(valid&crop&(px[:,0]>=x0)&(px[:,0]<x1)&(px[:,1]>=y0)&(px[:,1]<y1));seedpixels=np.unique(px[seedids],axis=0)
    diag.update(source_inside_visible_seed_observations=len(seedids),unique_foreground_seed_pixels=len(seedpixels))
    if len(seedpixels)<config.min_unique_foreground_seed_pixels:return fail('insufficient_unique_point_backed_foreground_pixels')
    rh,rw=y1-y0,x1-x0;labels=np.full((rh,rw),cv2.GC_PR_BGD,np.uint8);labels[ry0-y0:ry1-y0,rx0-x0:rx1-x0]=cv2.GC_PR_FGD
    border=config.outer_background_border_pixels;labels[:border,:]=cv2.GC_BGD;labels[-border:,:]=cv2.GC_BGD;labels[:,:border]=cv2.GC_BGD;labels[:,-border:]=cv2.GC_BGD
    sx,sy=seedpixels[:,0]-x0,seedpixels[:,1]-y0;labels[sy,sx]=cv2.GC_FGD;bg_count=int((labels==cv2.GC_BGD).sum());diag['hard_background_seed_pixels']=bg_count
    if bg_count<config.min_background_seed_pixels:return fail('insufficient_hard_background_pixels')
    patch=np.ascontiguousarray(rgb[y0:y1,x0:x1,::-1]);bg=np.zeros((1,65),np.float64);fg=np.zeros((1,65),np.float64)
    try:
        cv2.setRNGSeed(int(seed)&0x7fffffff);cv2.grabCut(patch,labels,None,bg,fg,config.grabcut_iterations,cv2.GC_INIT_WITH_MASK)
    except cv2.error as error:return fail('opencv_grabcut_error',opencv_error_code=int(error.code))
    raw_fg=(labels==cv2.GC_FGD)|(labels==cv2.GC_PR_FGD)
    n,components,stats,_=cv2.connectedComponentsWithStats(raw_fg.astype(np.uint8),connectivity=config.connected_component_connectivity)
    seed_component=components[sy,sx];choices=[]
    for label in range(1,n):
        hits=int((seed_component==label).sum())
        if hits:choices.append((hits,int(stats[label,cv2.CC_STAT_AREA]),-label,label))
    if not choices:return fail('no_foreground_component_with_point_seed')
    selected=max(choices)[-1];kept=components==selected;area=int(kept.sum());retained=int((seed_component==selected).sum())
    if area==0 or area==rh*rw:return fail('degenerate_empty_or_full_roi_foreground')
    mask=np.zeros((h,w),bool);mask[y0:y1,x0:x1]=kept;pointmask=np.zeros(len(u),bool);ids=np.flatnonzero(valid);pointmask[ids]=mask[px[ids,1],px[ids,0]]
    diag.update(state='point_seeded_grabcut_component',reason='segmentation_available_not_correctness_proof',foreground_components=n-1,seeded_components=len(choices),retained_seed_pixels=retained,discarded_seed_pixels=len(seedpixels)-retained,retained_seed_fraction=retained/len(seedpixels),foreground_area_pixels=area,foreground_fraction_of_roi=area/(rh*rw),foreground_fit_observations=int(pointmask.sum()),component_tie_rule='maxunique_seedhits thenarea then smallestcomponentlabel',mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest())
    return {'mask2d':mask,'point_mask_fit':pointmask,'valid':True,'diagnostics':diag}

def make_rgb_support(*,rgb,points_fit,source_box6,camera_spec,config=DEFAULT_CONFIG):
    """Rootrunner API, projection inside; CameraSpec supplied, no metadata reads.
    No q0 reselection and no extra CLIP. Future holdouteval must not be passed in P.
    """
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'round7'))
    from camera_projection import project_model_points
    p=np.asarray(points_fit,float);image=np.asarray(rgb)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all():raise ValueError('Finite points_fitNx3 required')
    if image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8:raise ValueError('uint8 RGB required')
    h,w=image.shape[:2]
    try:corners=_box_corners(source_box6)
    except ValueError:
        return {'mask2d':np.zeros((h,w),bool),'valid':False,'diagnostics':{'state':'segmentation_rejected','reason':'invalid_source_dimensions_or_coordinates','fake_all_foreground_fallback':False,'config_sha256':config_sha256(config)}}
    projected=project_model_points(p,camera_spec,(w,h),min_depth=config.min_depth);boxprojected=project_model_points(corners,camera_spec,(w,h),min_depth=config.min_depth)
    visible=visible_point_indices(p,projected['uv'],projected['depth'],(w,h),config.min_depth);box=np.asarray(source_box6,float);inside=(np.abs(p[visible]-box[:3])<=box[3:]/2).all(1)
    # Caller can choose deterministic seed via config-independent frame seed;
    # existing API has no seed argument, so CameraSpec-independent supplied arrays
    # determine a repeatable seed. It does not depend on GT/method score.
    seed=int.from_bytes(hashlib.sha256(np.ascontiguousarray(box,dtype='<f8').tobytes()+image.shape[0].to_bytes(8,'little')+image.shape[1].to_bytes(8,'little')).digest()[:4],'little')&0x7fffffff
    result=segment_q0_support(image,box_corners_uv=boxprojected['uv'],box_corner_depth=boxprojected['depth'],fit_uv=projected['uv'][visible],fit_depth=projected['depth'][visible],fit_source_crop_mask=inside,seed=seed,config=config)
    return {'mask2d':result['mask2d'],'valid':result['valid'],'diagnostics':{**result['diagnostics'],'input_points_count':len(p),'visible_unique_pixel_points':len(visible),'input_point_role':'only supplied points_fit consumed; caller declares wholeobs vsfit-only','projection_diagnostics':projected['diagnostics'],'visible_points_hash':hashlib.sha256(np.ascontiguousarray(p[visible],dtype='<f8').tobytes()).hexdigest(),'source_q0_only':True}}
