"""R15 whole-image independent object observations. Pure arrays; no source ROI.

Predicted segmentation ownership is evidence, never GT instance identity. File,
model, caption, q0 and task-annotation inputs are absent by construction.
"""
from dataclasses import asdict,dataclass
from pathlib import Path
import hashlib,json,re,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'round7'))
from camera_projection import project_model_points,raw_to_model_points

@dataclass(frozen=True)
class SceneConfig:
    minimum_depth_m:float=.1
    minimum_depth_attribute_points:int=8
DEFAULT_CONFIG=SceneConfig()

def array_sha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def config_sha256(cfg=DEFAULT_CONFIG):return hashlib.sha256(json.dumps(asdict(cfg),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def model_points(raw,camera):
    p=np.array(raw,np.float32,copy=True)
    if camera.platform=='drone':p[:,2]+=np.float32(1.8)
    elif camera.platform!='quad':raise ValueError('R15 target scope is Drone/Quad')
    return p

def _depth_stats(depth,minimum):
    d=np.asarray(depth,np.float64);defined=bool(len(d)>=minimum)
    return {'defined':defined,'n':len(d),'median':float(np.median(d)) if defined else None,
            'q10':float(np.quantile(d,.1)) if defined else None,'q90':float(np.quantile(d,.9)) if defined else None,
            'basis':'camera z depth of global-zbuffer exclusive mask points, not model XYZ'}

def build_scene_observation(*,raw_points,camera_spec,image_size_wh,instances,image_sha256,config=DEFAULT_CONFIG):
    """Full cloud visibility before masks; positive/foreign ownership disjoint.

    Required independent frontend fields are exact; injected task/GT fields fail.
    Original raw row IDs are representatives, not canonical-array row positions.
    """
    cfg=config;camera_spec.validated()
    if camera_spec.platform not in ('drone','quad'):raise ValueError('Only target Drone/Quad supported')
    raw=np.asarray(raw_points)
    if raw.ndim!=2 or raw.shape[1]!=3 or not np.isfinite(raw).all():raise ValueError('Finite raw Nx3 required')
    if len(image_size_wh)!=2:raise ValueError('Explicit image width/height required')
    W,H=map(int,image_size_wh)
    if W<=0 or H<=0:raise ValueError('Positive image dimensions required')
    if not isinstance(image_sha256,str) or not re.fullmatch('[0-9a-f]{64}',image_sha256):raise ValueError('Bound actual image SHA256 required')
    required={'boxes_xyxy','class_ids','scores','masks_bool','instance_indices'}
    if not isinstance(instances,dict) or set(instances)!=required:raise ValueError('Strict independent masks; task/GT/ROI fields forbidden')
    labels=np.asarray(instances['class_ids']);n=len(labels);masks=np.asarray(instances['masks_bool']);native=np.asarray(instances['instance_indices'])
    if labels.shape!=(n,) or not np.issubdtype(labels.dtype,np.integer) or (labels<=0).any():raise ValueError('Predicted integer positive COCO classes required')
    if native.shape!=(n,) or not np.issubdtype(native.dtype,np.integer) or ((native<0)|(native>=100000)).any() or len(np.unique(native))!=n:raise ValueError('Distinct nonnegative native detection indices required')
    if masks.dtype!=bool or masks.shape!=(n,H,W):raise ValueError('Original-resolution independent bool masks required')
    boxes=np.asarray(instances['boxes_xyxy'],np.float64);scores=np.asarray(instances['scores'],np.float64)
    if boxes.shape!=(n,4) or scores.shape!=(n,) or not np.isfinite(boxes).all() or not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any():raise ValueError('Finite detector boxes/scores required')
    unique,first=np.unique(raw.astype(np.float64),axis=0,return_index=True)
    # Raw projection bridge uses unrounded physical coordinates. Model support
    # keeps source float32 Drone in-place arithmetic separately and explicitly.
    pr=project_model_points(raw_to_model_points(unique,camera_spec.platform),camera_spec,(W,H),min_depth=cfg.minimum_depth_m)
    take=np.flatnonzero(pr['in_image']);pix=np.floor(pr['uv'][take]).astype(np.int64);keys=pix[:,1]*W+pix[:,0]
    a=unique[take];order=np.lexsort((a[:,2],a[:,1],a[:,0],pr['depth'][take],keys));take=take[order];keys=keys[order]
    take=take[np.r_[True,keys[1:]!=keys[:-1]]] if len(take) else take
    premerge=len(take);mp=model_points(unique[take],camera_spec)
    # Rare raw-double -> model-float32 collisions never multiply support. Choose
    # nearest physical camera depth, then raw XYZ/id deterministically.
    if len(take):
        _,inv=np.unique(mp,axis=0,return_inverse=True)
        rawids=first[take];p=unique[take]
        ord2=np.lexsort((rawids,p[:,2],p[:,1],p[:,0],pr['depth'][take],inv))
        groups=inv[ord2];keep=ord2[np.r_[True,groups[1:]!=groups[:-1]]];take=take[keep];mp=mp[keep]
    uv=pr['uv'][take];depth=pr['depth'][take];pixels=np.floor(uv).astype(np.int64);rawids=first[take].astype(np.int64)
    memberships=masks[:,pixels[:,1],pixels[:,0]] if len(take) else np.zeros((n,0),bool)
    memberships_n=memberships.sum(0);owners=np.full(len(take),-1,np.int64)
    owners[memberships_n>1]=-2
    if n:
        exclusive=memberships_n==1;owners[exclusive]=np.argmax(memberships[:,exclusive],axis=0)
    objects=[]
    for j in range(n):
        yx=np.argwhere(masks[j]);ids=np.flatnonzero(memberships[j]);ex=np.flatnonzero(owners==j)
        defined=bool(len(yx));lo=yx.min(0)[::-1] if defined else np.zeros(2);hi=yx.max(0)[::-1]+1 if defined else np.zeros(2)
        centroid=(yx[:,::-1].mean(0)+.5)/np.array([W,H]) if defined else None
        extent=(hi-lo)/np.array([W,H]) if defined else None
        border=bool(defined and (masks[j,0].any() or masks[j,-1].any() or masks[j,:,0].any() or masks[j,:,-1].any()))
        objects.append({'mask_id':f'M{int(native[j]):05d}','native_instance_id':int(native[j]),'array_index':j,
            'predicted_class_id':int(labels[j]),'detector_score':float(scores[j]),'detector_box_xyxy':boxes[j].copy(),
            'mask_bool':masks[j].copy(),'mask_sha256':array_sha(masks[j]),'mask_area_pixels':len(yx),
            'mask_geometry_defined':defined,'mask_centroid_defined':defined,'mask_extent_defined':defined,'mask_bbox_defined':defined,
            'mask_centroid_norm2':centroid,'mask_extent_norm2':extent,
            'mask_bbox_norm4':np.r_[lo,hi]/np.array([W,H,W,H]) if defined else None,
            'touches_image_border':border,'visible_indices':ids,'exclusive_visible_indices':ex,
            'n_visible':len(ids),'n_exclusive':len(ex),'n_ambiguous_overlap':int(np.sum(memberships[j] & (memberships_n>1))),
            'points_model':mp[ex].copy(),'physical_raw_ids':rawids[ex].copy(),
            'all_mask_visible_points_model':mp[ids].copy(),'all_mask_visible_raw_ids':rawids[ids].copy(),
            'depth_camera':_depth_stats(depth[ex],cfg.minimum_depth_attribute_points),
            'depth_camera_all_mask':_depth_stats(depth[ids],cfg.minimum_depth_attribute_points),
            'support_sha256':array_sha(mp[ex].astype('<f4')),'support_raw_ids_sha256':array_sha(rawids[ex].astype('<i8'))})
    object_ids=[o['mask_id'] for o in objects]
    visibility_hash=hashlib.sha256((array_sha(unique[take])+array_sha(rawids)+array_sha(uv)+array_sha(depth)+array_sha(owners)).encode()).hexdigest()
    result={'schema':'R15_whole_scene_observation_v1','platform':camera_spec.platform,'image_size_wh':[W,H],
        'image_sha256':image_sha256,'camera_spec':camera_spec,'objects':objects,
        'raw_points_input_sha256':array_sha(raw),'camera_input_sha256':hashlib.sha256((array_sha(camera_spec.validated()[0])+array_sha(camera_spec.validated()[1])+array_sha(camera_spec.validated()[2])+camera_spec.platform+camera_spec.missing_distortion_policy).encode()).hexdigest(),
        'visible_raw_points':unique[take].copy(),'visible_points_model':mp.copy(),'physical_raw_ids':rawids,
        'visible_uv':uv.copy(),'depth_camera':depth.copy(),'visible_pixels':pixels,'point_owner':owners,
        'visibility_sha256':visibility_hash,'mask_input_sha256':array_sha(masks),'config':asdict(cfg),'config_sha256':config_sha256(cfg),
        'diagnostics':{'raw_rows':len(raw),'unique_raw_XYZ':len(unique),'visible_before_modelmerge':premerge,'visible_unique_model_XYZ':len(take),
            'float32_model_collisions':premerge-len(take),'unassigned_visible_points':int(np.sum(owners==-1)),
            'ambiguous_multimask_visible_points':int(np.sum(owners==-2)),'exclusive_foreground_points':int(np.sum(owners>=0)),
            'all_masks_before_any_q0_or_semantic_selection':True,'q0_ROI_filter_applied':False,
            'continuous_FOV_then_floor_global_nearest_depth':True,'provided_pose_used':False,
            'camera_diagnostics':pr['diagnostics'],'predicted_mask_ownership_not_GT_identity':True}}
    assert len(object_ids)==len(set(object_ids)) and len(np.unique(mp,axis=0))==len(mp)
    return result
