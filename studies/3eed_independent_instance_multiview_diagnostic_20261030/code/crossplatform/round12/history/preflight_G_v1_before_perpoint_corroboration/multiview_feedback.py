"""R12 fixed-q0 geometry from independent masks and verified static multiview.
Pure CPU arrays. No files/models/GT. Direct boxes are no-parameter controls.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib,re,sys
import numpy as np
D=Path(__file__).resolve().parent;sys.path[:0]=[str(D),str(D.parent/'round10'),str(D.parent/'round7')]
import temporal_association as temporal
from geometry_feedback import generate_geometry_targets,GeometryConfig,box_corners
from rgb_support import make_rgb_support,RGBSupportConfig
from camera_projection import project_model_points,raw_to_model_points
ARMS=('source','grabcut_direct','single_instance_direct','multiview_instance_direct','wrongview_instance_direct','single_instance_half','multiview_instance_half','multiview_maskmatch_only_direct')
LEXICON=('car','vehicle','sedan','van','truck','bus','pedestrian','person','man','woman','people','child','boy','girl','cyclist','biker','bicycle','bike','taxi','suv','jeep','trailer','walker','worker')
MOTOR=frozenset(('vehicle','car','sedan','van','taxi','suv','jeep','truck','bus'))
PERSON=frozenset(('person','pedestrian','man','woman','people','child','boy','girl','walker','worker'))
@dataclass(frozen=True)
class FeedbackConfig:
    minimum_source_detector_box_IoU:float=.1
    minimum_visible_unique_instance_points_in_source_ROI:int=3
    fixed_half_fraction:float=.5
    source_ROI_uses_matched_R10_padding:bool=True
    veto_all_multiview_on_verified_static_contradiction:bool=True
DEFAULT_CONFIG=FeedbackConfig()

def parse_target_family(caption):
    m=re.search(r'\b(?:'+'|'.join(LEXICON)+r')\b',caption,re.I)
    if m is None:return None,(), 'NA_no_first_lexicon_noun'
    noun=m.group(0).lower()
    if noun in MOTOR:return noun,(3,6,8),'motorvehicle_family_not_exact_subclass_or_literal_identity'
    if noun in PERSON:return noun,(1,),'person_category_not_literal_referring_identity'
    if noun in ('bicycle','bike'):return noun,(2,),'bicycle_category_not_literal_referring_identity'
    return noun,(),'NA_unsupported_or_composite_COCO_noun'

def _sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def model_points(raw,camera):
    p=np.array(raw,np.float32,copy=True)
    if camera.platform=='drone':p[:,2]+=np.float32(1.8)
    elif camera.platform=='waymo':p=raw_to_model_points(p,'waymo',camera.waymo_view_suffix).astype(np.float32)
    elif camera.platform!='quad':raise ValueError('unsupported platform')
    return p

def _frame(frame):
    required={'frame_alias','rgb','raw_points','camera','instances'}
    if set(frame)!=required:raise ValueError('strict frame array interface required; no GT/annotation fields accepted')
    im=np.asarray(frame['rgb']);p=np.asarray(frame['raw_points']);inst=frame['instances']
    if im.dtype!=np.uint8 or im.ndim!=3 or im.shape[2]!=3 or p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all():raise ValueError('finite rawXYZ and uint8 RGB required')
    required_inst={'boxes_xyxy','class_ids','scores','masks_bool','instance_indices'}
    if set(inst)!=required_inst:raise ValueError('strict independent-instance arrays required')
    n=len(inst['class_ids']);m=np.asarray(inst['masks_bool'])
    if m.dtype!=bool or m.shape!=(n,*im.shape[:2]) or np.asarray(inst['boxes_xyxy']).shape!=(n,4) or np.asarray(inst['scores']).shape!=(n,) or np.asarray(inst['instance_indices']).shape!=(n,):raise ValueError('invalid independent whole-image mask schema')
    frame['camera'].validated();return frame

def _union(frame):
    m=frame['instances']['masks_bool'];return m.any(0) if len(m) else np.zeros(frame['rgb'].shape[:2],bool)
def _family_labels(labels):
    x=np.asarray(labels).copy();x[np.isin(x,[3,6,8])]=3;return x

def _unique_target(anchor,source,family,config):
    cam=anchor['camera'];im=anchor['rgb'];H,W=im.shape[:2]
    pr=project_model_points(box_corners(source),cam,(W,H))
    if np.any(pr['depth']<=.1) or not np.isfinite(pr['uv']).all():return None,{'status':'NA_source_projection_not_fully_in_front','candidates':[]}
    lo=np.maximum(pr['uv'].min(0),[0,0]);hi=np.minimum(pr['uv'].max(0),[W,H]);srcarea=float(np.maximum(hi-lo,0).prod())
    if srcarea<=0:return None,{'status':'NA_source_projection_outside_image','candidates':[]}
    p,uv,vd=temporal._visible_raw(anchor['raw_points'],cam,(H,W),np.zeros((H,W),bool),temporal.DEFAULT_CONFIG)
    mp=model_points(p,cam);gc=GeometryConfig();margin=np.minimum(gc.crop_fraction*source[3:]+gc.crop_margin_add_m,gc.crop_margin_cap_m)
    inside=(np.abs(mp-source[:3])<=source[3:]/2+margin).all(1);pix=np.floor(uv).astype(int)
    candidates=[];alltests=[]
    ins=anchor['instances']
    for i,(box,label) in enumerate(zip(ins['boxes_xyxy'],ins['class_ids'])):
        if int(label) not in family:continue
        bb=np.asarray(box,float);inter=float(np.maximum(np.minimum(hi,bb[2:])-np.maximum(lo,bb[:2]),0).prod());area=float(np.maximum(bb[2:]-bb[:2],0).prod());den=srcarea+area-inter;iou=inter/den if den>0 else 0.
        count=int(np.sum(inside & ins['masks_bool'][i,pix[:,1],pix[:,0]]))
        good=iou>=config.minimum_source_detector_box_IoU and count>=config.minimum_visible_unique_instance_points_in_source_ROI
        alltests.append({'instance_index':i,'COCO_class_id':int(label),'source_detector_2D_IoU':iou,'visible_unique_foreground_in_source_ROI':count,'eligible':bool(good)})
        if good:candidates.append(i)
    status='unique_independent_mask_geometric_hypothesis_not_literal_identity' if len(candidates)==1 else 'NA_multiple_independent_samefamily_masks' if len(candidates)>1 else 'NA_no_independent_mask_with_depth_ROI_support'
    return candidates[0] if len(candidates)==1 else None,{'status':status,'candidates':candidates,'candidate_tests':alltests,'source_ROI':'matched R10 padded sourcebox; q0 only hypothesis, never frontend seed','visibility':vd}

def _endpoint_mask(points,mask):
    x=np.asarray(points);H,W=mask.shape;good=np.isfinite(x).all(1)&(x[:,0]>=0)&(x[:,0]<W)&(x[:,1]>=0)&(x[:,1]<H)
    ids=np.flatnonzero(good);pix=np.floor(x[ids]).astype(int);good[ids]&=mask[pix[:,1],pix[:,0]];return good

def _attempt_view(anchor,neighbor,target_index,seed):
    A=anchor;B=_frame(neighbor)
    if A['camera'].platform!=B['camera'].platform:return {'status':'NA_cross_platform_view','verified':False,'contradiction':False}
    reg=temporal.estimate_pair_registration(rgb_A=A['rgb'],raw_points_A=A['raw_points'],camera_A=A['camera'],rgb_B=B['rgb'],raw_points_B=B['raw_points'],camera_B=B['camera'],excluded_mask_A=_union(A),excluded_mask_B=_union(B),seed=seed)
    diag={'frame_alias':B['frame_alias'],'registration':reg,'verified':False,'contradiction':False,'clock_unverified_no_velocity':True}
    if not reg['accepted'] or not reg['observation_diagnostics']['background_only_qualification']:diag['status']='NA_sensor_registration_or_background_fit_protocol';return diag
    try:flow=temporal.match_instance_masks(rgb_A=A['rgb'],rgb_B=B['rgb'],masks_A=A['instances']['masks_bool'],masks_B=B['instances']['masks_bool'],class_ids_A=_family_labels(A['instances']['class_ids']),class_ids_B=_family_labels(B['instances']['class_ids']))
    except ValueError as error:diag.update(status='NA_mask_flow_sensor_shape',reason=str(error));return diag
    matches=[m for m in flow['matches'] if m['instance_A']==target_index]
    diag['mask_matching']={k:v for k,v in flow.items() if k not in ('flow_points_A_to_B','flow_points_B_to_A')}
    if len(matches)!=1:diag['status']='NA_no_unique_bidirectional_mask_track';return diag
    match=matches[0];j=match['instance_B'];ma=A['instances']['masks_bool'][target_index];mb=B['instances']['masks_bool'][j]
    fa=flow['flow_points_A_to_B'][target_index];fb=flow['flow_points_B_to_A'][j]
    keepa=_endpoint_mask(fa['end_xy'],mb);keepb=_endpoint_mask(fb['end_xy'],ma)
    xyA=np.r_[fa['start_xy'][keepa],fb['end_xy'][keepb]];xyB=np.r_[fa['end_xy'][keepa],fb['start_xy'][keepb]]
    bound=temporal.bind_raw_depth_to_pixel_pairs(pixel_A=xyA,pixel_B=xyB,raw_points_A=A['raw_points'],camera_A=A['camera'],raw_points_B=B['raw_points'],camera_B=B['camera'],image_hw_A=A['rgb'].shape[:2],image_hw_B=B['rgb'].shape[:2],include_mask_A=ma,include_mask_B=mb)
    n=bound['n_unique_physical_pairs'];diag.update(neighbor_mask_index=j,matched_COCO_classes=[int(A['instances']['class_ids'][target_index]),int(B['instances']['class_ids'][j])],target_depthbound_unique_pairs=n,depthbound_pairs_sha256=_sha(np.c_[bound['points_A'],bound['points_B']]))
    if n<temporal.DEFAULT_CONFIG.minimum_cycle_tracks:diag['status']='NA_insufficient_depthbound_track_pairs_not_proven_motion';return diag
    static=temporal.check_static_object_support(points_A=bound['points_A'],points_B=bound['points_B'],T_raw_B_to_A=reg['T_raw_B_to_A']);diag['static_support']=static
    if not static['accepted']:diag.update(status='contradiction_static_target_support',contradiction=True);return diag
    foreground,uv,vd=temporal._visible_raw(B['raw_points'],B['camera'],B['rgb'].shape[:2],~mb,temporal.DEFAULT_CONFIG)
    diag.update(status='verified_static_multiview_hypothesis_not_literal_identity',verified=True,neighbor_visible_foreground_raw=foreground,neighbor_foreground_sha256=_sha(foreground),neighbor_visibility=vd)
    return diag

def _geometry(anchor,source,q0,mask,rawpoints):
    p=model_points(rawpoints,anchor['camera']);p=np.unique(p,axis=0);return generate_geometry_targets(points_fit=p,raw_indices=np.arange(len(p),dtype=np.int64),source_box6=source,q0=q0,camera_spec=anchor['camera'],rgb_mask2d=mask,segmentation_valid=bool(mask.any()))

def _multiview_geometry(anchor,source,q0,mask,attempts):
    verified=[v for v in attempts if v['verified']];contradiction=any(v['contradiction'] for v in attempts)
    diag={'neighbor_n':len(attempts),'verified_neighbor_n':len(verified),'contradiction_veto':contradiction,'original_raw_unique_n':len(np.unique(anchor['raw_points'],axis=0)),'neighbors':[]}
    if contradiction:return None,diag,'NA_multiview_static_contradiction_all_sides_veto'
    if not verified:return None,diag,'NA_no_verified_static_neighbor'
    arrays=[np.asarray(anchor['raw_points'],np.float32)];H,W=anchor['rgb'].shape[:2];gc=GeometryConfig();margin=np.minimum(gc.crop_fraction*source[3:]+gc.crop_margin_add_m,gc.crop_margin_cap_m)
    for v in verified:
        fg=v['neighbor_visible_foreground_raw'];moved=temporal.transform_points(fg,v['registration']['T_raw_B_to_A']);mp=model_points(moved,anchor['camera']);pr=project_model_points(mp,anchor['camera'],(W,H));keep=pr['in_image'].copy();ids=np.flatnonzero(keep);pix=np.floor(pr['uv'][ids]).astype(int);keep[ids]&=mask[pix[:,1],pix[:,0]]
        fov=int(pr['in_image'].sum());inmask=int(keep.sum());keep&=(np.abs(mp-source[:3])<=source[3:]/2+margin).all(1)
        allowed=moved[keep].astype(np.float32);arrays.append(allowed)
        diag['neighbors'].append({'frame_alias':v['frame_alias'],'visible_instance_points_before_transform':len(fg),'anchor_FOV':fov,'anchor_mask':inmask,'anchor_mask_and_source_ROI':len(allowed),'added_raw_support_sha256':_sha(allowed)})
    merged=np.unique(np.concatenate(arrays),axis=0);diag.update(merged_raw_unique_n=len(merged),new_raw_unique_count=len(merged)-diag['original_raw_unique_n'],merged_raw_support_sha256=_sha(merged),no_whole_background_neighbor_aggregated=True,anchor_mask_unchanged=True)
    return _geometry(anchor,source,q0,mask,merged),diag,'verified_static_geometry_fit'

def build_geometry_feedback(*,anchor,neighbors,wrong_neighbor,source_box6,q0,caption,seed=0,config=DEFAULT_CONFIG):
    A=_frame(anchor);source=np.asarray(source_box6,np.float32)
    if source.shape!=(6,) or not np.isfinite(source).all() or not isinstance(q0,(int,np.integer)):raise ValueError('finite original source6 and fixed query required')
    if len(neighbors)>2:raise ValueError('at most two fixed-order retrieved neighbors')
    boxes=np.tile(source,(8,1));accept=np.zeros((8,6),bool);provenance=np.full((8,6),'source_prior',dtype='U200');support=np.array(['0'*64]*8);status=['source_q0']+['source_fallback_not_computed']*7
    out={'arm_names':np.array(ARMS),'arm_boxes_raw6':boxes,'arm_query_index':np.full(8,int(q0),np.int64),'acceptance_mask6':accept,'point_support_sha256':support,'provenance':provenance,'arm_status':status,'diagnostics':{'config':asdict(config),'q0_fixed':True,'neural_parameter_updates':False,'independent_frontend_no_q0_seed':True,'clock_unverified_no_speed_or_causal_video_claim':True,'samefamily_not_literal_caption_identity':True}}
    if np.any(source[3:]<=0):status[1:]=['NA_nonpositive_source_dimensions']*7;return out
    p=model_points(A['raw_points'],A['camera']);p=np.unique(p,axis=0)
    grab=make_rgb_support(rgb=A['rgb'],points_fit=p,source_box6=source,camera_spec=A['camera'],config=RGBSupportConfig())
    gg=generate_geometry_targets(points_fit=p,raw_indices=np.arange(len(p),dtype=np.int64),source_box6=source,q0=q0,camera_spec=A['camera'],rgb_mask2d=grab['mask2d'],segmentation_valid=bool(grab['valid']))
    def assign(a,g,label=None):
        boxes[a]=np.asarray(g['arm_boxes_raw6'][1],np.float32);accept[a]=g['acceptance_mask6'][1];support[a]=g['point_support_sha256'][1];provenance[a]=g['provenance'][1];status[a]=label or g['arm_status'][1]
    assign(1,gg);out['diagnostics']['grabcut']={'segmentation':grab['diagnostics'],'geometry':gg}
    noun,family,nounreason=parse_target_family(caption);out['diagnostics']['target_noun']={'noun':noun,'COCO_class_family':family,'scope':nounreason}
    if not family:status[2:]=[nounreason]*6;return out
    selected,association=_unique_target(A,source,family,config);out['diagnostics']['anchor_association']=association
    if selected is None:status[2:]=[association['status']]*6;return out
    mask=A['instances']['masks_bool'][selected];single=_geometry(A,source,q0,mask,A['raw_points']);assign(2,single)
    boxes[5]=source+np.float32(config.fixed_half_fraction)*(boxes[2]-source);accept[5]=accept[2];support[5]=support[2];provenance[5]=provenance[2];status[5]='fixed_half_of_single_instance_direct'
    out['diagnostics'].update(selected_anchor_mask_index=selected,single_geometry=single,anchor_mask_sha256=_sha(mask))
    attempts=[_attempt_view(A,B,selected,seed+i+1) for i,B in enumerate(neighbors)]
    mv,md,ms=_multiview_geometry(A,source,q0,mask,attempts);out['diagnostics']['multiview']=md
    # Retain all attempted sides; raw neighbor arrays are represented by hashes/counts only.
    out['diagnostics']['neighbor_attempts']=[{k:v for k,v in x.items() if k!='neighbor_visible_foreground_raw'} for x in attempts]
    if mv is not None:
        assign(3,mv);assign(7,single,'verified_matchgate_anchor_geometry_only');out['diagnostics']['multiview_geometry']=mv
        boxes[6]=source+np.float32(config.fixed_half_fraction)*(boxes[3]-source);accept[6]=accept[3];support[6]=support[3];provenance[6]=provenance[3];status[6]='fixed_half_of_multiview_instance_direct'
    else:status[3]=status[6]=status[7]=ms
    if wrong_neighbor is None:status[4]='NA_prelocked_wrong_view_missing_no_replacement'
    else:
        wrong=_attempt_view(A,wrong_neighbor,selected,seed+101);wg,wd,ws=_multiview_geometry(A,source,q0,mask,[wrong]);out['diagnostics']['wrongview']={'attempt':{k:v for k,v in wrong.items() if k!='neighbor_visible_foreground_raw'},'geometry_inputs':wd,'status':ws}
        if wg is not None:assign(4,wg);out['diagnostics']['wrongview_geometry']=wg
        else:status[4]=ws
    assert np.isfinite(boxes).all() and np.all(boxes[:,3:]>0) and np.array_equal(boxes[0],source)
    return out
