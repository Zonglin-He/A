"""R13 B fixed-q0 independent-mask support extraction, pure CPU arrays.

R12 unique-target/visibility/model-coordinate/R10-ROI conventions, with original
raw row IDs retained. First noun only checks a category family; it does not verify
full-sentence identity. No mask-index override, query reselection, model, files,
normal/plane/face equality or task-label inputs.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import itertools
import json
import re
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'round7'))
from camera_projection import project_model_points, raw_to_model_points

LEXICON = ('car','vehicle','sedan','van','truck','bus','pedestrian','person','man','woman',
           'people','child','boy','girl','cyclist','biker','bicycle','bike','taxi','suv',
           'jeep','trailer','walker','worker')
MOTOR = frozenset(('vehicle','car','sedan','van','taxi','suv','jeep','truck','bus'))
PERSON = frozenset(('person','pedestrian','man','woman','people','child','boy','girl','walker','worker'))


@dataclass(frozen=True)
class SupportConfig:
    minimum_source_detector_box_IoU: float = .1
    minimum_visible_unique_instance_points_in_source_ROI: int = 3
    minimum_depth_m: float = .1
    crop_fraction: float = .25
    crop_margin_add_m: float = .1
    crop_margin_cap_m: float = .75


DEFAULT_CONFIG = SupportConfig()


def config_sha256(config=DEFAULT_CONFIG):
    return hashlib.sha256(json.dumps(asdict(config),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def parse_target_family(caption):
    if not isinstance(caption,str):
        raise ValueError('Literal input caption string required')
    m=re.search(r'\b(?:'+'|'.join(LEXICON)+r')\b',caption,re.I)
    if m is None:return None,(), 'NA_no_first_lexicon_noun'
    noun=m.group(0).lower()
    if noun in MOTOR:return noun,(3,6,8),'motorvehicle_family_not_exact_subclass_or_literal_identity'
    if noun in PERSON:return noun,(1,),'person_category_not_literal_referring_identity'
    if noun in ('bicycle','bike'):return noun,(2,),'bicycle_category_not_literal_referring_identity'
    return noun,(),'NA_unsupported_or_composite_COCO_noun'


def model_points(raw,camera):
    """R12 model conversion, including actual float32 Drone in-place +1.8."""
    p=np.array(raw,np.float32,copy=True)
    if camera.platform=='drone':p[:,2]+=np.float32(1.8)
    elif camera.platform=='waymo':p=raw_to_model_points(p,'waymo',camera.waymo_view_suffix).astype(np.float32)
    elif camera.platform!='quad':raise ValueError('Unsupported platform')
    return p


def _validate_frame(frame):
    if not isinstance(frame,dict) or set(frame)!={'frame_alias','rgb','raw_points','camera','instances'}:
        raise ValueError('Strict R12 frame arrays only; GT/annotation/override fields forbidden')
    image=np.asarray(frame['rgb']);raw=np.asarray(frame['raw_points']);inst=frame['instances']
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
        raise ValueError('Original-resolution uint8 RGB required')
    if raw.ndim!=2 or raw.shape[1]!=3 or not np.isfinite(raw).all():
        raise ValueError('Finite raw Nx3 required')
    if not isinstance(inst,dict) or set(inst)!={'boxes_xyxy','class_ids','scores','masks_bool','instance_indices'}:
        raise ValueError('Strict independent whole-image instance arrays required')
    labels=np.asarray(inst['class_ids']);n=len(labels);masks=np.asarray(inst['masks_bool'])
    if labels.shape!=(n,) or not np.issubdtype(labels.dtype,np.integer):raise ValueError('Integer instance class IDs required')
    if masks.dtype!=bool or masks.shape!=(n,*image.shape[:2]):raise ValueError('Full-resolution bool instance masks required')
    for name,shape in [('boxes_xyxy',(n,4)),('scores',(n,)),('instance_indices',(n,))]:
        a=np.asarray(inst[name])
        if a.shape!=shape or not np.isfinite(a).all():raise ValueError('Invalid '+name)
    if not np.issubdtype(np.asarray(inst['instance_indices']).dtype,np.integer):raise ValueError('Integer native instance indices required')
    frame['camera'].validated()
    return image,raw,inst


def _visible_with_raw_ids(raw,camera,image_hw,cfg):
    """R12 visibility semantics plus original raw row IDs.

    Canonical unique raw XYZ -> double calibrated projection -> continuous FOV
    -> floor -> nearest depth per pixel, XYZ lexicographic tie break. Z buffering
    occurs BEFORE masks, so background returns cannot reappear behind foreground.
    """
    points,first=np.unique(np.asarray(raw,np.float64),axis=0,return_index=True)
    H,W=image_hw
    pr=project_model_points(raw_to_model_points(points,camera.platform,camera.waymo_view_suffix),
                            camera,(W,H),min_depth=cfg.minimum_depth_m)
    ids=np.flatnonzero(pr['in_image']);pixels=np.floor(pr['uv'][ids]).astype(np.int64)
    key=pixels[:,1]*W+pixels[:,0];p=points[ids]
    order=np.lexsort((p[:,2],p[:,1],p[:,0],pr['depth'][ids],key))
    ids=ids[order];key=key[order]
    ids=ids[np.r_[True,key[1:]!=key[:-1]]] if len(ids) else ids
    return points[ids],first[ids].astype(np.int64),pr['uv'][ids],pr['depth'][ids],{
        'raw_rows':len(raw),'raw_unique_XYZ':len(points),'visible_zbuffer_unique':len(ids),
        'camera_diagnostics':pr['diagnostics'],
        'visibility_rule':'continuous FOV then floor pixel; whole-cloud nearest depth; XYZ tie',
        'visible_original_raw_ids_sha256':_sha(first[ids].astype(np.int64)),
        'visible_raw_XYZ_sha256':_sha(points[ids])}


def extract_fixed_q0_support(*,frame,source6,q0,caption,config=DEFAULT_CONFIG):
    """Return source-ROI support only for a unique compatible independent mask.

    Other mask candidates are tested only for ambiguity. No override parameter is
    accepted. Returned foreground can still be a wrong object; a unique category
    family + q0 overlap is a hypothesis, not literal referring identity evidence.
    """
    cfg=config;image,raw,inst=_validate_frame(frame);source=np.asarray(source6,np.float32)
    if source.shape!=(6,) or not np.isfinite(source).all():raise ValueError('Finite unmodified source6 required')
    if not isinstance(q0,(int,np.integer)) or int(q0)<0:raise ValueError('Fixed nonnegative q0 required')
    H,W=image.shape[:2]
    result={'q0':int(q0),'segmentation_valid':False,'selected_mask_index':None,
        'selected_instance_id':None,'mask_bool':np.zeros((H,W),bool),
        'points_support_model':np.empty((0,3),np.float32),
        'points_support_raw':np.empty((0,3),np.float64),
        'physical_raw_ids':np.empty(0,np.int64),
        'support_raw_ids_before_model_XYZ_merge':np.empty(0,np.int64),
        'visible_raw_ids':np.empty(0,np.int64),'status':'NA_not_computed',
        'diagnostics':{'config':asdict(cfg),'config_sha256':config_sha256(cfg),
            'source_q0_fixed':True,'first_noun_is_not_full_sentence_identity':True,
            'frontend_independent_no_q0_prompt_or_mask_override':True,
            'source_ROI':'R12/R10 padded q0 box min(.25*size+.1m,.75m)',
            'class_family_only_no_verified_identity':True}}
    def fail(status):
        result['status']=status;result['diagnostics']['status']=status
        return result
    if np.any(source[3:]<=0):return fail('NA_nonpositive_source_dimensions')
    noun,family,reason=parse_target_family(caption)
    result['diagnostics']['target_noun']={'noun':noun,'COCO_class_family':list(family),'scope':reason}
    if not family:return fail(reason)
    signs=np.array(list(itertools.product((-1.,1.),repeat=3)))
    pr=project_model_points(source[:3].astype(float)+signs*source[3:].astype(float)/2,frame['camera'],(W,H))
    if np.any(pr['depth']<=cfg.minimum_depth_m) or not np.isfinite(pr['uv']).all():
        return fail('NA_source_projection_not_fully_in_front')
    lo=np.maximum(pr['uv'].min(0),[0,0]);hi=np.minimum(pr['uv'].max(0),[W,H])
    source_area=float(np.maximum(hi-lo,0).prod())
    result['diagnostics']['source_projected_xyxy']=np.r_[lo,hi].tolist()
    if source_area<=0:return fail('NA_source_projection_outside_image')
    points,rawids,uv,depth,vd=_visible_with_raw_ids(raw,frame['camera'],(H,W),cfg)
    result['visible_raw_ids']=rawids.copy()
    model=model_points(points,frame['camera'])
    margin=np.minimum(cfg.crop_fraction*source[3:]+cfg.crop_margin_add_m,cfg.crop_margin_cap_m)
    inside=(np.abs(model-source[:3])<=source[3:]/2+margin).all(1)
    pixels=np.floor(uv).astype(np.int64)
    candidates=[];tests=[];supports={}
    for j,(box,label) in enumerate(zip(inst['boxes_xyxy'],inst['class_ids'])):
        if int(label) not in family:continue
        bb=np.asarray(box,float)
        inter=float(np.maximum(np.minimum(hi,bb[2:])-np.maximum(lo,bb[:2]),0).prod())
        area=float(np.maximum(bb[2:]-bb[:2],0).prod());den=source_area+area-inter
        iou=inter/den if den>0 else 0.
        take=np.flatnonzero(inside & inst['masks_bool'][j,pixels[:,1],pixels[:,0]])
        # R12 uses unique raw observations. Geometry later also collapses rare
        # float32 model-coordinate collisions, retained explicitly below.
        eligible=bool(iou>=cfg.minimum_source_detector_box_IoU and len(take)>=cfg.minimum_visible_unique_instance_points_in_source_ROI)
        tests.append({'mask_array_index':j,'native_instance_id':int(inst['instance_indices'][j]),
            'COCO_class_id':int(label),'source_detector_2D_IoU':iou,
            'visible_unique_raw_foreground_in_source_ROI':len(take),'eligible':eligible})
        if eligible:candidates.append(j);supports[j]=take
    result['diagnostics'].update(visibility=vd,candidate_tests=tests,candidates=candidates,
        ROI_margin_model_m=margin.tolist(),visible_in_source_ROI=int(inside.sum()),
        candidate_detector_scores_do_not_rank_or_break_ambiguity=True)
    if len(candidates)!=1:
        return fail('NA_multiple_independent_samefamily_masks' if len(candidates)>1 else
                    'NA_no_independent_mask_with_depth_ROI_support')
    selected=candidates[0];take=supports[selected]
    selected_model=model[take]
    unique_model,inverse=np.unique(selected_model,axis=0,return_inverse=True)
    # A representative original raw row is retained for every actual model point.
    # Do not reinterpret original row IDs as positions in canonical model arrays.
    representatives=np.array([take[np.flatnonzero(inverse==j)[np.argmin(rawids[take[inverse==j]])]]
                              for j in range(len(unique_model))],np.int64)
    ids=rawids[representatives]
    result.update(status='unique_independent_mask_geometric_hypothesis_not_literal_identity',
        segmentation_valid=True,selected_mask_index=int(selected),
        selected_instance_id=int(inst['instance_indices'][selected]),
        mask_bool=np.array(inst['masks_bool'][selected],copy=True),
        points_support_model=unique_model,points_support_raw=points[representatives],
        physical_raw_ids=ids,support_raw_ids_before_model_XYZ_merge=rawids[take].copy())
    result['diagnostics'].update(status=result['status'],selected_visible_raw_points=len(take),
        selected_unique_model_XYZ=len(unique_model),float32_model_collisions=len(take)-len(unique_model),
        support_model_XYZ_sha256=_sha(unique_model),support_original_raw_ids_sha256=_sha(ids),
        mask_sha256=_sha(result['mask_bool']),
        raw_ID_basis='original raw array row IDs, first raw XYZ occurrence; canonical modelXYZ representative; never canonical model index',
        wrong_unique_mask_can_pass=True,
        geometry_min8_gate_separate_from_association_min3=True)
    assert np.array_equal(result['points_support_model'],model_points(result['points_support_raw'],frame['camera']))
    assert np.array_equal(np.asarray(raw,np.float64)[ids],result['points_support_raw'])
    return result
