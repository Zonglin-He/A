"""R13 support-only synthetic audit: no files except this code/receipt, no model."""
from pathlib import Path
import sys,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'crossplatform/round13'))
from support import extract_fixed_q0_support,model_points,DEFAULT_CONFIG,config_sha256
from camera_projection import CameraSpec


def run_checks():
    spec=CameraSpec('quad',np.array([[100.,0,80],[0,100.,80],[0,0,1]]),np.eye(4),
                    missing_distortion_policy='assume_zero_declared')
    source=np.array([0,0,5,2,2,2],np.float32)
    xy=np.array([(x,y) for x in np.linspace(-.4,.4,5) for y in np.linspace(-.4,.4,5)])
    front=np.c_[xy,np.full(25,5.)].astype(np.float32)
    behind=(front*1.6).astype(np.float32)
    neighbor=np.array([[3+x,y,5] for x,y in xy],np.float32)
    raw=np.r_[front,behind,neighbor,front[:3]]
    mask=np.zeros((160,160),bool);mask[69:92,69:92]=True
    other=np.zeros_like(mask);other[68:93,128:154]=True
    def frame(points=raw,masks=None,boxes=None,labels=None):
        masks=np.asarray([mask,other] if masks is None else masks)
        n=len(masks)
        return {'frame_alias':'synthetic_only','rgb':np.zeros((160,160,3),np.uint8),
            'raw_points':points,'camera':spec,
            'instances':{'boxes_xyxy':np.array([[65,65,98,98],[125,65,155,98]] if boxes is None else boxes,float),
                         'class_ids':np.array([3,3] if labels is None else labels,np.int64),
                         'scores':np.full(n,.8),'masks_bool':masks,
                         'instance_indices':np.arange(n,dtype=np.int64)+10}}
    base=extract_fixed_q0_support(frame=frame(),source6=source,q0=7,caption='the car near another car')
    assert base['segmentation_valid'] and base['q0']==7 and base['selected_mask_index']==0
    assert len(base['points_support_model'])==25
    assert np.array_equal(base['points_support_raw'],raw[base['physical_raw_ids']].astype(float))
    assert np.all(base['points_support_model'][:,2]==5)
    assert np.all(base['physical_raw_ids']<25)
    checks={'background_neighbor_ROI_and_occluded_background_excluded':True,
            'original_raw_row_IDs_replay_exact_and_front_depth_only':True,
            'fixed_q0_no_reselection':True}
    multi=extract_fixed_q0_support(frame=frame(masks=[mask,mask],boxes=[[65,65,98,98]]*2),
        source6=source,q0=7,caption='the car')
    assert not multi['segmentation_valid'] and multi['status']=='NA_multiple_independent_samefamily_masks'
    assert len(multi['physical_raw_ids'])==0 and not multi['mask_bool'].any()
    checks['samefamily_multiple_candidates_NA_no_confidence_tie']=True
    unsupported=extract_fixed_q0_support(frame=frame(),source6=source,q0=7,caption='cyclist beside a car')
    assert unsupported['status']=='NA_unsupported_or_composite_COCO_noun'
    checks['unsupported_firstnoun_no_later_object_GT_fallback']=True
    wrongclass=extract_fixed_q0_support(frame=frame(labels=[1,1]),source6=source,q0=7,caption='the car')
    assert not wrongclass['segmentation_valid']
    checks['incompatible_category_family_NA']=True
    # A wrong unique mask on a second same-family object in q0 ROI can satisfy
    # the geometric check. This is retained as a limitation, never called true ID.
    target=np.array([[-.4+x,y,5] for x in [-.04,0,.04] for y in [-.04,0,.04]],np.float32)
    distract=np.array([[.55+x,y,5] for x in [-.05,0,.05] for y in [-.05,0,.05]],np.float32)
    wrongmask=np.zeros_like(mask);wrongmask[76:85,88:96]=True
    wrong=extract_fixed_q0_support(frame=frame(points=np.r_[target,distract],masks=[wrongmask],
        boxes=[[70,70,110,100]],labels=[3]),source6=source,q0=7,caption='the car on the left')
    assert wrong['segmentation_valid'] and np.all(wrong['points_support_model'][:,0]>.3)
    assert wrong['diagnostics']['wrong_unique_mask_can_pass']
    checks['wrong_unique_mask_can_pass_not_full_sentence_identity']=True
    badframe=frame();badframe['GT_box']=[1,2,3]
    try:extract_fixed_q0_support(frame=badframe,source6=source,q0=7,caption='car')
    except ValueError:checks['annotation_GT_extra_fields_rejected']=True
    else:raise AssertionError('GT extra field accepted')
    try:extract_fixed_q0_support(frame=frame(),source6=source,q0=7,caption='car',mask_index_override=0)
    except TypeError:checks['no_mask_index_override_API']=True
    else:raise AssertionError('override accepted')
    order=np.random.default_rng(4).permutation(len(raw));permuted=extract_fixed_q0_support(
        frame=frame(points=raw[order]),source6=source,q0=7,caption='the car')
    assert np.array_equal(permuted['points_support_model'],base['points_support_model'])
    assert np.array_equal(permuted['points_support_raw'],raw[order][permuted['physical_raw_ids']])
    checks['point_permutation_support_invariant_original_IDs_rebased_correctly']=True
    # Actual R12 visibility/helper bridge uses double raw projection; only model
    # geometry delivery uses the original float32 +1.8 convention.
    sys.path.insert(0,str(ROOT/'crossplatform/round12'))
    import temporal_association as old
    r12p,r12uv,_=old._visible_raw(raw,spec,(160,160),np.zeros((160,160),bool),old.DEFAULT_CONFIG)
    visible=raw[base['visible_raw_ids']].astype(float)
    assert np.array_equal(r12p,visible)
    checks['R12_wholecloud_zbuffer_visibleXYZ_bitexact_bridge']=True
    dspec=CameraSpec('drone',spec.intrinsic,np.eye(4),missing_distortion_policy='assume_zero_declared')
    p=np.array([[0,0,3.2],[.2,.1,4]],np.float32);expected=p.copy();expected[:,2]+=np.float32(1.8)
    assert np.array_equal(model_points(p,dspec),expected)
    checks['Drone_float32_inplace_model_shift_exact']=True
    # Negative-small u must not become pixel0, and valid right-edge u must not
    # be rounded out. They do not change q0 support, but wholecloud IDs expose it.
    edge=np.array([[-4.02,0,5],[3.995,0,5],[4.,0,5]],np.float32)
    er=extract_fixed_q0_support(frame=frame(points=np.r_[front,edge]),source6=source,q0=7,caption='car')
    assert 25 not in er['visible_raw_ids'] and 26 in er['visible_raw_ids'] and 27 not in er['visible_raw_ids']
    checks['continuous_FOV_before_floor_boundary']=True
    json.dumps(base['diagnostics'],allow_nan=False)
    return {'state':'passed_synthetic_support_checks_no_real_task_evidence','checks':checks,
        'config':vars(DEFAULT_CONFIG),'config_sha256':config_sha256(),
        'real_sensor_or_GT_or_model_or_GPU_access':False,
        'wrong_unique_mask_fixture_scope':'same-family q0-compatible support can be wrong referring object; relation text not resolved',
        'program_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [ROOT/'crossplatform/round13/support.py',Path(__file__),ROOT/'crossplatform/round12/multiview_feedback.py',
             ROOT/'crossplatform/round12/temporal_association.py',ROOT/'crossplatform/round7/camera_projection.py']}}


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    r=run_checks();out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'state':r['state'],'n_checks':len(r['checks']),'config_sha256':r['config_sha256']}))
