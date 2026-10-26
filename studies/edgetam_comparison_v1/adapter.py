"""Same-eight-box EdgeTAM adapter; import never constructs a predictor."""
from contextlib import contextmanager
import numpy as np
import torch
from studies.review_strengthening_v1.sam2_input_adapter_v1 import lossless_rgb_loader,mask_box

def pixel_boxes(native,height,width):
    b=np.asarray(native,dtype=np.float64)
    assert b.shape==(8,4)and np.isfinite(b).all()and(b[:,2:]>=0).all()
    xy=np.concatenate([b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2],axis=1)
    return xy*np.asarray([width,height,width,height],np.float64)

def keep_observed(boxes,native,positions):
    out=np.asarray(boxes,dtype=np.float64).copy();native=np.asarray(native)
    assert out.ndim==2 and out.shape[1]==4 and len(set(positions))==8
    out[list(positions)]=native
    assert np.array_equal(out[list(positions)],native)
    return out

@contextmanager
def scoped_rgb_loader(module):
    old=module.load_video_frames;module.load_video_frames=lossless_rgb_loader
    try:yield
    finally:module.load_video_frames=old

@contextmanager
def checkpoint_only_backbone(timm_backbone_module):
    """Avoid redundant ImageNet download; complete official checkpoint loads strictly.

    All original vendor files stay unchanged. At actual model construction the
    caller must verify every final state tensor against the pinned EdgeTAM
    checkpoint. This context is not used by the CPU preparation preflight.
    """
    original=timm_backbone_module.create_model
    def create(name,**kwargs):
        assert name=='repvit_m1.dist_in1k'and kwargs['pretrained']is True
        return original(name,**{**kwargs,'pretrained':False})
    timm_backbone_module.create_model=create
    try:yield
    finally:timm_backbone_module.create_model=original

def propagate_boxes(predictor,module,frames,positions,native,checkpoint_scores):
    """Uncalled until an independently frozen GPU pilot is released.

    All eight original boxes condition one object before tracking. Official
    forward starts explicitly at frame0 and must yield every native frame once.
    No reverse pass, custom fusion, our DP, GT, text or new expert is used.
    Raw scores are checkpointed by the caller before finite/mask/box processing.
    """
    positions=[int(p)for p in positions];assert len(set(positions))==8
    prompts=pixel_boxes(native,*frames.shape[1:3])
    with scoped_rgb_loader(module):
        state=predictor.init_state(frames,offload_video_to_cpu=True,
                                  offload_state_to_cpu=False,async_loading_frames=False)
    for j in sorted(range(8),key=lambda j:positions[j]):
        predictor.add_new_points_or_box(state,frame_idx=positions[j],obj_id=1,
            box=prompts[j],clear_old_points=True,normalize_coords=True)
    assert set(state['temp_output_dict_per_obj'][0]['cond_frame_outputs'])==set(positions)
    boxes={};empty=[]
    for i,obj,scores in predictor.propagate_in_video(state,start_frame_idx=0,reverse=False):
        checkpoint_scores(i,scores)  # Caller preserves paid outputs before assertions.
        assert obj==[1]and scores.shape==(1,1,*frames.shape[1:3])
        assert i not in boxes and torch.isfinite(scores).all()
        mask=(scores[0,0]>0).detach().cpu().numpy();boxes[i]=mask_box(mask)
        if not mask.any():empty.append(i)
    assert set(boxes)==set(range(len(frames)))
    raw=np.stack([boxes[i]for i in range(len(frames))])
    return keep_observed(raw,native,positions),dict(full_native_frame_coverage=True,
        all_eight_box_prompts=True,one_object=True,forward_start=0,reverse=False,
        observed_outputs_exact_input_boxes=True,empty_propagated_frames=empty,
        no_Source_fallback=True,no_our_DP=True,raw_mask_boxes=raw)
