"""Single-pass spatial guidance of TA-STVG's actual temporal cross-attention.

Diagnostic only; does not modify any released method. No GT reader, optimizer,
pixel crop, temporal resampling, DINO confidence-as-event-label, or feedback
iteration. Text/global contextual features and the native temporal query stay.
"""
import math
import numpy as np
import torch


def tree(x, device=None):
    if torch.is_tensor(x):
        return x.detach().clone().to(device) if device is not None else x.detach().clone()
    if isinstance(x, dict):
        return {k:tree(v,device) for k,v in x.items()}
    if isinstance(x, (list,tuple)):
        return type(x)(tree(v,device) for v in x)
    return x


def capture(model, batch):
    """Capture only the final native refinement pass of each official offset."""
    from vg_tta.decota_tastvg_episode_v1 import forward
    captures=[]; size={}
    def grid(_m,args,kw):
        size['hw']=tuple(map(int,kw['encoded_info']['fea_map_size']))
    def inputs(_m,args,kw):
        captures.append(dict(kwargs=tree(kw),grid=size['hw']))
    handles=[model.ground_decoder.register_forward_pre_hook(grid,with_kwargs=True),
             model.ground_decoder.time_decoder.register_forward_pre_hook(inputs,with_kwargs=True)]
    try:
        base,hs,records=forward(model,batch)
    finally:
        for h in handles:h.remove()
    assert len(captures)==4
    caps=[captures[1],captures[3]]
    for c,r,h in zip(caps,records,hs):
        c['frame_ids']=list(r['frame_ids'])
        c['native_head_input']=h.detach().clone()
        c['native_logits']=r['temporal_logits'].clone()
        n=len(c['frame_ids']);height,width=c['grid'];kw=c['kwargs']
        assert kw['encoded_feature'].shape[1]==n
        assert kw['encoded_mask'].shape==(n,kw['encoded_feature'].shape[0])
        # These single-video inputs have no spatial padding. Fail visibly if a
        # different runtime violates that premise; don't compare unequal masks.
        assert not kw['encoded_mask'][:,-height*width:].any()
    return base,caps,records


def rectangle(box, height, width):
    box=np.asarray(box,dtype=float)
    if box.shape!=(4,) or not np.isfinite(box).all() or (box[2:]<=0).any():
        raise ValueError('finite valid normalized cxcywh required')
    lo=np.clip(box[:2]-box[2:]/2,0,1);hi=np.clip(box[:2]+box[2:]/2,0,1)
    x0=min(width-1,int(np.floor(lo[0]*width)));y0=min(height-1,int(np.floor(lo[1]*height)))
    x1=max(x0+1,min(width,int(np.ceil(hi[0]*width))));y1=max(y0+1,min(height,int(np.ceil(hi[1]*height))))
    return (y0,y1,x0,x1)


def region_masks(frame_ids, anchors, native_boxes, grid, *, kind='expert', seed=20260910):
    """Controls have the EXACT accepted frame support. Random/native-center
    keep each expert rectangle's rasterized width/height, not just float area.
    """
    height,width=grid;out=np.zeros((len(frame_ids),height,width),bool)
    native_boxes=np.asarray(native_boxes);byid={int(a['frame_id']):a for a in anchors}
    assert len(byid)==len(anchors)
    rects=[]
    for i,fid in enumerate(frame_ids):
        if fid not in byid:continue
        a=byid[fid];r=rectangle(a['box'],height,width)
        y0,y1,x0,x1=r;hh=y1-y0;ww=x1-x0
        if kind=='native_box':r=rectangle(native_boxes[i],height,width)
        elif kind=='native_center':
            cx,cy=native_boxes[i,:2]
            x0=int(np.clip(round(cx*width-ww/2),0,width-ww));y0=int(np.clip(round(cy*height-hh/2),0,height-hh))
            r=(y0,y0+hh,x0,x0+ww)
        elif kind=='random':
            rng=np.random.default_rng(np.random.SeedSequence([int(seed),int(fid)]))
            x0=int(rng.integers(width-ww+1));y0=int(rng.integers(height-hh+1))
            r=(y0,y0+hh,x0,x0+ww)
        elif kind not in ('expert','uniform_mass'):
            raise ValueError(kind)
        y0,y1,x0,x1=r;out[i,y0:y1,x0:x1]=True
        rects.append(dict(frame_id=fid,rectangle=list(r),area=int(out[i].sum())))
    return torch.from_numpy(out.reshape(len(frame_ids),height*width)),rects


def guided_replay(model, cache, roi, *, gain=4., uniform_mass=False):
    """Add log(gain) to ROI attention logits at EVERY time-decoder layer.

    Not hard masking: all original valid keys remain available. The matched
    uniform control distributes the same sum of exp(bias) across visual keys.
    Language keys and pre-existing padding are unchanged. gain=1 returns the
    exact native boolean-mask path, avoiding a dtype/kernel no-op confound.
    """
    if gain<1 or not math.isfinite(gain):raise ValueError('gain >= 1 required')
    kw=dict(cache['kwargs']);roi=roi.to(kw['encoded_feature'].device)
    height,width=cache['grid'];l=height*width
    assert roi.dtype==torch.bool and roi.shape==(kw['encoded_feature'].shape[1],l)
    altered=bool(roi.any()) and gain!=1
    if altered:
        original=kw['encoded_mask'];mask=torch.zeros_like(original,dtype=torch.float32)
        bias=roi.float()*math.log(gain)
        if uniform_mass:
            bias=torch.log1p(roi.float().mean(1,keepdim=True)*(gain-1)).expand_as(bias)
        mask[:,-l:]=bias;mask.masked_fill_(original,-torch.inf)
        kw['encoded_mask']=mask
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        hidden=model.ground_decoder.time_decoder(**kw)
        logits=model.temp_embed(hidden)[-1]
    assert torch.isfinite(logits).all()
    return logits.detach().cpu(),hidden.detach(),dict(altered=altered,gain=gain,
        guided_frames=int(roi.any(1).sum()),region_tokens=int(roi.sum()),
        language_keys_unchanged=True,global_context_kept=True,uniform_mass=uniform_mass)


def combine(logits, records, frame_ids):
    from vg_tta.decota_tastvg_episode_v1 import fitted_merge
    # Official TA postprocessing runs FP16 log_softmax and endpoint addition
    # on CUDA. CPU FP16 log_softmax can round ties differently despite exact
    # logits. Retain the native device for EVERY diagnostic decode.
    indices=fitted_merge([z.cuda() for z in logits],records,frame_ids)
    z=torch.stack([logits[i%2][0,i//2].float() for i in range(len(frame_ids))])
    return z,list(indices)


def readouts(z,frame_ids,native_indices,decota_indices,free_indices):
    from vg_tta.native_coverage_calibration_v1 import native_at_length
    a=native_at_length(z,frame_ids,reference=decota_indices)
    b=native_at_length(z,frame_ids,reference=native_indices)
    return dict(decota_length=list(a['indices']),native_length=list(b['indices']),
                free=list(free_indices)),dict(decota_length=a,native_length=b)
