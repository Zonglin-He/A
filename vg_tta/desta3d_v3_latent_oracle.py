"""Source-GT oracle on post-reader THW features; no pixel or parameter updates."""
from contextlib import contextmanager
import hashlib
import numpy as np
import torch

def masks_from_source_record(label,frame_ids,height,width):
    """Physical-time indicator and per-cell normalized GT box area coverage.

    Spatial evidence uses all available valid referent boxes, independently of
    the event interval. Missing boxes are neutral ones, not negative evidence.
    """
    if list(frame_ids)!=label['frame_ids'] or height<1 or width<1:raise ValueError('Oracle grid identity mismatch')
    ids=np.asarray(frame_ids);n=len(ids)
    interval=label['event_interval'];active=(ids>=interval['begin_fid'])&(ids<interval['end_fid'])
    if not np.array_equal(active,np.asarray(label['event_active'],dtype=bool)):raise ValueError('Physical event labels disagree')
    if not len(label['box_valid'])==len(label['boxes_xyxy'])==n:raise ValueError('Oracle box support mismatch')
    spatial=np.ones((n,height,width),dtype=np.float64);used=[]
    x0=np.arange(width)/width;x1=(np.arange(width)+1)/width
    y0=np.arange(height)/height;y1=(np.arange(height)+1)/height
    for i,(valid,box) in enumerate(zip(label['box_valid'],label['boxes_xyxy'])):
        if not valid:continue
        box=np.asarray(box,dtype=np.float64)
        if box.shape!=(4,) or not np.isfinite(box).all() or np.any(box<0) or np.any(box>1) or np.any(box[2:]<=box[:2]):
            raise ValueError('Declared valid source box violates normalized geometry')
        dx=np.maximum(0,np.minimum(x1,box[2])-np.maximum(x0,box[0]))*width
        dy=np.maximum(0,np.minimum(y1,box[3])-np.maximum(y0,box[1]))*height
        spatial[i]=dy[:,None]*dx[None,:];used.append(i)
    event=np.broadcast_to(active[:,None,None],(n,height,width)).copy()
    return {'event':torch.tensor(event[None],dtype=torch.float32),'spatial':torch.tensor(spatial[None],dtype=torch.float32),
        'support':{'frame_ids':list(frame_ids),'THW':[n,height,width],'event_active':active.tolist(),
            'spatial_valid_positions':used,'spatial_neutral_missing_positions':[i for i in range(n) if i not in used],
            'spatial_uses_event_active':False,'spatial_mask':'normalized cell intersection area / cell area; missing GT box neutral1'}}

def feature_sha(x):
    a=x.detach().cpu().contiguous();return hashlib.sha256(a.view(torch.uint8).numpy().tobytes()).hexdigest()

@contextmanager
def privileged_branch_latents(adapter,masks,*,event=False,spatial=False,alpha=.25):
    """Weight d-channel post-LN/SiLU branch features just before out_proj.

    Gates, output bias, base PTD tokens, query pool, shared representation and
    all parameters remain untouched. Both branch forward paths are evaluated by
    the historical adapter; only the requested projection inputs are replaced.
    """
    if not 0<=alpha<=1:raise ValueError('alpha must be in [0,1]')
    handles=[];records=[]
    for branch,enabled in [('event',event),('spatial',spatial)]:
        if not enabled:continue
        def hook(module,args,branch=branch):
            z=args[0];m=masks[branch].to(device=z.device,dtype=z.dtype)
            if z.ndim!=5 or m.shape!=z.shape[:-1] or not torch.isfinite(m).all() or torch.any(m<0) or torch.any(m>1):
                raise ValueError('Privileged branch mask must match actual [B,T,H,W] support')
            weight=alpha+(1-alpha)*m
            changed=z*weight.unsqueeze(-1)
            records.append({'branch':branch,'shape':list(z.shape),'mask_sha':feature_sha(m),'weight_sha':feature_sha(weight),
                'original_feature_sha':feature_sha(z),'privileged_feature_sha':feature_sha(changed),
                'changed_elements':int((z!=changed).sum()),'total_elements':z.numel(),
                'relative_feature_delta':float((changed-z).float().norm()/z.float().norm().clamp_min(1e-12)),
                'weight_min':float(weight.min()),'weight_max':float(weight.max()),'weight_mean':float(weight.mean()),
                'position':'post reader / branch LayerNorm / SiLU; before frozen output projection and unchanged gate'})
            return (changed,*args[1:])
        handles.append(getattr(adapter,'out_proj_'+branch).register_forward_pre_hook(hook))
    try:yield records
    finally:
        for handle in reversed(handles):handle.remove()

def matched_wrong_masks(label,frame_ids,height,width):
    """Outcome-blind negative controls, with explicit non-discriminating cases.

    Temporal: same physical integer duration and observed foreground count;
    choose least observed overlap, then least physical overlap, then greatest
    displacement, then earliest begin. If no different support exists, keep it
    and mark ineligible. Spatial: preserve each box's size/valid-time support,
    translate to the farthest feasible corner (fixed corner order breaks ties).
    """
    import copy
    correct=masks_from_source_record(label,frame_ids,height,width)
    ids=np.asarray(frame_ids,dtype=np.int64);iv=label['event_interval'];start=int(iv['begin_fid']);end=int(iv['end_fid']);length=end-start
    if length<=0:raise ValueError('Nonpositive GT duration')
    active=np.asarray(label['event_active'],dtype=bool);candidates=[]
    for begin in range(int(ids[0]),int(ids[-1])+2-length):
        mask=(ids>=begin)&(ids<begin+length)
        if int(mask.sum())!=int(active.sum()) or np.array_equal(mask,active):continue
        overlap=max(0,min(end,begin+length)-max(start,begin))
        candidates.append(((int((mask&active).sum()),overlap,-abs(begin-start),begin),begin,mask))
    if candidates:
        _,begin,mask=min(candidates,key=lambda x:x[0]);eligible=True
    else:begin=start;mask=active.copy();eligible=False
    tlabel=copy.deepcopy(label);tlabel['event_interval']={'begin_fid':begin,'end_fid':begin+length};tlabel['event_active']=mask.tolist()
    wrong_t=masks_from_source_record(tlabel,frame_ids,height,width)
    slabel=copy.deepcopy(label);shifts=[]
    for i,(valid,box) in enumerate(zip(label['box_valid'],label['boxes_xyxy'])):
        if not valid:continue
        x,y,x2,y2=map(float,box);w=x2-x;h=y2-y
        corners=[(0.,0.),(1-w,0.),(0.,1-h),(1-w,1-h)]
        nx,ny=max(corners,key=lambda p:(p[0]-x)**2+(p[1]-y)**2)
        moved=[nx,ny,nx+w,ny+h];slabel['boxes_xyxy'][i]=moved
        inter=max(0,min(x2,moved[2])-max(x,nx))*max(0,min(y2,moved[3])-max(y,ny))
        shifts.append({'position':i,'from':list(map(float,box)),'to':moved,'area':w*h,'box_IoU':inter/max(2*w*h-inter,1e-15),
            'translation':[nx-x,ny-y],'changed':moved!=list(map(float,box))})
    wrong_s=masks_from_source_record(slabel,frame_ids,height,width)
    if not torch.equal(correct['event'].sum(),wrong_t['event'].sum()):raise AssertionError('Temporal total weight differs')
    if not torch.allclose(correct['spatial'].double().sum((-2,-1)),wrong_s['spatial'].double().sum((-2,-1)),atol=2e-5,rtol=1e-7):
        raise AssertionError('Translated box area differs on grid')
    return {'temporal':wrong_t,'spatial':wrong_s,'diagnostic':{
        'temporal':{'eligible_changed_support':eligible,'correct_interval':[start,end],'wrong_interval':[begin,begin+length],
            'same_physical_duration':length,'observed_count':int(active.sum()),'observed_overlap':int((active&mask).sum()),
            'physical_overlap':max(0,min(end,begin+length)-max(start,begin)),'selection':'minimum overlap, maximum displacement, earliest; no outcomes'},
        'spatial':{'eligible_changed_support':bool(shifts) and any(x['changed'] for x in shifts),'shifts':shifts,
            'same_box_area_and_valid_time_support':True,'selection':'farthest feasible corner, TL/TR/BL/BR tie order'}}}
