"""Explicit per-video class×time activity latent, not pretrained-weight TTA."""
import copy,json,torch,numpy as np
from pycocotools import mask as mu
from scipy.ndimage import median_filter

def evidence(cosine,margin=.04,temperature=.02):
    # Independent multi-label relative evidence; not softmax/probability calibration.
    cls=cosine[:,:26];bg=cosine[:,26:].max(-1,keepdims=True)
    reference=np.maximum(bg,cls.max(-1,keepdims=True)-margin)
    z=np.clip((cls-reference)/temperature,-40,40)
    return (1/(1+np.exp(-z))).astype('float32')

def visibility(predictions,video_id,n_frames,nclass=26):
    out=np.zeros((n_frames,nclass),dtype=bool)
    for track in predictions:
        if track['video_id']!=video_id:continue
        cid=track['category_id']-1
        for t,m in enumerate(track['segmentations']):
            if m is not None and int(mu.area(m))>0:out[t,cid]=True
    return out

def optimize(q,valid,temporal=True,steps=20,lr=.5):
    q=torch.as_tensor(q,dtype=torch.float32);valid=torch.as_tensor(valid,dtype=torch.bool)
    residual=torch.zeros_like(q,requires_grad=True);records=[]
    if not valid.any():return np.ones(q.shape,bool),{'skip':'no_source_visible_candidates','steps':0,'residual_l2':0.,'GT_access':False}
    opt=torch.optim.Adam([residual],lr=lr);weight=(2*q-1).abs().detach();edges=valid[1:]&valid[:-1]
    for step in range(steps):
        opt.zero_grad(set_to_none=True);logits=3+residual;p=logits.sigmoid()
        audio=(torch.nn.functional.binary_cross_entropy_with_logits(logits,q,reduction='none')*weight)[valid].mean()
        tv=(p[1:]-p[:-1]).abs()[edges].mean() if temporal and edges.any() else residual.sum()*0
        anchor=residual.square()[valid].mean();loss=audio+.1*tv+.01*anchor
        loss.backward();grad=torch.nn.utils.clip_grad_norm_([residual],1.,error_if_nonfinite=True);opt.step()
        with torch.no_grad():residual.clamp_(-8,8)
        assert residual.isfinite().all();records.append({'total':float(loss.detach()),'audio_BCE_weighted':float(audio.detach()),'temporal_TV':float(tv.detach()),'source_anchor':float(anchor.detach()),'gradient_norm':float(grad)})
    with torch.no_grad():active=(3+residual).sigmoid()>=.5
    return active.numpy(),{'steps':steps,'optimizer':'Adam','lr':lr,'base_logit':3,'residual_l2':float(residual.detach().norm()),'residual_max_abs':float(residual.detach().abs().max()),'valid_class_frames':int(valid.sum()),'suppressed_class_frames':int((valid&~active).sum()),'temporal_enabled':temporal,'losses':records,'GT_access':False}

def apply_gates(predictions,video_id,gates):
    rows=[];suppressed=0;dropped=0
    for original in predictions:
        if original['video_id']!=video_id:continue
        track=copy.deepcopy(original);any_nonempty=False
        for t,seg in enumerate(track['segmentations']):
            if seg is None:continue
            if not bool(gates[t,track['category_id']-1]):
                if int(mu.area(seg))>0:suppressed+=1
                h,w=seg['size'];r=mu.encode(np.asfortranarray(np.zeros((h,w),dtype='uint8')));r['counts']=r['counts'].decode();track['segmentations'][t]=r
            elif int(mu.area(seg))>0:any_nonempty=True
        if any_nonempty:rows.append(track)
        else:dropped+=1
    return rows,{'suppressed_instance_frames':suppressed,'fully_inactive_tracks_dropped':dropped,'confidence':'source globaltrack scores unchanged; same policy for allteacherarms'}

def direct(q,smooth=False):return (median_filter(q,size=(5,1),mode='nearest') if smooth else q)>=.5
