"""Experimental decision-only additions. No labels, no production mutation."""
import copy
import math
import re
import numpy as np
import torch
from scipy.special import expit, logit


def candidate_path(path):
    losses=[float(s['loss']) for s in path]
    if not losses or not all(map(math.isfinite,losses)): raise ValueError('Nonfinite optimization trajectory')
    best=min(range(len(losses)),key=lambda k:(losses[k],k))
    eligible=[k for k in range(len(losses)) if k==0 or losses[k]<min(losses[:k])]
    # Default first guarantees that equal intervals / equal utilities keep B.
    candidates=[];seen=set()
    for k in [best]+eligible:
        ij=tuple(path[k]['prediction']['indices'])
        if ij not in seen: candidates.append(dict(step=k,indices=list(ij)));seen.add(ij)
    return candidates,dict(best_step=best,eligible_steps=eligible,losses=losses)


def weights(ids,start,end):
    ids=np.asarray(ids,float);w=np.diff(np.r_[start,(ids[:-1]+ids[1:])/2,end])
    if len(w)!=len(ids) or not np.isfinite(w).all() or (w<=0).any():raise ValueError('Invalid physical grid')
    return w/w.sum()


def select(candidates, probability, w, c=0.):
    """Soft temporal IoU; first candidate is default. Never consumes GT."""
    if not candidates:raise ValueError('Empty candidate set')
    p=np.asarray(probability,float);w=np.asarray(w,float)
    if p.shape!=w.shape or not np.isfinite(p).all() or (p<0).any() or (p>1).any():
        return {**candidates[0], 'reason':'invalid_signal_keep_default','scores':None,'selected':0}
    p=expit(logit(np.clip(p,1e-6,1-1e-6))+float(c));scores=[]
    for v in candidates:
        s,e=v['indices'];inside=float(np.dot(w[s:e+1],p[s:e+1]));den=float(w[s:e+1].sum()+w@p-inside)
        scores.append(inside/den if den>0 else 0.)
    k=0
    for j in range(1,len(scores)):
        if scores[j]>scores[k]+1e-12:k=j
    return {**candidates[k], 'reason':'readout_improved' if k else 'keep_default','scores':scores,'selected':k}


def repair_query(parser,caption):
    """Only add an explicit declarative entity when old parser returned empty."""
    from vg_tta.tg_spatial_tta_v1 import visual_query
    from vg_tta.decota_cal_spatial_v1 import parse_query
    old=visual_query(parser,caption);new=dict(old);detail=parse_query(parser,caption)
    if not old['phrase'] and detail['reason']=='declarative_explicit_entity_fallback':
        new.update(phrase=detail['phrase'],entity=detail['entity'],rule='explicit_entity_fallback',distinctive=True)
    return new,dict(original=old,repaired=new!=old,detail=detail)


def transition_prompt(parsed,caption):
    """Narrow lexical probe, not general event understanding. Others unchanged."""
    phrase=parsed['phrase'];low=caption.lower();remove=None
    if re.search(r'\b(seated|sitting)\b',phrase) and re.search(r'\b(stands? up|gets? up|rises?)\b',low):
        remove=r'\b(seated|sitting)\b'
    elif re.search(r'\bstanding\b',phrase) and re.search(r'\b(sits? down|sits? on)\b',low):
        remove=r'\bstanding\b'
    if remove: return {**parsed,'phrase':' '.join(re.sub(remove,'',phrase).split())}
    return dict(parsed)


@torch.no_grad()
def observe(expert,rgb,parsed):
    """Exactly production detection scores/gate, plus top-3 pure visual ROI."""
    from torchvision.ops import nms,box_convert
    from vg_tta.anchor_appearance_probe_v1 import visual_roi_features
    captured=[]
    h=expert.model.model.backbone.conv_encoder.register_forward_hook(lambda m,a,o:captured.append(o))
    try:z=expert(rgb,parsed['phrase'],parsed['entity'])
    finally:h.remove()
    assert len(captured)==1
    xy=box_convert(z['all_boxes'].float(),'cxcywh','xyxy').clamp(0,1)
    keep=nms(xy,z['all_phrase_scores'],.5)[:3]
    keep=keep[(xy[keep,2:]>xy[keep,:2]).all(1)]
    boxes=box_convert(xy[keep],'xyxy','cxcywh')
    features,shapes=visual_roi_features(captured[0],boxes.cuda())
    score=z['all_phrase_scores'][keep]
    if z['accepted']:assert torch.equal(boxes[0],torch.tensor(z['box']))
    return dict(boxes=boxes,unary=score,reference=score.clone(),roi=features,
        candidate_ids=keep.tolist(),accepted=z['accepted'],reason=z['reason'],margin=z['margin'],
        text=z['text'],detection=z,roi_shapes=shapes,GT_online=False)


def fixed_associate(probes,weight=0.,reference=True,wrong_seed=None):
    """Every accepted time contributes exactly one observation; exact chain DP."""
    support=[i for i,p in enumerate(probes) if p['accepted'] and len(p['boxes'])]
    if not support:return [],dict(support=[],reference=None,weight=weight,objective=0.)
    ref=(support[0],0);dp=None;paths=None;rng=np.random.default_rng(wrong_seed)
    previous=None
    for i in support:
        p=probes[i];f=np.asarray(p['roi'],float);f=f/np.maximum(np.linalg.norm(f,axis=1,keepdims=True),1e-12)
        if wrong_seed is not None and len(f)>1:
            # Nonidentity rotation, unlike random permutations that may do nothing.
            f=np.roll(f,int(rng.integers(1,len(f))),axis=0)
        u=np.log(np.maximum(np.asarray(p['unary'],float),1e-30)/.35)
        if reference and i==ref[0]:u[1:]=-np.inf
        if dp is None:dp=u;paths=[[(i,j)] for j in range(len(u))]
        else:
            edge=weight*np.log(np.maximum((1+np.clip(previous@f.T,-1,1))/2,1e-12))
            joint=dp[:,None]+edge+u[None,:];parent=np.argmax(joint,axis=0)
            dp=joint[parent,np.arange(len(u))];paths=[paths[k]+[(i,j)] for j,k in enumerate(parent)]
        previous=f
    best=int(np.argmax(dp));path=paths[best]
    assert [i for i,j in path]==support
    assert not reference or ref in path
    return path,dict(support=support,reference=list(ref) if reference else None,weight=weight,
        objective=float(dp[best]),anchors=len(path),wrong_seed=wrong_seed,skipped_accepted=0)


def replay_path(source_head,inputs,lr):
    from vg_tta.temporal_optimizer_probe_v1 import snapshot,update,cpu_state
    source=cpu_state(source_head);head=copy.deepcopy(source_head).eval().requires_grad_(True)
    opt=torch.optim.AdamW(head.parameters(),lr=lr,eps=1e-4,weight_decay=0.)
    path=[snapshot(head,inputs)]
    for _ in range(5):path.append(update(head,inputs,opt))
    assert all(torch.equal(v,cpu_state(source_head)[k]) for k,v in source.items())
    return path
