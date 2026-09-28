"""Fixed one-observation temporal / one-cell spatial context, no outcome input."""
import copy
import torch
import torch.nn.functional as F

def temporal_context(mask, frame_ids, interval):
    if mask.ndim != 4 or mask.shape[0] != 1 or mask.shape[1] != len(frame_ids):
        raise ValueError('Expected [1,T,H,W] observed support')
    ids=torch.as_tensor(frame_ids, dtype=torch.int64)
    start,end=interval
    if end<=start or not len(ids) or torch.any(ids[1:]<ids[:-1]):
        raise ValueError('Invalid physical interval/frame order')
    active=(ids>=start)&(ids<end)
    expected=active[None,:,None,None].expand_as(mask).to(mask)
    if not torch.equal(mask,expected):raise ValueError('Original physical support mismatch')
    before=torch.where(ids<start)[0];after=torch.where(ids>=end)[0]
    chosen=([int(before[-1])] if len(before) else [])+([int(after[0])] if len(after) else [])
    out=mask.clone()
    for pos in chosen:out[:,pos]=1
    return out,dict(interval=list(interval),added_positions=chosen,added_frame_ids=[int(ids[i]) for i in chosen],
        before_available=bool(len(before)),after_available=bool(len(after)),
        old_active_positions=torch.where(active)[0].tolist(),new_active_positions=torch.where(out[0,:,0,0]>0)[0].tolist())

def spatial_context(mask):
    if mask.ndim!=4 or mask.shape[0]!=1 or min(mask.shape[1:])<1 or not torch.isfinite(mask).all() or torch.any((mask<0)|(mask>1)):
        raise ValueError('Expected finite occupancy [1,T,H,W] in [0,1]')
    # Time is a batch dimension: Chebyshev radius1, fractional maximum, no temporal dilation.
    return F.max_pool2d(mask[0,:,None],kernel_size=3,stride=1,padding=1)[:,0][None]

def context_masks(correct,wrong,frame_ids):
    out={};diagnostic={}
    for branch,ob in [('event','temporal'),('spatial','spatial')]:
        pair={};details={}
        for which,old in [('correct',correct),('wrong',wrong[ob])]:
            if branch=='event':
                m,info=temporal_context(old[branch],frame_ids,wrong['diagnostic']['temporal'][which+'_interval'])
            else:
                m=spatial_context(old[branch]);info={}
                for p in old['support']['spatial_neutral_missing_positions']:
                    assert torch.equal(m[:,p],old[branch][:,p])
            assert torch.all(m>=old[branch])
            pair[which]={**old,branch:m}
            details[which]={**info,'old_weight':float(old[branch].double().sum()),'new_weight':float(m.double().sum()),
                'old_positive_cells':int((old[branch]>0).sum()),'new_positive_cells':int((m>0).sum()),
                'neutral':bool(torch.all(m==1)),'changed_cells':int((m!=old[branch]).sum())}
        distinct=not torch.equal(pair['correct'][branch],pair['wrong'][branch])
        both=not details['correct']['neutral'] and not details['wrong']['neutral']
        diagnostic[ob]=dict(original_eligible=wrong['diagnostic'][ob]['eligible_changed_support'],
            support_distinct=distinct,both_non_neutral=both,eligible_changed_support=distinct and both,
            correct=details['correct'],wrong=details['wrong'],
            same_expansion_rule=True,equal_expanded_weight=details['correct']['new_weight']==details['wrong']['new_weight'],
            limitation='Boundary clipping may change matched weight/area; no shifted control reselection. Norm matched only for nonzero directions.')
        out[branch]=pair
    return {'masks':out,'diagnostic':diagnostic}
