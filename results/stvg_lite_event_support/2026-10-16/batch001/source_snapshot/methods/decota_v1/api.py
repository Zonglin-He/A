from __future__ import annotations
import copy
import numpy as np
import torch
from vg_tta.metrics import interval_from_logits
from vg_tta.native_coverage_calibration_v1 import native_at_length
VERSION='decota_v1'

def decode(native_logits,adapted_indices,frame_ids,*,native_indices=None):
    """No labels. Inclusive sampled endpoints encode [id[s],id[e]+1).

    Preserve the native prediction when the estimated extent is unchanged.
    This is automatic for a single native argmax (TubeDETR), and explicit for
    TA-STVG's official two-offset envelope, which is not an interleaved argmax.
    """
    ids=list(map(int,frame_ids));a,b=map(int,adapted_indices)
    if not 0<=a<=b<len(ids) or (len(ids)>1 and a==b):raise ValueError('Invalid native interval')
    if native_indices is None:native_indices=interval_from_logits(torch.as_tensor(native_logits).reshape(1,len(ids),2))
    i,j=map(int,native_indices)
    wanted=ids[b]+1-ids[a]
    if wanted==ids[j]+1-ids[i]:
        return {'indices':(i,j),'physical_span':(ids[i],ids[j]+1),'length':wanted,'unchanged_extent_preserves_native':True}
    r=native_at_length(native_logits,ids,reference=(a,b));s,e=r['indices']
    assert r['length_mismatch']==0
    return {'indices':(s,e),'physical_span':(ids[s],ids[e]+1),'length':wanted,'unchanged_extent_preserves_native':False,'length_audit':r}

def fit(source_head,inputs,*,backbone,lr,steps,gamma=0.,eps=1e-4):
    """Fit a private native head; original module is never mutated."""
    head=copy.deepcopy(source_head).eval();device=next(head.parameters()).device
    if backbone=='tubedetr':
        if len(inputs)!=1:raise ValueError('TubeDETR uses one cached feature sequence')
        from ._fullspan import fit_fullspan_head,replay_temporal_head
        audit=fit_fullspan_head(head,[{'head_input':inputs[0]} for _ in range(int(steps))],lr,anchor_gamma=gamma,optimizer_eps=eps)
        with torch.no_grad():zs=[replay_temporal_head(head,inputs[0],device)]
    elif backbone=='tastvg':
        from vg_tta.tastvg_fullspan_multistep import fit_tastvg_fullspan_multistep
        from vg_tta.tastvg_baseline_expansion import replay_temporal_head_native
        audit=fit_tastvg_fullspan_multistep(head,inputs,lr=lr,steps=steps,anchor_gamma=gamma,optimizer_eps=eps)
        with torch.no_grad():zs=[replay_temporal_head_native(head,h,device) for h in inputs]
    else:raise ValueError(backbone)
    return head,[z.detach() for z in zs],audit
