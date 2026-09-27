"""Source-only evidence objectives and fixed optimizer recipe for v2.

No dataset I/O, target adaptation, sampling gate, or checkpoint selection.
"""
from __future__ import annotations
import math
import torch
from torch.nn import functional as F

def source_evidence_losses(outputs, record):
    from scripts.desta3d_source_fit_v1 import aux_targets, masked_bce
    ref=outputs['referent_logits'];event=outputs['event_logits']
    targets=aux_targets(record,ref.shape[-2],ref.shape[-1],ref.device)
    # Event annotation is temporal; missing spatial boxes must not erase it.
    event_target=torch.tensor(record['event_active'],device=event.device,dtype=event.dtype)[None]
    assert event.shape==event_target.shape and ref.shape[1]==len(record['frame_ids'])
    return {'ref':masked_bce(ref,targets['referent_target'],targets['referent_mask']),
            'event':F.binary_cross_entropy_with_logits(event.float(),event_target.float())}

def make_source_optimizer(adapter, recipe, stage):
    adapter.set_train_stage('A evidence' if stage=='A' else 'B integration')
    if recipe=='current':
        assert stage=='B'
        return torch.optim.AdamW([p for p in adapter.parameters() if p.requires_grad],lr=1e-4,weight_decay=0.)
    assert recipe=='repaired'
    head_groups={'out_projection','readout_heads','gates'}
    groups=[]
    for name,params in adapter.parameter_groups().items():
        params=[p for p in params if p.requires_grad]
        if params:
            rate=1e-4 if name in head_groups else 3e-5
            groups.append({'params':params,'lr':rate,'base_lr':rate,'name':name})
    return torch.optim.AdamW(groups,weight_decay=0.)

def warmup_cosine(step,total,warmup_fraction=.05):
    if total<1 or not 0<=step<total:raise ValueError((step,total))
    warm=max(1,math.ceil(total*warmup_fraction))
    if step<warm:return (step+1)/warm
    # Nonnegative decay, with last scheduled update at zero for total>warm+1.
    return .5*(1+math.cos(math.pi*(step-warm)/max(1,total-warm-1)))

def set_source_lr(optimizer,step,total,recipe):
    scale=warmup_cosine(step,total) if recipe=='repaired' else 1.
    for group in optimizer.param_groups:
        if recipe=='repaired':group['lr']=group['base_lr']*scale
    return scale
