"""Episodic native-probability STVG ports of Tent/MEMO/SAR.

The objective uses both endpoints and final actionness, not DeCoTA supervision.
Compatibility deviations: decoder LN affine scope, eval-mode dropout, full
post-update prediction, episodic reset. SAR's video-level gate is scaled by the
declared native-output maximum entropy. These are STVG ports, not original-task
reproductions. ViTTA is deliberately not implemented as another entropy method.
"""
import copy
import math
import numpy as np
import torch

from vg_tta.native_probability_interface_v1 import (
    NativeOutput, offsets_entropy, memo_native_marginal, entropy_ceiling,
)

DEFAULTS = dict(
    Tent=dict(lr=1e-3, steps=1),
    MEMO=dict(lr=5e-3, steps=1, views=2),
    SAR=dict(lr=1e-3, steps=3, rho=.05, margin_fraction=.4, reset_entropy=.2),
)


def _restore(params, values):
    with torch.no_grad():
        for p,v in zip(params,values): p.copy_(v)


def _norm(values):
    vals=[v.float().square().sum() for v in values if v is not None]
    return torch.stack(vals).sum().sqrt() if vals else torch.tensor(0.)


@torch.enable_grad()
def adapt(params, closure, final_inference, method, config):
    """closure(view)->independent original offsets, with live parameter gradients."""
    if method not in DEFAULTS or not params: raise ValueError('Unsupported method/scope')
    cfg={**DEFAULTS[method],**config}
    if cfg['lr']<0 or cfg['steps']<0: raise ValueError('Invalid optimizer budget')
    initial=[p.detach().clone() for p in params]; flags=[p.requires_grad for p in params]
    for p in params: p.requires_grad_(True)
    opt=(torch.optim.Adam(params,lr=cfg['lr']) if method=='Tent' else
         torch.optim.SGD(params,lr=cfg['lr'],momentum=.9 if method=='SAR' else 0.,weight_decay=0.))
    opt0=copy.deepcopy(opt.state_dict()); ema=None
    audit=dict(method=method,config=cfg,backwards=0,optimizer_steps=0,skipped_steps=0,recoveries=0,
               forward_calls=0,gradient_norms=[],unused_parameter_indices=[],trace=[],GT_online=False)

    def outputs(view=0):
        audit['forward_calls']+=1
        return closure(view)

    def grad(loss):
        if not loss.requires_grad or not bool(torch.isfinite(loss)):
            raise RuntimeError('Invalid live adaptation loss')
        loss.backward(); audit['backwards']+=1
        g=_norm([p.grad for p in params])
        if not bool(torch.isfinite(g)): raise RuntimeError('Nonfinite baseline gradient')
        audit['gradient_norms'].append(float(g))
        audit['unused_parameter_indices'].append([i for i,p in enumerate(params) if p.grad is None])
        return g

    try:
        for step in range(cfg['steps']):
            opt.zero_grad(set_to_none=True)
            if method=='MEMO':
                views=[outputs(v) for v in range(cfg['views'])]
                if len({len(v) for v in views})!=1: raise ValueError('Offset count changed across MEMO views')
                loss=torch.stack([memo_native_marginal(list(v)) for v in zip(*views)]).mean()
            else:
                native=outputs(); loss=offsets_entropy(native)
            record=dict(step=step,loss=float(loss.detach()),updated=False)
            if method!='SAR':
                grad(loss); opt.step(); audit['optimizer_steps']+=1; record['updated']=True
            else:
                ceiling=sum(entropy_ceiling(o) for o in native)/len(native)
                margin=cfg['margin_fraction']*ceiling
                record.update(ceiling=ceiling,margin=margin,first_reliable=record['loss']<margin)
                if not record['first_reliable']:
                    audit['skipped_steps']+=1; audit['trace'].append(record); continue
                magnitude=grad(loss)
                base=[p.detach().clone() for p in params]
                try:
                    with torch.no_grad():
                        for p in params:
                            if p.grad is not None: p.add_(p.grad*(cfg['rho']/(magnitude+1e-12)))
                    opt.zero_grad(set_to_none=True)
                    second=offsets_entropy(outputs())
                    record.update(second_loss=float(second.detach()),second_reliable=float(second.detach())<margin)
                    if record['second_reliable']:
                        grad(second)
                        ema=record['second_loss'] if ema is None else .9*ema+.1*record['second_loss']
                    else:
                        opt.zero_grad(set_to_none=True); audit['skipped_steps']+=1
                finally:
                    # Exact saved parameter restoration, not subtracting a
                    # floating perturbation; second-pass gradient is retained.
                    _restore(params,base)
                if record['second_reliable']:
                    opt.step(); audit['optimizer_steps']+=1; record['updated']=True
                    if ema<cfg['reset_entropy']:
                        _restore(params,initial); opt.load_state_dict(copy.deepcopy(opt0))
                        ema=None; audit['recoveries']+=1; record['recovered']=True
                record['ema']=ema
            audit['trace'].append(record)
        audit['parameter_delta_l2']=float(_norm([p.detach()-v for p,v in zip(params,initial)]))
        audit['parameter_changed']=any(not torch.equal(p,v) for p,v in zip(params,initial))
        with torch.no_grad(): final=final_inference()
        return final,audit
    finally:
        _restore(params,initial)
        for p,flag in zip(params,flags): p.grad=None; p.requires_grad_(flag)
        audit['source_restored']=all(torch.equal(p,v) for p,v in zip(params,initial))
        if not audit['source_restored']: raise RuntimeError('Baseline reset failed')


def photometric_view(frames, view):
    """Two fixed clip-wide photometric views; no geometry/time remapping."""
    if view not in (0,1): raise ValueError('Only the locked two-view port')
    scale=.9 if view==0 else 1.1
    return np.ascontiguousarray(np.clip(np.rint(frames.astype(np.float32)*scale),0,255).astype(np.uint8))


def live_output(model,batch,ids,subject):
    from methods.decota_final_simplified_v1.backbone import query_subject,inserted_state,offset_batch
    from methods.decota_final_simplified_v1.objectives import prediction
    native,boxes,logits,records=[],[],[],[]
    with query_subject(model,batch,subject), inserted_state(model,{}):
        for offset in (0,1):
            view=offset_batch(batch,offset)
            with torch.autocast('cuda',dtype=torch.float16):
                out=model(view['videos'],view['texts'],view['targets'],iteration_rate=-1)
            f=tuple(view['targets'][0]['frame_ids']); z=out['pred_sted'].reshape(-1,2)
            native.append(NativeOutput(z,out['pred_actioness'].reshape(-1),
                                       torch.ones(len(f),device=z.device,dtype=torch.bool),f))
            boxes.append(out['pred_boxes']);logits.append(out['pred_sted'])
            records.append(dict(offset=offset,flip=False,frame_ids=list(f)))
    merged=torch.stack([boxes[i%2][i//2] for i in range(len(ids))]).float()
    return native,prediction(logits,merged,records,ids)


def episode(model,parser,frames,ids,metadata,method,config=None):
    from methods.decota_final_simplified_v1.backbone import make_batch,validate_input
    from methods.decota_final_simplified_v1.tensors import state_hash
    from vg_tta.tastvg_baseline_expansion import decoder_layernorm_scope
    validate_input(frames,ids,metadata)
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Expected frozen eval source model')
    before=state_hash(model.state_dict())
    names,params,scope=decoder_layernorm_scope(model)
    batch=make_batch(frames,ids,metadata,model); subject=parser(metadata['caption'])['subject']
    with torch.no_grad(): _,frozen=live_output(model,batch,ids,subject)
    views=([make_batch(photometric_view(frames,v),ids,metadata,model) for v in (0,1)]
           if method=='MEMO' else [batch])
    final,audit=adapt(params,lambda v:live_output(model,views[v],ids,subject)[0],
                      lambda:live_output(model,batch,ids,subject)[1],method,config or {})
    if before!=state_hash(model.state_dict()): raise RuntimeError('Full model state not restored')
    audit.update(parameter_names=names,parameter_count=sum(p.numel() for p in params),scope=scope,
                 source_state_hash=before,expert_calls=0,
                 compatibility=['decoder_LN_affine','eval_dropout','episodic','native_endpoint_plus_actionness',
                                'post_update_forward'],photometric_scales=[.9,1.1] if method=='MEMO' else None)
    return dict(Frozen=frozen,prediction=final,audit=audit)
