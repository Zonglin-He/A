"""Reinsert an episodic state into the original full backbone; restore source.

Imported by the finite worker. No labels, selection, persisted source mutation.
"""
import copy
import torch
from vg_tta.closure_replay_v1 import floating32
from vg_tta.shared_state_v1 import capture_shared
from vg_tta.decota_tastvg_episode_v1 import make_batch
from vg_tta.foreground_runtime import state_digest


def full_prediction(model,frames,ids,metadata,subject,state,expected):
    before=state_digest(model)
    norm=model.ground_encoder.encoder.norm;space=model.ground_decoder.decoder;head=model.temp_embed
    saved=[copy.deepcopy(m.state_dict()) for m in (norm,space,head)]
    sn,ss,sh=[copy.deepcopy(s) for s in saved]
    delta=torch.zeros(256,device='cuda')
    for n,z in state.items():
        if n.startswith('shared.'):sn[n[len('shared.'):]]=z
        elif n.startswith('head.'):sh[n[len('head.'):]]=z
        elif n=='spatial.query_residual':delta=z.to('cuda')
        elif n.startswith('spatial.layers.'):ss[n[len('spatial.'):]]=z
        else:raise ValueError(n)
    norm.load_state_dict(sn);space.load_state_dict(ss);head.load_state_dict(sh)
    contexts=[];calls=[0]
    def norm_before(mod,args):
        ctx=torch.autocast('cuda',enabled=False);ctx.__enter__();contexts.append(ctx)
        return floating32(args)
    def ground_before(mod,args,kw):return floating32(args),floating32(kw)
    def query_before(mod,args,kw):
        calls[0]+=1
        if calls[0]%2==0:
            kw=dict(kw);kw['query_tgt']=kw['query_tgt']+delta[None,None,:]
        return args,kw
    def model_after(mod,args,out):contexts.pop().__exit__(None,None,None)
    hooks=[norm.register_forward_pre_hook(norm_before),
        model.ground_decoder.register_forward_pre_hook(ground_before,with_kwargs=True),
        space.register_forward_pre_hook(query_before,with_kwargs=True),model.register_forward_hook(model_after)]
    try:
        batch=make_batch(frames,ids,metadata,subject,model)
        base,_,_,_,_,views=capture_shared(model,batch)
        boxes=base['raw_boxes'].float().cpu();zs=[v['output']['pred_sted'].cpu() for v in views]
        bd=float((boxes-expected['boxes']).abs().max());zd=max(float((z-q).abs().max()) for z,q in zip(zs,expected['logits'][:2]))
        assert bd==0 and zd==0,('full_forward_reinsertion_mismatch',bd,zd)
        assert list(base['predicted_indices'])==expected['indices']
        out=dict(boxes=boxes,indices=list(base['predicted_indices']),logits=zs,
                 audit=dict(box_max_error=bd,logit_max_error=zd,full_model=True,source_restored=True,GT_online=False))
    finally:
        for h in hooks:h.remove()
        while contexts:contexts.pop().__exit__(None,None,None)
        for m,s in zip((norm,space,head),saved):m.load_state_dict(s)
    assert state_digest(model)==before
    return out
