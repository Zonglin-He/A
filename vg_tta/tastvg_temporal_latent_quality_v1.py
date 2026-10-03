"""Observe the final temporal decoder hidden input; never change model outputs."""
import torch


@torch.no_grad()
def observe(actor, data, with_candidates=False):
    hidden=[];headed=[]
    def capture(module, args, out):
        x=args[0]
        assert x.shape[0]==6 and x.shape[-1]==256
        assert out.shape[:-1]==x.shape[:-1] and out.shape[-1]==2
        hidden.append(x.detach().cpu().clone());headed.append(out.detach().cpu().clone())
    hook=actor.model.temp_embed.register_forward_hook(capture)
    try:
        if with_candidates:
            from vg_tta.tastvg_selected_rollout_v1 import central_with_candidates
            _,_,p,_,support=central_with_candidates(actor,data)
        else:
            _,_,p=actor.values(data);support=None
    finally:hook.remove()
    assert len(hidden)==len(headed)==2
    for h,z,logits in zip(hidden,headed,p['logits']):
        direct=actor.model.temp_embed(h.to(next(actor.model.parameters()).device)).cpu()
        assert torch.equal(direct,z), 'Hidden state must reproduce the exact start/end head'
        assert torch.equal(z[-1],logits)
    ids=data['frame_ids']; merged=torch.empty(len(ids),256);seen=set()
    for h,r in zip(hidden,data['records']):
        part=h[-1].reshape(-1,256)
        assert len(part)==len(r['frame_ids'])
        for v,f in zip(part,r['frame_ids']):
            i=ids.index(f);assert i not in seen;seen.add(i);merged[i]=v
    assert seen==set(range(len(ids))) and torch.isfinite(merged).all()
    return merged, p, support
