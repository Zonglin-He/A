"""Frozen pre-adaptation position selection and true-image target ablations. No I/O/GT."""
import copy, math
import torch
from vg_tta.ptd_spatial_adapter_ab_v1 import Adapter
from vg_tta.ptd_dense_opd_v1 import reverse_kl, vector

ARMS=['S1','S2','P-Uniform','P-Disagreement','P-Visual']

def design(z,qp,qm):
    qp,qm=qp.double(),qm.double();p=z['logits'].double().log_softmax(-1)
    n=len(p);m=max(1,math.ceil(n*.25));shift=max(1,n//2)
    assert n>=2
    permutation=(torch.arange(n)+shift)%n
    qshift=(qm+(qp-qm)[permutation]).log_softmax(-1)
    d=reverse_kl(p,qp).mean(-1);v=reverse_kl(qp,qm).mean(-1)
    uniform=[int((j+.5)*n/m) for j in range(m)]
    def top(x):return sorted(torch.argsort(x,descending=True,stable=True)[:m].tolist())
    masks={'S1':list(range(n)),'S2':list(range(n)),'P-Uniform':uniform,'P-Disagreement':top(d),'P-Visual':top(v)}
    assert all(len(masks[a])==m for a in ARMS[2:]) and len(set(uniform))==m
    return dict(indices=masks,disagreement=d.tolist(),visual=v.tolist(),permutation=permutation.tolist(),shift=shift,
        shifted_kl=reverse_kl(qp,qshift).mean(-1).tolist(),shifted_total_variation=(qp.exp()-qshift.exp()).abs().sum(-1).mean(-1).div(2).tolist(),
        shifted_argmax_changed=int((qp.argmax(-1)!=qshift.argmax(-1)).sum()),qshift=qshift,GT_used=False)

def fit(h,base,target,selected,*,seed=20260924,model=None):
    model=Adapter(h.shape[-1]) if model is None else model
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=0.)
    params=list(model.parameters());target=target.float().detach();selected=torch.as_tensor(selected)
    assert not opt.state and torch.equal(base+model(h),base)
    history=[];updates=[]
    for step in range(4):
        lp=(base+model(h)).log_softmax(-1)
        loss=reverse_kl(lp,target)[selected].mean()
        gen=torch.Generator(device=h.device).manual_seed(seed+step)
        acts=torch.multinomial(lp.detach().exp().reshape(-1,lp.shape[-1]),1,generator=gen).reshape(lp.shape[:-1])
        history.append(dict(step=step,state={k:v.detach().clone() for k,v in model.state_dict().items()},optimizer=copy.deepcopy(opt.state_dict()),
            loss=float(loss.detach()),logp=lp.detach().clone(),tokens=lp.argmax(-1).detach(),sampled_actions=acts,
            sampled_logp=lp.detach().gather(-1,acts[...,None]).squeeze(-1),seed=seed+step))
        if step==3:break
        opt.zero_grad(set_to_none=True);loss.backward()
        grad=vector([p.grad.detach().clone() for p in params]);before=vector([p.detach().clone() for p in params])
        norm=float(torch.nn.utils.clip_grad_norm_(params,1.));opt.step()
        displacement=vector([p.detach() for p in params])-before
        assert torch.isfinite(displacement).all() and displacement.norm()>0
        updates.append(dict(step=step+1,gradient=grad,preclip_norm=norm,displacement=displacement.clone(),displacement_norm=float(displacement.norm())))
    restored=copy.deepcopy(model);restored.load_state_dict(history[-1]['state']);assert torch.equal(restored(h),model(h))
    model.load_state_dict(history[0]['state']);assert torch.equal(base+model(h),base)
    return dict(history=history,updates=updates,selected=selected.tolist(),parameter_count=sum(p.numel() for p in params),
        optimizer_initial_empty=True,reload_exact=True,reset_exact=True,GT_used=False)
