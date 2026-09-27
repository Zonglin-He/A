"""Exact coordinate-policy OPD with episodic adapter/optimizer state. No data I/O."""
import copy
import math
import torch
from .ptd_spatial_adapter_ab_v1 import Adapter


def reverse_kl(lp, lq):
    return (lp.exp() * (lp-lq)).sum(-1)


def vector(parts):
    return torch.cat([p.reshape(-1) for p in parts])


def angle(a, b):
    na, nb = float(a.norm()), float(b.norm())
    if na == 0 or nb == 0:
        return dict(cosine=None, degrees=None, undefined='zero_gradient')
    c = max(-1., min(1., float(torch.dot(a, b)/(a.norm()*b.norm()))))
    return dict(cosine=c, degrees=math.degrees(math.acos(c)))


def fit(plus, minus, qplus, qminus, arm, seed=20260924, steps=3):
    assert arm in ['D1', 'D2', 'D3']
    h, hm = plus['h'].clone(), minus['h'].clone()
    base, bm = plus['logits'].clone(), minus['logits'].clone()
    qp, qm = qplus.float().detach(), qminus.float().detach()
    assert qp.shape == qm.shape == base.shape == bm.shape
    assert torch.allclose(qp.logsumexp(-1), torch.zeros_like(qp[..., 0]), atol=2e-6)
    model = Adapter(h.shape[-1], seed=20260924).to(h.device)
    opt = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=0.)
    assert len(opt.state) == 0 and torch.equal(base+model(h), base)
    params = list(model.parameters())
    vds = reverse_kl(qp, qm).flatten()
    order = torch.argsort(vds, descending=True, stable=True)
    mask = torch.zeros_like(vds, dtype=torch.bool)
    mask[order[:math.ceil(.3*len(order))]] = True
    history, updates = [], []
    for step in range(steps+1):
        lp = (base+model(h)).log_softmax(-1)
        lm = (bm+model(hm)).log_softmax(-1)
        qv = (lm.detach()+qp-qm).log_softmax(-1).detach()
        std = reverse_kl(lp, qp).mean()
        vis = reverse_kl(lp, qv).mean()
        prior = reverse_kl(lm, qm).flatten()[mask].mean()
        gen = torch.Generator(device=h.device).manual_seed(seed+step)
        action = torch.multinomial(lp.detach().exp().reshape(-1,1001), 1, generator=gen).reshape(lp.shape[:-1])
        gather = lambda z: z.detach().gather(-1,action[...,None]).squeeze(-1).cpu()
        history.append(dict(step=step, state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
            optimizer=copy.deepcopy(opt.state_dict()), tokens=lp.argmax(-1).detach().cpu(),
            losses=dict(standard=float(std.detach()),visual=float(vis.detach()),prior=float(prior.detach())),
            sampled_actions=action.cpu(), sampled_logp=gather(lp), sampled_logq=gather(qp),
            sampled_logqvis=gather(qv), logp=lp.detach().cpu(), logp_minus=lm.detach().cpu(),
            qvis=qv.cpu(), seed=seed+step))
        if step == steps:
            break
        gs = torch.autograd.grad(std, params, retain_graph=True)
        gv = torch.autograd.grad(vis, params, retain_graph=True)
        gp = torch.autograd.grad(prior, params)
        vs, vv, vp = vector(gs), vector(gv), vector(gp)
        raw = vs if arm == 'D1' else vs+vv
        if arm == 'D3':
            raw = raw+.01*vp
        scale = 1. if arm == 'D1' else float(vs.norm()/raw.norm()) if raw.norm() else 0.
        total = raw*scale
        assert torch.isfinite(total).all()
        before = vector([p.detach().clone() for p in params])
        opt.zero_grad(set_to_none=True)
        offset = 0
        for p in params:
            p.grad = total[offset:offset+p.numel()].reshape_as(p).clone()
            offset += p.numel()
        clip_norm = float(torch.nn.utils.clip_grad_norm_(params,1.))
        opt.step()
        displacement = vector([p.detach() for p in params])-before
        updates.append(dict(step=step+1, standard_gradient=vs.cpu(),visual_gradient=vv.cpu(),prior_gradient=vp.cpu(),
            standard_norm=float(vs.norm()),visual_norm=float(vv.norm()),prior_norm=float(vp.norm()),
            std_vis=angle(vs,vv),std_prior=angle(vs,vp),prior_vis=angle(vp,vv),
            gradient_scale=scale,unscaled_norm=float(raw.norm()),scaled_norm=float(total.norm()),preclip_norm=clip_norm,
            displacement=displacement.cpu(),displacement_norm=float(displacement.norm()),
            displacement_vs_negative_std=angle(displacement,-vs),adam_step=int(opt.state[params[0]]['step'])))
        assert float(displacement.norm()) > 0, 'NO_PARAMETER_UPDATE'
    restored = Adapter(h.shape[-1]).to(h.device)
    restored.load_state_dict(history[-1]['state'])
    assert torch.equal(restored(h),model(h))
    model.load_state_dict(history[0]['state'])
    assert torch.equal(base+model(h),base)
    return dict(arm=arm,history=history,updates=updates,lp_mask=mask.reshape(base.shape[:-1]).cpu(),
                teacher_vds=vds.cpu(),optimizer_initial_empty=True,reload_exact=True,reset_exact=True,
                parameter_count=sum(p.numel() for p in params),GT_used=False)


def cpu_control():
    torch.manual_seed(13)
    logits=torch.randn(2,4,1001,dtype=torch.float64,requires_grad=True)
    target=torch.randn_like(logits).log_softmax(-1).detach()
    lp=logits.log_softmax(-1); loss=reverse_kl(lp,target).mean()
    grad,=torch.autograd.grad(loss,logits)
    expected=lp.exp()*(lp-target-reverse_kl(lp,target)[...,None])/8
    assert torch.allclose(grad,expected,atol=1e-12,rtol=1e-10)
    assert angle(torch.zeros(3),torch.ones(3))['degrees'] is None
    return dict(reverse_KL_analytic_max_error=float((grad-expected).abs().max()),zero_gradient_angle=None)
