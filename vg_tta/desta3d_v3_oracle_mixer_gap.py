"""Frozen B1 analytic-vs-amortized direction audit; no optimizer or new model."""
import math
import torch
from vg_tta.desta3d_v3_decomposition import norm, project, unit_to


def analytic_joint(gt, gs, basis, stock, radius):
    if gt.shape != gs.shape or gt.shape != stock.shape:
        raise ValueError('Common THW support required')
    if any(not torch.isfinite(v).all() for v in (gt, gs, stock, basis)):
        raise ValueError('Nonfinite oracle inputs')
    balanced = unit_to(gt, 1.) + unit_to(gs, 1.)
    return unit_to(-project(balanced, basis), radius*norm(stock))


def cosine(x, y):
    denominator = norm(x)*norm(y)
    return float((x.double()*y.double()).sum())/denominator if denominator else None


def direction_readout(gradients, deltas, stock, radius):
    fn = norm(stock)
    return dict(stock_norm=fn, radius=radius,
                gradient_norms={b:norm(g) for b,g in gradients.items()},
                gradient_cosine=cosine(gradients['event'], gradients['spatial']),
                delta_norms={a:norm(d) for a,d in deltas.items()},
                norm_over_cap={a:norm(d)/(radius*fn) if fn else None for a,d in deltas.items()},
                cosine_to_oracle={a:cosine(d,deltas['oracle']) for a,d in deltas.items()},
                descent_dot={b:{a:float(-(g.double()*d.double()).sum()) for a,d in deltas.items()} for b,g in gradients.items()})


def mixer_readout(mixer, args):
    """Observe unmodified output preactivation; independently reconstruct delta."""
    raw=[]
    h=mixer.output.register_forward_hook(lambda _m,_a,out:raw.append(out.detach().clone()))
    try:
        with torch.no_grad(): delta=mixer(*args)
    finally: h.remove()
    assert len(raw)==1
    logits=raw[0];a=logits.tanh();stock=args[-1]
    scale=stock.detach().float().square().mean().sqrt()*math.sqrt(stock.shape[-1]/a.shape[-1])*mixer.radius
    with torch.no_grad(): reconstructed=torch.nn.functional.linear(a*scale,mixer.basis)
    assert torch.equal(reconstructed,delta),'Observer must not alter mixer'
    rms=float(a.double().square().mean().sqrt())
    observed=norm(delta)/norm(stock)/mixer.radius
    assert abs(observed-rms)<2e-6
    meta=dict(coefficient_rms=rms,norm_over_cap=observed,scale=float(scale),
              fraction_abs_above_099=float((a.abs()>.99).double().mean()),
              fraction_abs_above_0999=float((a.abs()>.999).double().mean()),
              mean_tanh_derivative=float((1-a.double().square()).mean()),
              coefficient_shape=list(a.shape),delta_reconstruction_exact=True)
    return delta,a.detach().cpu(),meta
