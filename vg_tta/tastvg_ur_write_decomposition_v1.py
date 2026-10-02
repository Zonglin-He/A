"""Gradient difference and one ordinary SGD write, without normalization."""
import torch


def difference(routed, uniform):
    assert list(routed) == list(uniform)
    return {name: routed[name] - uniform[name] for name in routed}


def sgd(state, gradients, lr):
    assert lr > 0 and list(state) == list(gradients)
    result = {name: value.detach().clone() for name, value in state.items()}
    with torch.no_grad():
        for name, value in result.items():
            assert torch.isfinite(gradients[name]).all()
            value.add_(gradients[name].to(value), alpha=-lr)
    return result


def gradient_geometry(uniform, routed):
    u = torch.cat([value.detach().cpu().double().flatten() for value in uniform.values()])
    r = torch.cat([value.detach().cpu().double().flatten() for value in routed.values()])
    un, rn = float(u.norm()), float(r.norm())
    cosine = float(u @ r / (un * rn)) if un > 0 and rn > 0 else None
    return dict(norm_U=un, norm_R=rn, norm_specific=float((r-u).norm()),
                dot_U_R=float(u @ r), cosine_U_R=cosine,
                cosine_defined=cosine is not None,
                specific_norm_from_rounded_float32=float(torch.cat([
                    value.double().flatten() for value in difference(routed, uniform).values()]).norm()))
