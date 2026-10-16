"""Diagnostic reference perturbations; neither GT access nor method mutation."""
import torch

PERTURBATIONS=['x_minus','x_plus','y_minus','y_plus','scale_minus','scale_plus']


def perturb(anchor,kind):
    b=torch.tensor(anchor['box'],dtype=torch.float64);before=b.clone()
    if kind.startswith('x_'):b[0]+=(.05 if kind.endswith('plus') else -.05)*b[2]
    elif kind.startswith('y_'):b[1]+=(.05 if kind.endswith('plus') else -.05)*b[3]
    elif kind.startswith('scale_'):b[2:] *= 1.05 if kind.endswith('plus') else .95
    else:raise ValueError(kind)
    b=b.clamp(1e-6,1-1e-6)
    return dict(anchor,box=b.tolist()),dict(requested=kind,actual_delta=(b-before).tolist(),clipped=bool(((b==1e-6)|(b==1-1e-6)).any()))


def transfer_state(initial,donor,donor_initial,own,matched):
    delta={k:donor[k].to(v)-donor_initial[k].to(v) for k,v in initial.items()}
    dn=float(torch.cat([v.reshape(-1).double() for v in delta.values()]).norm())
    on=float(torch.cat([(own[k].to(v)-v).reshape(-1).double() for k,v in initial.items()]).norm())
    scale=on/dn if matched and dn>0 else 0. if matched else 1.
    return {k:v+scale*delta[k] for k,v in initial.items()},dict(donor_norm=dn,own_norm=on,scale=scale,donor_zero=dn==0)
