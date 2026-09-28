"""No-update matched-magnitude intervention before/after the frozen local reader."""
from contextlib import contextmanager
import torch
from vg_tta.desta3d_v3_latent_oracle import privileged_branch_latents

@contextmanager
def mask_at_location(adapter,branch,mask,location,alpha=.25):
    if branch not in ('event','spatial') or location not in ('early','late'):raise ValueError('Unknown location/branch')
    if location=='late':
        with privileged_branch_latents(adapter,{branch:mask},event=branch=='event',spatial=branch=='spatial',alpha=alpha):yield
        return
    # P1 is disabled; this is AFTER FiLM, BEFORE branch local convolution.
    if adapter.p1_enabled or getattr(adapter,branch+'_reader') is None:raise ValueError('Requires P0 separate local readers')
    def hook(module,args):
        x=args[0];m=mask.to(x)
        if x.ndim!=5 or m.shape!=(x.shape[0],*x.shape[2:]) or not torch.isfinite(m).all() or (m<0).any() or (m>1).any():raise ValueError('Early mask support mismatch')
        return (x*(alpha+(1-alpha)*m).unsqueeze(1),*args[1:])
    h=getattr(adapter,branch+'_reader').register_forward_pre_hook(hook)
    try:yield
    finally:h.remove()

def match_delta(base,candidate,target_norm):
    if base.dtype!=torch.float32 or candidate.dtype!=torch.float32 or candidate.shape!=base.shape:raise ValueError('FP32 identical merger support required')
    raw=candidate-base;norm=raw.double().norm();target=float(target_norm)
    if not torch.isfinite(raw).all() or not torch.isfinite(norm) or not 0<=target<float('inf'):raise ValueError('Nonfinite delta/norm')
    if target==0:scaled=torch.zeros_like(raw);scale=0.
    elif float(norm)==0:raise ValueError('Nonzero target cannot normalize zero intervention')
    else:scale=target/float(norm);scaled=raw*scale
    realized=(base+scaled)-base;actual=float(realized.double().norm())
    # Matching pre-BF16 realized FP32 differences, not post-BF16 counts/norms.
    if abs(actual-target)>max(1e-10,target*1e-3):raise ValueError('Realized FP32 magnitude violates fixed tolerance')
    return raw,scaled,{'raw_norm':float(norm),'target_norm':target,'scale':scale,'realized_norm':actual,
        'normalization':'FP64 norm, FP32 delta multiplication/addition; no amplitude search','tolerance_relative':1e-3}
