"""Source-only native actuation controls. No model parameter is optimized.

Free merger deltas and an orthonormal parameterization of the *same frozen*
branch output span use token-space units. The latter absorbs the nonzero gate
and QR scale in a change of variables; it does not test small latent updates.
"""
from contextlib import contextmanager
import re, math
import torch
from torch.nn import functional as F


def native_targets(label, prediction, branch):
    if branch == 'event':
        match=re.search(r'<\|time_start\|>\s*<t(\d+)>\s*<t(\d+)>',label['response'])
        if not match:raise ValueError('Missing registered source endpoint tokens')
        targets=torch.tensor([int(match[1])-1,int(match[2])-1])
        if not (0 <= targets[0] <= targets[1] < len(label['frame_ids'])):raise ValueError('Endpoint outside observed support')
        return targets,torch.ones(2,dtype=torch.bool)
    if branch != 'spatial':raise ValueError(branch)
    positions=prediction['positions']
    if not positions:raise ValueError('No native spatial support; no GT-prefix replacement allowed')
    boxes=torch.tensor([label['boxes_xyxy'][i] for i in positions],dtype=torch.float64)
    valid=torch.tensor([label['box_valid'][i] for i in positions],dtype=torch.bool)
    if not valid.any():raise ValueError('Native spatial support has no annotated referent boxes')
    targets=torch.round(boxes*1000).long()
    if ((targets[valid]<0)|(targets[valid]>1000)).any():raise ValueError('Invalid normalized GT')
    return targets,valid[:,None].expand(-1,4).clone()


def native_ce(logits,targets,valid):
    if logits.shape[:-1] != targets.shape or targets.shape != valid.shape:raise ValueError('Native action support mismatch')
    if not torch.isfinite(logits).all() or not valid.any():raise ValueError('Nonfinite/empty support')
    return F.cross_entropy(logits[valid].float(),targets[valid].long(),reduction='mean')


def margin(logits,targets,valid):
    x=logits.detach().double().cpu()[valid.cpu()];y=targets.cpu()[valid.cpu()]
    target=x.gather(-1,y[:,None]).squeeze(-1)
    other=x.clone();other.scatter_(-1,y[:,None],-torch.inf)
    return (target-other.max(-1).values).tolist()


def span_basis(weight):
    q,r=torch.linalg.qr(weight.detach().double(),mode='reduced')
    if (r.diag().abs()<1e-12).any():raise ValueError('Rank deficient output projection')
    return q.float(),r


def token_delta(parameter,mode,basis=None):
    if mode=='free':return parameter
    if mode!='span' or basis is None:raise ValueError(mode)
    # Unit parameter coordinate variance gives unit mean token variance.
    return F.linear(parameter,basis)*math.sqrt(basis.shape[0]/basis.shape[1])


@contextmanager
def inject_delta(adapter,branch,parameter,mode='free',basis=None):
    if branch not in ('event','spatial'):raise ValueError(branch)
    calls=[]
    def hook(module,args,result):
        old=result['updated_tokens_'+branch]
        delta=token_delta(parameter,mode,basis)
        if old.shape!=delta.shape:raise ValueError('Delta must cover exactly the original merger support')
        out=dict(result);out['updated_tokens_'+branch]=old+delta
        calls.append(1)
        return out
    h=adapter.register_forward_hook(hook)
    try:yield calls
    finally:h.remove()
