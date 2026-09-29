"""Label-free R16 signal losses and coefficient geometry. No GT/data/model access."""
import hashlib
import torch

def select_dev16(rows):
    parents=sorted({r['source'] for r in rows},key=lambda p:hashlib.sha256(('DESTA-A05-v1|parent|'+str(p)).encode()).hexdigest())
    chosen=[]
    for parent in parents:
        candidates=[(i,r) for i,r in enumerate(rows) if r['source']==parent]
        i,r=min(candidates,key=lambda x:hashlib.sha256(('DESTA-A05-v1|query|'+str(x[1]['key'])).encode()).hexdigest())
        chosen.append({**r,'dev64_index':i})
    return chosen

def signal_loss(logits,teacher,signal):
    if logits.ndim not in (2,3) or logits.shape!=teacher.shape:raise ValueError('Exact native support required')
    logp=logits.float().log_softmax(-1)
    if signal=='U-Consistency':
        logt=teacher.detach().to(logits.device).float().log_softmax(-1)
        return (logt.exp()*(logt-logp)).sum(-1).mean()
    if signal=='U-Entropy':return -(logp.exp()*logp).sum(-1).mean()
    raise ValueError(signal)

def unit(x):
    x=x.double();n=x.norm()
    if not torch.isfinite(n):raise ValueError('Nonfinite gradient')
    return x/n if n>0 else torch.zeros_like(x)

def balanced_direction(gt,gs):
    return -unit(unit(gt)+unit(gs))

def passes(median,event_positive,event_count,spatial_positive,spatial_count):
    return (median>=.10 and event_count>0 and spatial_count>0
      and event_positive/event_count>=.65 and spatial_positive/spatial_count>=.65)
