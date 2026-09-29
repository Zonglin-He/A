"""Matched conditional pairwise/reverse-KL rank students with sparse replay."""
import torch
from torch import nn
from torch.nn import functional as F


def create():
    with torch.random.fork_rng():
        torch.manual_seed(20260929)
        net=nn.Sequential(nn.Linear(768,128,dtype=torch.float64),nn.ReLU(),nn.Linear(128,1,dtype=torch.float64))
        nn.init.zeros_(net[2].weight);nn.init.zeros_(net[2].bias)
    return net


def scores(net,item):return item['base']+net(item['phi']).squeeze(-1)


def objective(net,item,mode):
    s=scores(net,item);r=item['teacher']
    if mode=='pairwise':
        pairs=[(i,j) for i in range(len(r)) for j in range(len(r)) if r[i]>r[j]+1e-12]
        return torch.stack([F.softplus(-(s[i]-s[j])) for i,j in pairs]).mean() if pairs else s.sum()*0
    assert mode=='reverse_kl'
    logp=F.log_softmax(s,dim=0);logq=F.log_softmax(r,dim=0)
    return (logp.exp()*(logp-logq)).sum()


def update(net,current,replay,mode):
    def total():
        cur=objective(net,current,mode)
        past=torch.stack([objective(net,x,mode) for x in replay]).mean() if replay else cur*0
        return cur+past,cur,past
    loss,cur,past=total();before=dict(total=float(loss.detach()),current=float(cur.detach()),replay=float(past.detach()))
    net.zero_grad(set_to_none=True);loss.backward();grad={n:v.grad.detach().clone() for n,v in net.named_parameters()}
    with torch.no_grad():
        for n,v in net.named_parameters():v.add_(grad[n],alpha=-.001)
        after,c,a=total()
    assert all(torch.isfinite(v).all() for v in net.parameters())
    return dict(before=before,after=dict(total=float(after),current=float(c),replay=float(a)),gradient_norm=float(torch.sqrt(sum((g*g).sum() for g in grad.values()))),gradient_by_parameter={n:float(g.norm()) for n,g in grad.items()},lr=.001,steps=1,replay_positions=[x['position'] for x in replay]),grad
