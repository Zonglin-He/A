"""Persistent spatial parameters, detached student support, expert scalar RKL."""
import torch
from vg_tta.tastvg_causal_round2_v1 import forward


def geometry(central,candidates,l1_coef,giou_coef):
    q=candidates.detach();p=central.unsqueeze(0)
    a,b=p[...,:2]-p[...,2:]/2,p[...,:2]+p[...,2:]/2
    c,d=q[...,:2]-q[...,2:]/2,q[...,:2]+q[...,2:]/2
    inter=(torch.minimum(b,d)-torch.maximum(a,c)).clamp_min(0).prod(-1)
    union=p[...,2:].prod(-1)+q[...,2:].prod(-1)-inter
    enclosing=(torch.maximum(b,d)-torch.minimum(a,c)).prod(-1)
    giou=inter/union-(enclosing-union)/enclosing
    return l1_coef*(p-q).abs().sum(-1).mean(-1)+giou_coef*(1-giou).mean(-1)


def reverse_kl(central,candidates,rewards,coeff):
    distances=geometry(central,candidates,*coeff);logp=(-distances).log_softmax(0)
    logq=rewards.detach().log_softmax(0)
    return (logp.exp()*(logp-logq)).sum(),logp.exp(),logq.exp(),distances


class SpatialActor:
    def __init__(self,model):
        self.model=model;self.query=torch.nn.Parameter(torch.zeros(256,device=next(model.parameters()).device))
        self.named=[('spatial.query_residual',self.query)]
        for ln in ['norm1','norm3','norm4']:
            for name,p in getattr(model.ground_decoder.decoder.layers[5],ln).named_parameters():p.requires_grad_(True);self.named.append((f'spatial.layers.5.{ln}.{name}',p))
        self.initial=self.state();assert sum(p.numel() for _,p in self.named)==1792
    def state(self):return {n:p.detach().clone() for n,p in self.named}
    def restore(self,state):
        with torch.no_grad():
            for n,p in self.named:p.copy_(state[n]);p.grad=None
    def values(self,data):
        count=[0]
        def inject(module,args,kwargs):
            count[0]+=1
            if count[0]%2==0:kwargs=dict(kwargs);kwargs['query_tgt']=kwargs['query_tgt']+self.query[None,None,:]
            return args,kwargs
        hook=self.model.ground_decoder.decoder.register_forward_pre_hook(inject,with_kwargs=True)
        try:return forward(self.model,data,[v['H'] for v in data['views']])
        finally:hook.remove()
    def close(self):
        self.restore(self.initial)
        for _,p in self.named:p.requires_grad_(False)
