"""Finite source-trained conditional joint tube prior. No datasets or GT readers.

VE denoising, dynamic ROI evidence, and an explicit inlier/outlier observation
mixture. This is an approximate conditional posterior, not an exact sampler.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
K=32

def logit(x):return torch.logit(x.clamp(1e-4,1-1e-4))
def bounds(x):
    s=x[:,0,0].sigmoid();e=s+(1-s)*x[:,0,1].sigmoid()
    return torch.stack((s,e),-1)
def encode_bounds(a,b):return torch.stack((logit(a),logit((b-a)/(1-a).clamp_min(1e-4))),-1)
def times(x,unit=0.):
    a,b=bounds(x).unbind(-1);return a[:,None]+(b-a-torch.as_tensor(unit,device=x.device).reshape(-1)).clamp_min(1e-6)[:,None]*torch.linspace(0,1,K,device=x.device)
def roi(maps,boxes,t):
    """Differentiable space/time ROI samples from frozen pooled visual maps.

    Four inner ROI points; clipping affects sampling only. No box is replaced.
    maps B,C,32,8,8; boxes B,L,4 normalized cxcywh, t B,L in [0,1].
    """
    c=boxes[...,:2];wh=boxes[...,2:];d=boxes.new_tensor([[-.25,-.25],[-.25,.25],[.25,-.25],[.25,.25]])
    xy=(c[:,:,None]+wh[:,:,None]*d).clamp(0,1)*2-1
    tt=(t.clamp(0,1)*2-1)[:,:,None,None].expand(-1,-1,4,1)
    grid=torch.cat((xy,tt),-1)[:,:,None]
    return F.grid_sample(maps,grid,mode='bilinear',padding_mode='border',align_corners=True).mean(-1).squeeze(-1).transpose(1,2)

class JointPrior(nn.Module):
    def __init__(self,separated=False):
        super().__init__();self.separated=separated
        self.state=nn.Linear(7,64);self.roi=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,64))
        self.context=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,64));self.text=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,64))
        self.cross=nn.MultiheadAttention(64,4,batch_first=True,dropout=0.)
        self.sequence=nn.TransformerEncoder(nn.TransformerEncoderLayer(64,4,128,dropout=0.,batch_first=True,activation='gelu',norm_first=True),2,enable_nested_tensor=False)
        self.out=nn.Sequential(nn.LayerNorm(64),nn.Linear(64,4));self.position=nn.Parameter(torch.zeros(1,K+1,64))
        mask=torch.zeros(K+1,K+1,dtype=torch.bool)
        if separated:mask[0,1:]=True;mask[1:,0]=True
        self.register_buffer('block',mask)
    def forward(self,x,sigma,c):
        B=len(x);t=times(x,c['unit'])if not self.separated else times(c['native'],c['unit'])
        r=roi(c['maps'],x[:,1:].sigmoid(),t);r=torch.cat((r.mean(1,keepdim=True)*0,r),1)
        sig=torch.as_tensor(sigma,device=x.device).expand(B)
        ss=torch.stack((sig.log(),sig.log().sin(),sig.log().cos()),-1)[:,None].expand(-1,K+1,-1)
        v=self.state(torch.cat((x,ss),-1))+self.roi(r)+self.position
        ctxt=torch.cat((self.context(c['context']),self.text(c['query'])),1)
        pad=torch.cat((torch.zeros(B,K,dtype=torch.bool,device=x.device),c['qmask']),1)
        h,_=self.cross(v,ctxt,ctxt,key_padding_mask=pad,need_weights=False)
        h=self.sequence(v+h,mask=self.block if self.separated else None)
        out=self.out(h);return out

class Association(nn.Module):
    """Source geometric compatibility, not verified target identity recognition."""
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.LayerNorm(768),nn.Linear(768,64),nn.GELU(),nn.Linear(64,1))
    def forward(self,visual,query):return self.net(torch.cat((visual,query),-1)).squeeze(-1)

def observation_loglik(x,c,obs,prior,variance):
    """Marginalize non-target / unidentifiable observations; do not GT-filter.

    Event-exterior observations have an uninformative event likelihood. This
    prototype does NOT learn event-exterior identity trajectories.
    """
    if not len(obs['time']):return x.sum((1,2))*0
    a,b=bounds(x).unbind(-1);u=((obs['time'][None]-a[:,None])/(b-a-c['unit'].reshape(-1))[:,None].clamp_min(1e-4)).clamp(0,1)
    grid=torch.stack((u*2-1,torch.zeros_like(u)),-1)[:,None]
    traj=x[:,1:].sigmoid().transpose(1,2)[:,:,None]
    predicted=F.grid_sample(traj,grid,align_corners=True,padding_mode='border').squeeze(2).transpose(1,2)
    residual=(predicted-obs['boxes'][None]).square().sum(-1)
    lognormal=-.5*residual/variance-2*math.log(2*math.pi*variance)
    p=prior.clamp(1e-4,1-1e-4)[None]
    logmix=torch.logaddexp(p.log()+lognormal,torch.log1p(-p))
    inside=((obs['time'][None]>=a[:,None])&(obs['time'][None]<=b[:,None])).detach().float()
    return (logmix*inside).sum(-1)/max(len(obs['time']),1)

def posterior(model,c,obs,prior,variance,kind,seed,samples=4,levels=16):
    """Fixed finite VE probability-flow approximation; no target-score search."""
    gen=torch.Generator(device=c['native'].device).manual_seed(seed)
    sigma=torch.logspace(math.log10(2.),math.log10(.03),levels,device=c['native'].device)
    x=c['native']+sigma[0]*torch.randn(c['native'].shape,device=c['native'].device,generator=gen)
    trace=[]
    for i,s in enumerate(sigma):
        x[:,0,2:]=0; x=x.detach().requires_grad_(True)
        with torch.no_grad():clean=model(x,s.expand(len(x)),c)
        if kind=='deterministic':proposal=.5*x+.5*clean
        else:
            sn=sigma[i+1]if i+1<len(sigma)else s*0
            proposal=clean+(sn/s)*(x-clean)
        ll=observation_loglik(x,c,obs,prior,variance)
        grad=torch.autograd.grad(ll.sum(),x)[0]
        # One fixed, bounded observation step; the prior update remains joint.
        delta=(.02*s.square()*grad).clamp(-.2,.2)
        x=(proposal+delta).detach()
        assert torch.isfinite(x).all()
        trace.append(dict(level=i,sigma=float(s),mean_loglik=float(ll.mean().detach()),observation_gradient_norm=float(grad.norm().detach())))
    return x,trace
