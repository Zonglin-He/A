"""Frozen clean-estimate observation guidance; no training/data/GT readers.

Fixed clock uses exact Native physical membership. Free clock retains the old
event-relative hard gate. Neither implements a full-video identity trajectory.
"""
import math
import torch
from torch.nn import functional as F
from vg_tta.stvg_joint_posterior_v1 import bounds


def project_time(x,c,fixed):
    if not fixed:return x
    return torch.cat((c['native'][:,:1],x[:,1:]),1)


def observation_terms(clean,c,obs,variance,fixed=False):
    if not len(obs['time']):
        return clean.sum((1,2))*0,dict(active_count=0.,responsibility_mean=0.,residual_mean=0.)
    a,b=bounds(clean).unbind(-1)
    u=((obs['time'][None]-a[:,None])/(b-a-c['unit'].reshape(-1))[:,None].clamp_min(1e-4)).clamp(0,1)
    grid=torch.stack((u*2-1,torch.zeros_like(u)),-1)[:,None]
    pred=F.grid_sample(clean[:,1:].sigmoid().transpose(1,2)[:,:,None],grid,align_corners=True,padding_mode='border').squeeze(2).transpose(1,2)
    residual=(pred-obs['boxes'][None]).square().sum(-1)
    lognormal=-.5*residual/variance-2*math.log(2*math.pi*variance)
    # Same fixed .5 mixture prior for actual and oracle; outlier density1 retained.
    mix=torch.logaddexp(lognormal-math.log(2),torch.full_like(lognormal,-math.log(2)))
    inside=(obs['fixed_inside'][None].expand(len(clean),-1)if fixed else((obs['time'][None]>=a[:,None])&(obs['time'][None]<=b[:,None]))).detach().float()
    ll=(mix*inside).sum(-1)/len(obs['time'])
    resp=(lognormal-math.log(2)-mix).exp()
    stats=dict(active_count=float(inside.sum(-1).mean().detach()),responsibility_mean=float((resp*inside).sum().detach()/inside.sum().clamp_min(1)),residual_mean=float((residual*inside).sum().detach()/inside.sum().clamp_min(1)))
    return ll,stats


def guided_rollout(model,c,epsilon,obs,variance,fixed=False):
    sigma=torch.logspace(math.log10(.213),math.log10(.03),16,device=c['native'].device)
    x=project_time(c['native']+sigma[0]*epsilon,c,fixed).clone();initial=x.cpu();trace=[]
    guided=len(obs['time'])>0
    for i,s in enumerate(sigma):
        x=x.detach();x[:,0,2:]=0;x.requires_grad_(guided)
        with torch.set_grad_enabled(guided):
            effective=project_time(x,c,fixed)
            clean=project_time(model(effective,s.expand(len(x)),c),c,fixed)
            if guided:
                ll,stats=observation_terms(clean,c,obs,variance,fixed)
                grad,=torch.autograd.grad(ll.sum(),x)
            else:
                grad=torch.zeros_like(x);ll=x.new_zeros(len(x));stats=dict(active_count=0.,responsibility_mean=0.,residual_mean=0.)
            delta=(.02*s.square()*grad).clamp(-.2,.2)
            if fixed:delta=delta.clone();delta[:,0]=0
            sn=sigma[i+1]if i+1<len(sigma)else s*0
            proposal=project_time(clean+(sn/s)*(effective-clean)+delta,c,fixed)
            assert torch.isfinite(proposal).all()
            if fixed:assert torch.equal(proposal[:,0],c['native'][:,0])
            trace.append(dict(level=i,sigma=float(s),loglik=float(ll.mean().detach()),gradient_norm=float(grad.norm().detach()),spatial_gradient_norm=float(grad[:,1:].norm().detach()),temporal_gradient_norm=float(grad[:,0,:2].norm().detach()),delta_norm=float(delta.norm().detach()),delta_max=float(delta.abs().max().detach()),clipped_components=int((.02*s.square()*grad).abs().gt(.2).sum().detach()),**stats))
            x=proposal.detach()
    assert all(p.grad is None for p in model.parameters())
    return dict(initial=initial,final=x.cpu(),trace=trace,fixed_clock=fixed,reference_count=len(obs['time']))
