"""Twenty external variables only; no backbone module or expert reader."""
import hashlib,math
import numpy as np
import torch

ARMS=['A_no_learning','B_Direct','C_sparse_OPD','D_dense_OPD','F_fixed_rollout']
CONFIG=dict(lr=.01,tau=.25,steps=5,samples=32,standardized_sigma=1.,Adam_reset_each_expert=True)


def phi(base):
    return torch.cat((base.new_ones(1),base.mean(0)))


def transform(base,a,scale):
    d=a*scale
    return torch.cat((base[...,:2]+d[...,None,:2],base[...,2:]*torch.exp(d[...,None,2:])),dim=-1)


def iou(a,b):
    ap=torch.cat((a[...,:2]-a[...,2:]/2,a[...,:2]+a[...,2:]/2),-1)
    bp=torch.cat((b[...,:2]-b[...,2:]/2,b[...,:2]+b[...,2:]/2),-1)
    inter=(torch.minimum(ap[...,2:],bp[...,2:])-torch.maximum(ap[...,:2],bp[...,:2])).clamp_min(0).prod(-1)
    union=a[...,2:].prod(-1)+b[...,2:].prod(-1)-inter
    return inter/union


def direct_loss(boxes,target):
    from methods.decota_final_simplified_v1.objectives import generalized_iou
    return (5*(boxes-target).abs().sum(-1)+2*(1-generalized_iou(boxes,target))).mean()


@torch.no_grad()
def rollout(mu,base,scale,target,positions,seedkey,step):
    seed=int(hashlib.sha256(f'external-opd20-v1|{seedkey}|{step}'.encode()).hexdigest()[:15],16)
    generator=torch.Generator().manual_seed(seed)
    eps=torch.randn((16,4),generator=generator,dtype=torch.float64);eps=torch.cat((eps,-eps),0)
    actions=mu[None]+eps;boxes=transform(base,actions,scale)
    reward=iou(boxes[:,positions],target[None,positions]).mean(1)
    weights=torch.softmax(reward/CONFIG['tau'],0).detach()
    return dict(mean=mu.clone(),actions=actions,noise=eps,rewards=reward,weights=weights,seed=seed,
        informative=bool((reward!=reward[:1]).any()))


def likelihood(mu,r):
    assert all(not r[k].requires_grad for k in ['mean','actions','rewards','weights'])
    kl=(mu-r['mean']).square().sum()/2
    logp=-(r['actions']-mu[None]).square().sum(1)/2-2*math.log(2*math.pi)
    return kl-((r['weights']-1/32)*logp).sum()


def snapshot(value):
    if torch.is_tensor(value):return value.detach().clone()
    if isinstance(value,dict):return {k:snapshot(v)for k,v in value.items()}
    if isinstance(value,list):return [snapshot(v)for v in value]
    return value


@torch.enable_grad()
def update(W,base,scale,target,positions,arm,key,active):
    assert W.shape==(4,5)and W.dtype==torch.float64 and W.requires_grad
    assert arm in ARMS[1:];psi=phi(base);before=W.detach().clone();trace=[];fixed=None
    opt=torch.optim.Adam([W],lr=CONFIG['lr'],betas=(.9,.999),eps=1e-8,weight_decay=0.)
    assert not opt.state
    used=list(positions)if arm=='C_sparse_OPD'else list(range(len(base)))
    assert not active or len(used)>0
    for step in range(CONFIG['steps']):
        old=W.detach().clone();mu=W@psi;boxes=transform(base,mu,scale);r=None
        if not active:
            trace.append(dict(step=step,updated=False,no_valid_expert=True,W_before=old,W_after=old));continue
        if arm=='B_Direct':loss=direct_loss(boxes,target);informative=True;error=None;gmu=None
        else:
            if arm=='F_fixed_rollout'and fixed is not None:r=fixed
            else:
                r=rollout(mu.detach(),base,scale,target,used,key,step)
                if arm=='F_fixed_rollout':fixed=r
            loss=likelihood(mu,r);informative=r['informative']
            gmu=torch.autograd.grad(loss,mu,retain_graph=True)[0]
            expected=mu.detach()-r['mean']-((r['weights']-1/32)[:,None]*(r['actions']-mu.detach()[None])).sum(0)
            error=float((expected-gmu).abs().max());assert error<1e-12
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);gradient=None
        if informative:
            gradient=torch.autograd.grad(loss,W)[0];assert torch.isfinite(gradient).all()
            if gmu is not None:assert float((gradient-gmu[:,None]*psi[None]).abs().max())<1e-12
            W.grad=gradient.detach();opt.step()
        assert torch.isfinite(W).all()
        with torch.no_grad():newmu=W@psi;post=transform(base,newmu,scale);displacement=post-boxes.detach()
        trace.append(dict(step=step,updated=informative,no_valid_expert=False,loss=float(loss.detach()),
            W_before=old,W_after=W.detach().clone(),mu=mu.detach().clone(),mu_after=newmu.clone(),
            gradient=None if gradient is None else gradient.detach().clone(),Adam_after=snapshot(opt.state_dict()),
            rollout=snapshot(r),analytic_mean_gradient_error=error,
            parameter_step_norm=float((W.detach()-old).norm()),output_displacement_l2=float(displacement.norm()),
            output_coordinate_absolute_mean=float(displacement.abs().mean()),
            reference_IoU_before=float(iou(boxes.detach(),target).mean()),reference_IoU_after=float(iou(post,target).mean()),
            reward_variance=None if r is None else float(r['rewards'].var(unbiased=False)),
            ESS=None if r is None else float(1/r['weights'].square().sum())))
    return dict(arm=arm,W_before=before,W_after=W.detach().clone(),phi=psi,
        active=active,positions=used,trace=trace,optimizer_initial_empty=True,
        actual_steps=sum(x['updated']for x in trace),parameter_count=20,
        distinct_rollouts=sum(x.get('rollout')is not None for x in trace)if arm!='F_fixed_rollout'else int(fixed is not None),
        backbone_parameter_updates=0,GT_used=False)


def contract():
    b=torch.tensor([[.3,.4,.1,.2],[.4,.5,.2,.3]],dtype=torch.float64)
    W=torch.zeros((4,5),dtype=torch.float64,requires_grad=True);s=torch.tensor([.05,.04,.1,.2],dtype=torch.float64)
    assert torch.equal(transform(b,W@phi(b),s),b)
    target=transform(b,torch.tensor([.3,-.2,.1,-.1],dtype=torch.float64),s).detach()
    r=rollout((W@phi(b)).detach(),b,s,target,[0,1],'contract',0)
    mu=W@phi(b);grad=torch.autograd.grad(likelihood(mu,r),W)[0]
    expected=-((r['weights']-1/32)[:,None]*(r['actions']-mu.detach()[None])).sum(0)[:,None]*phi(b)[None]
    err=float((grad-expected).abs().max());assert err<1e-12
    a=update(W,b,s,target,[0,1],'D_dense_OPD','contract',True)
    Q=torch.zeros_like(W,requires_grad=True);f=update(Q,b,s,target,[0,1],'F_fixed_rollout','contract',True)
    assert torch.equal(a['trace'][0]['W_after'],f['trace'][0]['W_after'])
    assert all(torch.equal(t['rollout']['actions'],f['trace'][0]['rollout']['actions'])for t in f['trace'])
    assert any(not torch.equal(t['rollout']['actions'],a['trace'][0]['rollout']['actions'])for t in a['trace'][1:])
    assert torch.equal(a['W_after'],W.detach())and a['actual_steps']==5
    return dict(status='pass',count=6,zero_exact_Source=True,parameter_count=20,mean_gradient_max_error=err,
        first_on_policy_fixed_update_exact=True,on_policy_actual_resampling=True,fixed_rollout_actual_reuse=True)
