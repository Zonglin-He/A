"""Original unguided VE update with an explicit initial noise level.

Existing model weights, physical coordinate units and Euler rule are unchanged.
This Native-warm-start procedure remains a finite approximation, not exact prior
sampling. No data readers, training, GT or candidate selection.
"""
import math
import torch


def frozen_rollout(model, c, seed, sigma_start=.213, sigma_end=.03, levels=16):
    gen=torch.Generator(device=c['native'].device).manual_seed(seed)
    schedule=torch.logspace(math.log10(sigma_start),math.log10(sigma_end),levels,device=c['native'].device)
    epsilon=torch.randn(c['native'].shape,device=c['native'].device,generator=gen)
    x=c['native']+schedule[0]*epsilon
    initial=x.clone();initial[:,0,2:]=0;steps=[]
    with torch.no_grad():
        for i,s in enumerate(schedule):
            x=x.clone();x[:,0,2:]=0
            clean=model(x,s.expand(len(x)),c)
            sn=schedule[i+1]if i+1<len(schedule)else s*0
            proposal=clean+(sn/s)*(x-clean)
            assert torch.isfinite(proposal).all()
            steps.append(dict(level=i,sigma=float(s),input=x.cpu(),clean=clean.cpu(),proposal=proposal.cpu()))
            x=proposal
    return dict(initial=initial.cpu(),epsilon=epsilon.cpu(),steps=steps,final=x.cpu())
