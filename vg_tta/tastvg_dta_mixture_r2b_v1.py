"""Equal mixture on legal joint support; unchanged R1 three-step head dynamics."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from vg_tta.tastvg_dta_oracle_v1 import penultimate,offsets,joint,gaussian,kl,native_decode,tensor_hash,STEPS,BETA
from vg_tta.tastvg_dta_expert_r2_v1 import read_guard as r2_read_guard

def validate_proposals(proposals):
    x=np.asarray(proposals,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=2 or not len(x) or not np.isfinite(x).all() or not (x[:,1]>x[:,0]).all():
        raise ValueError('Nonempty finite positive-duration proposal support required')
    return x

def mixture(frame_ids,proposals):
    x=validate_proposals(proposals)
    qs=[gaussian(frame_ids,span) for span in x]
    if len(x)==1:return qs[0]
    sigmas=[sigma for q,sigma in qs];assert len(set(sigmas))==1
    q=torch.logsumexp(torch.stack([q for q,sigma in qs]),0)-math.log(len(x))
    assert torch.isfinite(q).all() and abs(float(q.exp().sum())-1)<1e-12
    return q,sigmas[0]

def teacher_decode(teachers,frame_ids,parts):
    raw=[]
    for logq,at in zip(teachers,parts):
        pairs=torch.triu_indices(len(at),len(at),offset=1);ix=int(logq.argmax())
        raw.append([at[int(pairs[0,ix])],at[int(pairs[1,ix])]])
    pair=[min(r[0] for r in raw),max(r[1] for r in raw)]
    return dict(indices=pair,physical_interval=[frame_ids[pair[0]],frame_ids[pair[1]]+1],offset_indices=raw)

def fit_mixture(hidden,frame_ids,head,proposals,lr,records=None):
    proposals=validate_proposals(proposals);x=penultimate(hidden,head);parts=offsets(frame_ids,records)
    w=head['1.weight'].clone().requires_grad_(True);b=head['1.bias'].clone().requires_grad_(True)
    initial_w=w.detach().clone();initial_b=b.detach().clone();z0=F.linear(x,w,b).detach()
    prior=[joint(z0[p])[0].detach() for p in parts]
    teachers,sigmas=zip(*(mixture([frame_ids[i] for i in p],proposals) for p in parts))
    states=[dict(weight=initial_w,bias=initial_b)];trace=[];logits=[z0]
    def terms(z):
        pp=[joint(z[p])[0] for p in parts]
        q=torch.stack([kl(a,p) for a,p in zip(teachers,pp)]).mean()
        anchor=torch.stack([kl(a,p) for a,p in zip(prior,pp)]).mean()
        return q,anchor,q+BETA*anchor
    for step in range(STEPS):
        z=F.linear(x,w,b);q,anchor,loss=terms(z);gw,gb=torch.autograd.grad(loss,(w,b))
        assert torch.isfinite(loss) and torch.isfinite(gw).all() and torch.isfinite(gb).all()
        old_w=w.detach().clone();old_b=b.detach().clone()
        with torch.no_grad():
            w.add_(gw,alpha=-lr);b.add_(gb,alpha=-lr);after=F.linear(x,w,b);aq,aa,al=terms(after)
        displacement=torch.cat(((w.detach()-initial_w).flatten(),b.detach()-initial_b))
        trace.append(dict(step=step+1,teacher_KL=float(q.detach()),prior_KL=float(anchor.detach()),loss=float(loss.detach()),
            after_teacher_KL=float(aq),after_prior_KL=float(aa),after_loss=float(al),
            gradient_norm=float(torch.sqrt(gw.double().square().sum()+gb.double().square().sum())),bias_gradient_norm=float(gb.double().norm()),
            step_displacement=float(torch.sqrt((w.detach()-old_w).double().square().sum()+(b.detach()-old_b).double().square().sum())),
            arrival_displacement=float(displacement.double().norm()),parameters_sha256=tensor_hash(torch.cat((w.detach().flatten(),b.detach()))),
            prediction=native_decode(after,frame_ids,parts)))
        states.append(dict(weight=w.detach().clone(),bias=b.detach().clone(),weight_gradient=gw.detach().clone(),bias_gradient=gb.detach().clone()));logits.append(after.detach().clone())
    return dict(before=native_decode(z0,frame_ids,parts),after=trace[-1]['prediction'],teacher_MAP=teacher_decode(teachers,frame_ids,parts),trace=trace,
        states=states,logits=logits,logq=list(teachers),logp0=prior,sigma_frames=list(sigmas),proposal_count=len(proposals),
        head_initial_sha256=tensor_hash(torch.cat((initial_w.flatten(),initial_b))),head_final_sha256=trace[-1]['parameters_sha256'],
        spatial_changed=False,episodic_reset=True,GT_supervised=False,lr=lr,steps=STEPS,beta=BETA)

def read_guard(event,args):
    r2_read_guard(event,args)
    if event!='open':return
    path=str(args[0]);mode=args[1]
    if isinstance(mode,str) and mode=='w':return
    if '/results/' in path and path.endswith(('/CONFIG.json','/DEPLOY_SELECTION.json')):return
    if any(s in path for s in ['/TEACHER_SELECTION_SCORED','/SOURCE_CONCENTRATION','/TEACHER_DIAGNOSTICS','/EXECUTION_DIAGNOSTICS','/TRACES_SCORED','_TRACES_SCORED.json']):
        raise PermissionError('E-Mix cannot read scored/oracle evidence: '+path)
