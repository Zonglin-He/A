"""F25 isolated label-free boundary adaptation and sparse spatial adaptation.

No GT, cohort ID, source ID, data selection or outcome reader is used here.
The deployed DeCoTA implementation is intentionally not modified.
"""
import copy
import math
import numpy as np
import torch
from vg_tta.temporal_optimizer_probe_v1 import cpu_state, logits
from vg_tta.decota_tastvg_episode_v1 import fitted_merge, native_view_indices
from vg_tta.design_claims_v1 import SpatialInterface as ParentSpatialInterface
from methods.decota_v1.api import decode


def boundary_evidence(action_logits, frame_ids, fps, window=2):
    """Integrate a piecewise constant logit over midpoint physical time cells.

    End candidates are at (frame_id+1)/fps, as in the actual evaluator. Cells
    cover [first_id,last_id+1), not fictional unobserved outside-video frames.
    Partial windows are normalized by their actual observed duration. If
    either side is empty, return zero evidence and mark the boundary censored.
    """
    x=np.asarray(action_logits,dtype=np.float64).reshape(-1)
    f=np.asarray(frame_ids,dtype=np.float64);t=f/fps
    assert len(x)==len(t)>=2 and np.isfinite(x).all() and (np.diff(t)>0).all() and fps>0
    edge=np.r_[t[0],(t[:-1]+t[1:])/2,(f[-1]+1)/fps]
    h=float(window*np.median(np.diff(t)))
    def at(u,sign):
        before=np.maximum(0,np.minimum(edge[1:],u)-np.maximum(edge[:-1],u-h))
        after=np.maximum(0,np.minimum(edge[1:],u+h)-np.maximum(edge[:-1],u))
        a,b=float(before.sum()),float(after.sum());censored=a<=1e-12 or b<=1e-12
        lo=None if a<=1e-12 else float(before@x/a);hi=None if b<=1e-12 else float(after@x/b)
        return dict(value=0. if censored or abs(hi-lo)<1e-12 else float(np.tanh(sign*(hi-lo))),censored=censored,
                    before_duration=a,after_duration=b,before_mean=lo,after_mean=hi,u=float(u))
    s=[at(u,1) for u in t];e=[at(u,-1) for u in (f+1)/fps]
    return dict(start=np.array([v['value'] for v in s]),end=np.array([v['value'] for v in e]),
                start_audit=s,end_audit=e,h_seconds=h,window=window,cell_edges=edge,
                action_logits=x,frame_ids=list(map(int,frame_ids)),fps=fps)


def legal_logp(z):
    z=z.reshape(-1,2).double();ij=torch.triu_indices(len(z),len(z),1,device=z.device)
    s=z[ij[0],0]+z[ij[1],1]
    return s-torch.logsumexp(s,0),ij


def offset_evidence(evidence,records,ids,device):
    lookup={v:i for i,v in enumerate(ids)};out=[]
    for r in records:
        ix=[lookup[v] for v in r['frame_ids']]
        out.append(torch.tensor(np.stack([evidence['start'][ix],evidence['end'][ix]],axis=-1),device=device,dtype=torch.float64))
    return out


def distribution_loss(zs,reference,signals,eta):
    terms=[];parts=[]
    for z,l0,sg in zip(zs,reference,signals):
        lp,ij=legal_logp(z);p=lp.exp();E=sg[ij[0],0]+sg[ij[1],1]
        kl=(p*(lp-l0)).sum();reward=(p*E).sum();terms.append(kl-eta*reward)
        parts.append(dict(kl=float(kl.detach()),reward=float(reward.detach())))
    return torch.stack(terms).mean(),parts


def paired_readout(zs,records,ids,base):
    direct=list(fitted_merge(zs,records,ids))
    reposition=list(decode(base['temporal_logits'],direct,ids,native_indices=base['predicted_indices'])['indices'])
    return dict(direct=direct,reposition=reposition,offset_indices=[list(native_view_indices(z)) for z in zs])


def analytic_target(z0,signals,eta,records,ids,base):
    # Match the native decoder arithmetic, including FP16 tie handling.
    z=[(v.double()+eta*s[None]).to(v.dtype) for v,s in zip(z0,signals)]
    exact=[];identity=[]
    for v,s in zip(z0,signals):
        l0,ij=legal_logp(v);E=s[ij[0],0]+s[ij[1],1];a=l0+eta*E
        logz=torch.logsumexp(a,0);lp=a-logz
        identity.append(float(((lp.exp()*(lp-l0)).sum()-eta*(lp.exp()*E).sum()+logz).abs()))
        exact.append(lp.detach().cpu())
    return dict(**paired_readout(z,records,ids,base),logits=[v.cpu() for v in z],exact_log_pstar=exact,
                algebra_max_error=max(identity),decoder_dtype=str(z[0].dtype),parameters_updated=False)


def boundary_fit(source_head,inputs,records,ids,base,evidence,eta,lr,steps=5):
    h=copy.deepcopy(source_head).float().eval().requires_grad_(True);initial=cpu_state(h)
    device=next(h.parameters()).device;sg=offset_evidence(evidence,records,ids,device)
    with torch.no_grad():z0=logits(h,inputs);ref=[legal_logp(z)[0].detach() for z in z0]
    assert paired_readout(z0,records,ids,base)['direct']==list(base['predicted_indices'])
    opt=torch.optim.AdamW(h.parameters(),lr=lr,eps=1e-4,weight_decay=0.)
    analytic=analytic_target(z0,sg,eta,records,ids,base)
    path=[];best=float('inf');best_state=initial;best_step=0;failure=None
    def value():
        zs=logits(h,inputs);v,parts=distribution_loss(zs,ref,sg,eta);return v,zs,parts
    for step in range(steps+1):
        with torch.no_grad():loss,zs,parts=value()
        if not torch.isfinite(loss):failure='nonfinite_current_loss';break
        lv=float(loss)
        if lv<best:best=lv;best_state=cpu_state(h);best_step=step
        st=dict(step=step,loss=lv,parts=parts,**paired_readout(zs,records,ids,base),
                logits=[v.detach().cpu() for v in zs],head_state=cpu_state(h),
                state_delta_norm=math.sqrt(sum(float((v.double()-initial[k]).square().sum()) for k,v in cpu_state(h).items())))
        path.append(st)
        if step==steps:break
        # No signal / lr0 is an explicit native no-update condition, NOT B fallback.
        if lr==0 or all(not bool(s.any()) for s in sg):
            st.update(accepted=False,trials=[],reason='lr0_or_zero_evidence');continue
        opt.zero_grad(set_to_none=True);v,_,_=value();v.backward()
        grads=[p.grad for p in h.parameters()]
        if any(g is None or not bool(torch.isfinite(g).all()) for g in grads):
            failure='nonfinite_or_missing_gradient';st['accepted']=False;break
        st['gradient_norm']=math.sqrt(sum(float(g.double().square().sum()) for g in grads))
        before=cpu_state(h);ob=copy.deepcopy(opt.state_dict());opt.step();after=cpu_state(h)
        trials=[];accepted=False
        for alpha in [1.,.5,.25,.125]:
            candidate=after if alpha==1. else {n:before[n]+alpha*(after[n]-before[n]) for n in before}
            h.load_state_dict(candidate)
            with torch.no_grad():v,_,_=value()
            finite=bool(torch.isfinite(v));trials.append(dict(alpha=alpha,loss=float(v) if finite else None))
            if finite and float(v)<lv-1e-9:accepted=True;break
        if not accepted:h.load_state_dict(before);opt.load_state_dict(ob)
        st.update(accepted=accepted,trials=trials)
    # On a numerical failure use native; never a coverage-derived state.
    if failure:best_state=initial;best_step=0
    h.load_state_dict(best_state)
    with torch.no_grad():end,zs,parts=value()
    target=path[best_step];assert all(torch.equal(a.cpu(),b) for a,b in zip(zs,target['logits']))
    out=dict(eta=eta,lr=lr,steps=steps,coverage_coefficient=0.,anchor_coefficient=0.,
             objective='KL(P_phi||P0)-eta*E_Pphi[fixed_local_boundary_contrast]',
             evidence_frozen=True,parameters=sum(p.numel() for p in h.parameters()),path=path,
             best_step=best_step,best_restore_exact=True,failure=failure,**paired_readout(zs,records,ids,base),
             analytic=analytic,final_loss=float(end),GT_online=False)
    del h
    return out


class SpatialInterface(ParentSpatialInterface):
    def __init__(self,model,caches,n,scope):
        super().__init__(model,caches,n,'query_ln' if scope=='ln' else scope)
        if scope=='ln':
            self.delta.requires_grad_(False);self.named=[(k,v) for k,v in self.named if k!='query_residual']
            self.initial=self.state()


def spatial_fit(interface,base,anchors,lr,steps=10,legacy_alpha1=False):
    """Keep every step's actual student boxes; never paste teacher boxes back."""
    from vg_tta.metrics import generalized_box_iou_aligned_cxcywh
    interface.restore(interface.initial);named=interface.named
    result=dict(lr=lr,steps=steps,parameters={n:p.numel() for n,p in named},
                parameter_count=sum(p.numel() for _,p in named),anchors=anchors,states=[],GT_online=False)
    if not anchors:return {**result,'boxes':base.clone(),'best_step':0,'skipped':'no_anchors','failure':None}
    pos=torch.tensor([a['position'] for a in anchors],device='cuda')
    target=torch.tensor([a['box'] for a in anchors],device='cuda')
    def value():
        b,_=interface.values();pred=b[pos]
        data=(5*(pred-target).abs().sum(-1)+2*(1-generalized_box_iou_aligned_cxcywh(pred,target))).mean()
        return data+sum((p-interface.initial[n]).square().sum() for n,p in named)*1e-4,b
    opt=torch.optim.Adam([p for _,p in named],lr=lr,eps=1e-8)
    best=float('inf');saved=interface.state();selected=0;failure=None
    for k in range(steps+1):
        opt.zero_grad(set_to_none=True);loss,b=value();lv=float(loss.detach())
        if not torch.isfinite(loss):failure='nonfinite_loss';break
        if k==0:assert torch.equal(b.detach().cpu(),base),'spatial initial output mismatch'
        if lv<best:best=lv;saved=interface.state();selected=k
        st=dict(step=k,loss=lv,boxes=b.detach().cpu(),best_loss=best,best_step=selected)
        result['states'].append(st)
        if k==steps:break
        loss.backward();gn={n:float(p.grad.norm()) if p.grad is not None else None for n,p in named};st['gradient_norms']=gn
        if any(v is None or not math.isfinite(v) for v in gn.values()):failure='missing_nonfinite_gradient';break
        before=interface.state();ob=copy.deepcopy(opt.state_dict());opt.step();after=interface.state()
        trials=[];accepted=False
        for alpha in [1.,.5,.25,.125]:
            interface.restore(after if alpha==1. and not legacy_alpha1 else {n:before[n]+alpha*(after[n]-before[n]) for n in before})
            with torch.no_grad():v,_=value()
            finite=bool(torch.isfinite(v));trials.append(dict(alpha=alpha,loss=float(v) if finite else None))
            if finite and float(v)<lv-1e-7:accepted=True;break
        if not accepted:interface.restore(before);opt.load_state_dict(ob)
        st.update(accepted=accepted,trials=trials,
                  update_norm=math.sqrt(sum(float((p.detach()-before[n]).double().square().sum()) for n,p in named)))
    if failure:saved=interface.initial;selected=0
    interface.restore(saved)
    with torch.no_grad():lv,b=value()
    assert torch.equal(b.cpu(),result['states'][selected]['boxes'])
    result.update(boxes=b.cpu(),best_step=selected,failure=failure,best_restore_exact=True,
                  parameter_delta_norm=math.sqrt(sum(float((p.detach()-interface.initial[n]).double().square().sum()) for n,p in named)))
    interface.restore(interface.initial)
    return result
