"""F24 isolated diagnostic operators; no label reader or production mutation."""
import copy
import math
import time
import numpy as np
import torch
from vg_tta.temporal_optimizer_probe_v1 import cpu_state, logits, task
from vg_tta.fullspan_tta import fullspan_prior_loss
from vg_tta.decota_tastvg_episode_v1 import fitted_merge, native_view_indices
from vg_tta.native_coverage_calibration_v1 import native_at_length
from methods.decota_v1.api import decode


def temporal_snapshot(head, inputs, records, ids, base, fp32=False, gradient=False):
    head.zero_grad(set_to_none=True)
    zs=logits(head,inputs,fp32);loss=torch.stack([fullspan_prior_loss(z) for z in zs]).mean()
    # Production differentiable zero anchor.
    loss=loss+next(head.parameters()).float().sum()*0.
    if gradient:loss.backward()
    native=list(base['predicted_indices']);tmp=list(fitted_merge(zs,records,ids))
    merged=torch.stack([zs[i%2][0,i//2] for i in range(len(ids))]).detach().cpu()
    final=list(decode(base['temporal_logits'],tmp,ids,native_indices=native)['indices'])
    combinations={}
    for en,extent in [('native',native),('adapted',tmp)]:
        for sn,sc in [('native',base['temporal_logits']),('adapted',merged)]:
            v=native_at_length(sc,ids,reference=extent)
            assert v['length_mismatch']==0
            combinations[en+'_'+sn]=dict(v,indices=list(v['indices']))
    grads={n:p.grad.detach().cpu().clone() for n,p in head.named_parameters() if p.grad is not None} if gradient else {}
    return dict(loss=float(loss.detach()),logits=[z.detach().cpu() for z in zs],
        offset_losses=[float(fullspan_prior_loss(z).detach()) for z in zs],
        endpoint_probabilities=[[float(z.float().softmax(1)[0,0,0]),float(z.float().softmax(1)[0,-1,1])] for z in zs],
        offset_indices=[list(native_view_indices(z)) for z in zs],temporary=tmp,final=final,
        combinations=combinations,head_state=cpu_state(head),gradients=grads,
        gradient_norm=math.sqrt(sum(float(g.double().square().sum()) for g in grads.values())),
        fp32=fp32,merged_logits=merged)


def optimizer_cpu(opt):
    out=copy.deepcopy(opt.state_dict())
    for state in out['state'].values():
        for k,v in state.items():
            if torch.is_tensor(v):state[k]=v.detach().cpu()
    return out


def temporal_probe(source_head,inputs,records,ids,base,lr):
    start=time.perf_counter();head=copy.deepcopy(source_head).float().eval().requires_grad_(True)
    opt=torch.optim.AdamW(head.parameters(),lr=lr,eps=1e-4,weight_decay=0.)
    path=[];branches=[]
    def snap(fp32=False,gradient=False):return temporal_snapshot(head,inputs,records,ids,base,fp32,gradient)
    for k in range(6):
        before=snap(gradient=True);before['step']=k;before['optimizer']=optimizer_cpu(opt)
        path.append(before)
        if k==5:break
        opt.step();after=cpu_state(head);opt_after=copy.deepcopy(opt.state_dict())
        displacement={n:after[n]-before['head_state'][n] for n in after}
        before['next_update_norm']=math.sqrt(sum(float(d.double().square().sum()) for d in displacement.values()))
        before['gradient_dot_next_update']=sum(float((before['gradients'][n].double()*d.double()).sum()) for n,d in displacement.items())
        br={'step_from':k,'line':[]}
        for alpha in [0.,.125,.25,.5,1.]:
            head.load_state_dict({n:before['head_state'][n]+alpha*d for n,d in displacement.items()})
            s=snap();s.pop('head_state');s['alpha']=alpha;br['line'].append(s)
        for name,fp in [('reset_first',False),('fp32_step',True)]:
            head.load_state_dict(before['head_state']);opt.load_state_dict(copy.deepcopy(before['optimizer']))
            if name=='reset_first':
                for st in opt.state.values():st['exp_avg'].zero_()
            opt.zero_grad(set_to_none=True);value=task(head,inputs,fp);value.backward();opt.step()
            s=snap(fp);s['before_loss_same_precision']=float(task_from_state(source_head,before['head_state'],inputs,fp))
            s['parameter_dtype']=str(next(head.parameters()).dtype)
            s['moment_dtypes']=sorted({str(v.dtype) for st in opt.state.values() for k2,v in st.items() if torch.is_tensor(v) and k2!='step'})
            s.pop('head_state');br[name]=s
        branches.append(br);head.load_state_dict(after);opt.load_state_dict(opt_after)
    best=min(range(6),key=lambda k:(path[k]['loss'],k));head.load_state_dict(path[best]['head_state'])
    restored=snap();assert all(torch.equal(a,b) for a,b in zip(restored['logits'],path[best]['logits']))
    # Separate full head-FP32 trajectory; visual input features are the same captures.
    head.load_state_dict(source_head.state_dict());opt=torch.optim.AdamW(head.parameters(),lr=lr,eps=1e-4,weight_decay=0.)
    fp=[]
    for k in range(6):
        state=snap(True,True);state.pop('head_state');state.pop('gradients');state['step']=k;fp.append(state)
        if k<5:opt.step()
    return dict(path=path,branches=branches,best_step=best,best_restore_exact=True,fp32_path=fp,
        parameters=sum(p.numel() for p in head.parameters()),seconds=time.perf_counter()-start,GT_used=False)


def task_from_state(source,state,inputs,fp):
    h=copy.deepcopy(source).float().eval();h.load_state_dict(state)
    with torch.no_grad():v=float(task(h,inputs,fp))
    del h
    return v


def length_reward(z,ids,strength):
    z=torch.as_tensor(z).cpu().double().reshape(-1,2);n=len(ids);x=torch.tensor(ids)
    length=(x[None]+1-x[:,None]).double()/(ids[-1]+1-ids[0])
    lp=z.log_softmax(0);value=lp[:,0,None]+lp[None,:,1]+strength*length
    value=value.masked_fill(~torch.triu(torch.ones(n,n,dtype=torch.bool),diagonal=1),-torch.inf)
    a=int(value.argmax());return [a//n,a%n]


def regate(probes,score=.35,margin=.05):
    out=copy.deepcopy(probes)
    for p in out:
        s=np.asarray(p['target_scores']);order=np.argsort(-s,kind='stable')
        gap=float(s[order[0]]-s[order[1]]) if len(s)>1 else (float(s[0]) if len(s) else 0.)
        p.update(accepted=bool(len(s) and s[order[0]]>=score and gap>=margin),margin=gap)
    return out


class SpatialInterface:
    def __init__(self,model,caches,n,scope):
        self.decoder=copy.deepcopy(model.ground_decoder.decoder).eval().requires_grad_(False)
        self.delta=torch.zeros(256,device='cuda',requires_grad=True);self.caches=caches;self.n=n
        self.named=[('query_residual',self.delta)]
        if scope=='query_ln':
            last=len(self.decoder.layers)-1
            for norm in ['norm1','norm3','norm4']:
                for name,p in getattr(self.decoder.layers[last],norm).named_parameters():
                    p.requires_grad_(True);self.named.append((f'layers.{last}.{norm}.{name}',p))
        elif scope!='query':raise ValueError(scope)
        self.initial=self.state()

    def state(self):return {n:p.detach().clone() for n,p in self.named}
    def restore(self,state):
        with torch.no_grad():
            for n,p in self.named:p.copy_(state[n].to(p.device));p.grad=None
    def values(self,allowed=None):
        from vg_tta.st_causal_audit_v2 import KeyIntervention
        from contextlib import nullcontext
        views=[];audit=[]
        with torch.autocast('cuda',dtype=torch.float16):
            for off,c in enumerate(self.caches):
                kw=dict(c);kw['query_tgt']=kw['query_tgt']+self.delta.to(kw['query_tgt'].dtype)[None,None,:]
                mask=None if allowed is None else np.asarray(allowed[off],bool)
                ctx=KeyIntervention([l.self_attn for l in self.decoder.layers],allowed=mask) if mask is not None and mask.any() else nullcontext()
                with ctx:views.append(self.decoder(**kw)[-1,0])
                audit.append(dict(offset=off,allowed=None if mask is None else int(mask.sum()),empty_fallback=bool(mask is not None and not mask.any())))
        return torch.stack([views[i%2][i//2] for i in range(self.n)]).float(),audit


def spatial_fit(interface,base,anchors,lr,budgets=(3,10)):
    from vg_tta.metrics import generalized_box_iou_aligned_cxcywh
    interface.restore(interface.initial);start=time.perf_counter();named=interface.named
    opt=torch.optim.Adam([p for _,p in named],lr=lr,eps=1e-8)
    result=dict(lr=lr,parameter_count=sum(p.numel() for _,p in named),parameters={n:p.numel() for n,p in named},
        states=[],predictions={},failure=None,anchors=len(anchors),GT_reader=False)
    if not anchors:
        result.update(seconds=0.,skipped='no_anchors');result['predictions']={str(b):base for b in budgets};return result
    pos=torch.tensor([a['position'] for a in anchors],device='cuda');target=torch.tensor([a['box'] for a in anchors],device='cuda')
    def value():
        b,_=interface.values();pred=b[pos]
        data=(5*(pred-target).abs().sum(-1)+2*(1-generalized_box_iou_aligned_cxcywh(pred,target))).mean()
        penalty=sum((p-interface.initial[n]).square().sum() for n,p in named)*1e-4
        return data+penalty,b
    with torch.no_grad():initial,zero=value()
    assert torch.equal(zero.cpu(),base),'spatial delta0 mismatch'
    best=float(initial);saved=interface.state();selected=0
    for k in range(max(budgets)+1):
        opt.zero_grad(set_to_none=True);loss,b=value()
        if not torch.isfinite(loss):result['failure']='nonfinite_loss';break
        lv=float(loss.detach())
        if lv<best:best=lv;saved=interface.state();selected=k
        st=dict(step=k,loss=lv,best_loss=best,best_step=selected)
        if k in budgets:
            current=interface.state();interface.restore(saved)
            with torch.no_grad():lv2,bb=value()
            assert abs(float(lv2)-best)<1e-5
            result['predictions'][str(k)]=bb.detach().cpu();interface.restore(current)
            loss,b=value()  # restore changes tensor versions; use a fresh graph.
        if k==max(budgets):result['states'].append(st);break
        loss.backward();gn={n:float(p.grad.norm()) if p.grad is not None else None for n,p in named};st['gradient_norms']=gn
        if any(v is None or not math.isfinite(v) for v in gn.values()):result['failure']='missing_nonfinite_gradient';result['states'].append(st);break
        before=interface.state();optbefore=copy.deepcopy(opt.state_dict());opt.step();after=interface.state()
        trials=[];accepted=False
        for alpha in [1.,.5,.25,.125]:
            interface.restore({n:before[n]+alpha*(after[n]-before[n]) for n in before})
            with torch.no_grad():v,_=value()
            finite=bool(torch.isfinite(v));trials.append(dict(alpha=alpha,loss=float(v) if finite else None))
            if finite and float(v)<lv-1e-7:accepted=True;break
        if not accepted:interface.restore(before);opt.load_state_dict(optbefore)
        st.update(trials=trials,accepted=accepted,update_norm=math.sqrt(sum(float((p.detach()-before[n]).double().square().sum()) for n,p in named)))
        result['states'].append(st)
    for k in budgets:
        if str(k) not in result['predictions']:result['predictions'][str(k)]=base.clone()
    result['seconds']=time.perf_counter()-start;interface.restore(interface.initial)
    return result
