"""Fixed finite cached query512 update; independent regionteacher fromparentlock."""
import numpy as np,torch
import torch.nn.functional as F
from interfaces import Rank4QueryResidual,anchored_span_scores

def aabb_iou(reference,other):
    a=np.asarray(reference,dtype=np.float64);b=np.asarray(other,dtype=np.float64)
    lo=np.maximum(a[:,:3]-a[:,3:6]/2,b[:,:3]-b[:,3:6]/2);hi=np.minimum(a[:,:3]+a[:,3:6]/2,b[:,:3]+b[:,3:6]/2);inter=np.maximum(hi-lo,0).prod(-1);den=a[:,3:6].prod(-1)+b[:,3:6].prod(-1)-inter
    valid=(a[:,3:6]>0).all(-1)&(b[:,3:6]>0).all(-1)
    return np.where(valid,np.clip(inter/np.maximum(den,1e-30),0,1),0)

def cached_views(full):
    reference=full[0]['boxes'];result=[];weights=[]
    for v,a in enumerate(full):
        n=len(reference);index=np.arange(n) if v==0 else a['aligned_to_full_view0'].astype('int64');assert np.array_equal(np.sort(index),np.arange(n))
        weight=np.ones(n) if v==0 else aabb_iou(reference,a['boxes'][index]);weights.append(weight)
        result.append({'q':torch.as_tensor(a['proj_queries']), 't':torch.as_tensor(a['proj_tokens']), 'L':torch.as_tensor(a['exact_logits']), 'mask':torch.as_tensor(a['noun_mask']).bool(), 's':torch.as_tensor(a['semantic_scores']), 'index':torch.as_tensor(index)})
    return result,torch.as_tensor(np.stack(weights),dtype=result[0]['q'].dtype)

def weighted_scores(scores,weights):return (torch.stack(scores)*weights).sum(0)/weights.sum(0)
def source_aug(views,weights):return weighted_scores([v['s'][v['index']] for v in views],weights)
def probability(scores):
    scores=scores.clamp_min(1e-12);return scores/scores.sum()

def run_query(views,weights,teacher,informative,control=False):
    base=source_aug(views,weights).detach();p0=probability(base).detach();pt=p0.clone() if control or not informative else torch.as_tensor(teacher,dtype=p0.dtype)
    target=torch.zeros_like(p0) if control or not informative else pt.clamp_min(1e-12).log()-p0.log()
    model=Rank4QueryResidual(dtype=views[0]['q'].dtype);optimizer=torch.optim.Adam(model.parameters(),lr=.01);outputs={};records=[]
    def forward():
        adapted=[model(v['q']) for v in views];scores=[anchored_span_scores(q,v['q'],v['t'],v['L'],v['mask'],v['s'])[v['index']] for q,v in zip(adapted,views)]
        aggregate=weighted_scores(scores,weights);p=probability(aggregate);residual=p.log()-p0.log()
        fit=(F.smooth_l1_loss(residual,target,reduction='none')*pt).sum();trust=torch.stack([(q-v['q']).square().sum(-1).mean() for q,v in zip(adapted,views)]).mean();return fit+.1*trust,aggregate,adapted,fit,trust
    for step in range(4):
        loss,scores,qs,fit,trust=forward()
        if step in [0,1,3]:outputs[step]=scores.detach().numpy().copy()
        if step==0:assert torch.equal(scores,base),'query0/TTAug exactbridge failed'
        if step==3:break
        previous=[p.detach().clone() for p in model.parameters()];qbefore=[q.detach().clone() for q in qs];before=float(loss.detach());optimizer.zero_grad(set_to_none=True);loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));nz=sum(int(p.grad is not None and torch.count_nonzero(p.grad)>0) for p in model.parameters());optimizer.step();proposal=[p.detach()-b for p,b in zip(model.parameters(),previous)];accepted=False;alpha=1.
        for halving in range(11):
            with torch.no_grad():
                for p,b,d in zip(model.parameters(),previous,proposal):p.copy_(b+alpha*d)
                after,_,qa,*_=forward();stepmax=max(float(torch.linalg.vector_norm(a-b,dim=-1).max()) for a,b in zip(qa,qbefore));totalmax=max(float(torch.linalg.vector_norm(a-v['q'],dim=-1).max()) for a,v in zip(qa,views))
            if float(after)<=before+1e-10 and stepmax<=.05+1e-7 and totalmax<=.1+1e-7:accepted=True;break
            alpha*=.5
        if not accepted:
            with torch.no_grad():
                for p,b in zip(model.parameters(),previous):p.copy_(b)
            alpha=0.
        with torch.no_grad():after,_,qa,*_=forward();stepmax=max(float(torch.linalg.vector_norm(a-b,dim=-1).max()) for a,b in zip(qa,qbefore));totalmax=max(float(torch.linalg.vector_norm(a-v['q'],dim=-1).max()) for a,v in zip(qa,views))
        records.append({'step':step+1,'loss_before':before,'loss_after':float(after),'fit_before':float(fit.detach()),'trust_before':float(trust.detach()),'gradient_norm':grad,'nonzero_gradient_tensors':nz,'parameter_delta_l2':float(model.parameter_delta_l2()),'max_embedding_step_L2':stepmax,'max_embedding_total_L2':totalmax,'proposal_scale':alpha,'backtrack_halvings':halving,'accepted_step':accepted})
    if control or not informative:assert all(np.array_equal(x,base.numpy()) for x in outputs.values()) and float(model.parameter_delta_l2())==0,'no-spatial noinformation drift'
    model.reset_episode()
    with torch.no_grad():reset=forward()[1]
    assert torch.equal(reset,base) and all(p.grad is None for p in model.parameters())
    return outputs,{'trace':records,'parameter_count':512,'zero_bridge_to_TTAug_exact':True,'reset_to_TTAug_exact':True,'control':control,'informative_teacher':bool(informative),'no_actual_source_model_loaded_or_updated':True}
