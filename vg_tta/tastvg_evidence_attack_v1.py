"""Prediction-preserving visual-H perturbations and separate evidence metrics."""
import math
import torch
from torch.nn import functional as F
from vg_tta.tastvg_evidence_capture_v1 import combined, visual_mask


def iou(a, b):
    from methods.decota_final_simplified_v1.objectives import xyxy
    a, b = xyxy(a.double()), xyxy(b.double())
    inter = (torch.minimum(a[...,2:],b[...,2:])-torch.maximum(a[...,:2],b[...,:2])).clamp_min(0).prod(-1)
    union = (a[...,2:]-a[...,:2]).clamp_min(0).prod(-1)+(b[...,2:]-b[...,:2]).clamp_min(0).prod(-1)-inter
    return inter / union.clamp_min(1e-30)


def preservation(base, pred, ids):
    q = iou(base['boxes'].detach().cpu(), pred['boxes'].detach().cpu())
    a,b=base['indices']; c,d=pred['indices']
    same = base['physical_interval']==pred['physical_interval']
    mean = float(q[a:b+1].mean())
    left=max(a,c); right=min(b,d)
    numerator=float(q[left:right+1].sum()) if right>=left else 0.
    # Corrected sparse frame-count self-vIoU, as in this panel's native scorer.
    union=max(b,d)-min(a,c)+1
    self_v=numerator/union
    return dict(interval_exact=same, mean_native_interval_box_iou=mean,
                self_vIoU=self_v, strict=bool(same and mean>=.95),
                self_tIoU=max(0,min(base['physical_interval'][1],pred['physical_interval'][1])-max(base['physical_interval'][0],pred['physical_interval'][0]))/
                (max(base['physical_interval'][1],pred['physical_interval'][1])-min(base['physical_interval'][0],pred['physical_interval'][0])))


def jsd_logits(a,b):
    # Normalize sigmoid TTS scores across time; no attention-semantic assertion.
    p=a.reshape(-1).double().sigmoid().clamp_min(1e-12);p=p/p.sum()
    q=b.reshape(-1).double().sigmoid().clamp_min(1e-12);q=q/q.sum()
    m=(p+q)/2
    return ((p*(p/m).log()).sum()+(q*(q/m).log()).sum())/2


def cosdist(a,b):
    a,b=a.double().flatten(),b.double().flatten()
    return 1-F.cosine_similarity(a,b,dim=0,eps=1e-20)


def evidence_metrics(base, now):
    values={}; objective=[]
    for key in ('TTS_app','TTS_motion'):
        v=jsd_logits(base[key],now[key]); values[key+'_JSD']=v
        objective.append(v/math.log(2))
    temporal=torch.stack(objective).mean()
    spatial=[]; queries=[]
    for stage in (1,2):
        key=f'selected_stage{stage}'; old=base[key]; new=now[key]
        common=sorted(set(old)&set(new)); union=set(old)|set(new)
        values[key+'_jaccard']=len(common)/len(union) if union else 1.
        values[key+'_common_frames']=len(common)
        for branch in ('app','motion'):
            key=f'ASA{stage}_{branch}'
            if not common:
                values[key+'_cosine_drift']=None; values[key+'_top20_iou']=None
                continue
            a=base[key][[old.index(i) for i in common]]
            b=now[key][[new.index(i) for i in common]]
            ds=1-F.cosine_similarity(a.double(),b.double(),dim=1,eps=1e-20)
            drift=ds.mean(); values[key+'_cosine_drift']=drift; spatial.append(drift)
            k=max(1,math.ceil(a.shape[1]*.2)); ma=torch.zeros_like(a,dtype=torch.bool);mb=ma.clone()
            ma.scatter_(1,a.topk(k,dim=1).indices,True);mb.scatter_(1,b.topk(k,dim=1).indices,True)
            values[key+'_top20_iou']=((ma&mb).sum(1).double()/(ma|mb).sum(1)).mean()
        for branch in ('s','t'):
            key=f'Q{branch}{stage}'; v=cosdist(base[key],now[key]);values[key+'_cosine_drift']=v;queries.append(v)
    spatial_mean=torch.stack(spatial).mean() if spatial else temporal*0
    query_mean=torch.stack(queries).mean()
    # Explicit equal-family optimizer surrogate, NOT a reported attention score.
    return values, (temporal+spatial_mean+query_mean)/3


def metrics_pair(base, now):
    vals=[]; losses=[]
    for b,n in zip(base,now):
        v,l=evidence_metrics(b,n);vals.append(v);losses.append(l)
    return vals,torch.stack(losses).mean()


def output_loss(base_evidence, now, base_boxes, boxes):
    kl=[]
    for b,n in zip(base_evidence,now):
        p=b['pred_sted'].double().softmax(1).detach()
        kl.append((p*(p.clamp_min(1e-30).log()-n['pred_sted'].double().log_softmax(1))).sum(1).mean())
    return torch.stack(kl).mean()+(1-iou(base_boxes.to(boxes.device),boxes)).mean()


def norm(xs):
    return torch.stack([x.double().square().sum() for x in xs]).sum().sqrt()


def project(ds,masks,cap):
    ds=[d*m for d,m in zip(ds,masks)]
    length=norm(ds);scale=min(1.,cap/max(float(length),1e-30))
    return [(d*scale).detach() for d in ds]


def run_attack(model, data, rho, config, guard=lambda:None):
    from methods.decota_final_simplified_v1.tensors import detached
    views=data['views']; fields=[v['H'] for v in views]
    masks=[visual_mask(h,v['info']['fea_map_size'][0]*v['info']['fea_map_size'][1]) for h,v in zip(fields,views)]
    stock=float(norm([h*m for h,m in zip(fields,masks)])); cap=rho*stock
    base=data['evidence'];basepred=data['prediction'];ids=data['frame_ids'];records=data['records']
    seed=config['seed']+int(rho*1e6)
    gen=torch.Generator(device=fields[0].device).manual_seed(seed)
    rnd=[torch.randn(h.shape,device=h.device,generator=gen)*m for h,m in zip(fields,masks)]
    rnd=[d*(cap/max(float(norm(rnd)),1e-30)) for d in rnd]
    with torch.no_grad():
        re,rb,rp=combined(model,views,[h+d for h,d in zip(fields,rnd)],records,ids)
        rv,rl=metrics_pair(base,re)
        random=dict(prediction=rp,evidence=detached(re,'cpu'),metrics=detached(rv,'cpu'),preservation=preservation(basepred,rp,ids),surrogate=float(rl),delta_norm=float(norm(rnd)))
    ds=[d*.01 for d in rnd];path=[]
    best=dict(step=-1,delta=[torch.zeros_like(h).cpu() for h in fields],evidence=detached(base,'cpu'),prediction=detached(basepred,'cpu'),
              metrics=detached(metrics_pair(base,base)[0],'cpu'),preservation=preservation(basepred,basepred,ids),surrogate=0.)
    first_gradient=None
    for step in range(config['steps']+1):
        guard();ds=[d.detach().requires_grad_(True) for d in ds]
        ev,boxes,pred=combined(model,views,[h+d for h,d in zip(fields,ds)],records,ids)
        vv,drift=metrics_pair(base,ev);op=output_loss(base,ev,basepred['boxes'],boxes)
        hard=preservation(basepred,pred,ids)
        record=dict(step=step,evidence_surrogate=float(drift.detach()),prediction_loss=float(op.detach()),
                    delta_norm=float(norm(ds).detach()),relative_norm=float(norm(ds).detach())/stock,
                    preservation=hard,metrics=detached(vv,'cpu'))
        path.append(record)
        if hard['strict'] and float(drift.detach())>best['surrogate']:
            best=dict(step=step,delta=detached(ds,'cpu'),evidence=detached(ev,'cpu'),prediction=pred,
                      metrics=record['metrics'],preservation=hard,surrogate=record['evidence_surrogate'])
        if step==config['steps']:
            terminal=dict(prediction=pred,evidence=detached(ev,'cpu'),metrics=record['metrics'],preservation=hard)
            break
        loss=config['prediction_weight']*op-drift
        gg=torch.autograd.grad(loss,ds)
        gg=[g*m for g,m in zip(gg,masks)]
        assert all(torch.isfinite(g).all() for g in gg)
        if step==0:first_gradient=detached(gg,'cpu')
        length=float(norm(gg));alpha=2*cap/config['steps']
        record['gradient_norm']=length
        ds=project([d.detach()-alpha*g.detach()/max(length,1e-30) for d,g in zip(ds,gg)],masks,cap)
        del ev,boxes,pred,loss,gg,op,drift
    assert all(torch.count_nonzero(d*(1-m.cpu()))==0 for d,m in zip(best['delta'],masks))
    return dict(rho=rho,stock_visual_norm=stock,cap=cap,selected=best,terminal=terminal,
                random=random,path=path,initial_objective_gradient=first_gradient,
                GT_read=False,text_delta_exact_zero=True,backward_steps=config['steps'])
