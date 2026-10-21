"""Seven-observation proposals: no held teacher-output or GT input exists here."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from scripts.propagation_v0.bank import occupancy, component_boxes, iou, soft_dice

EPS=1e-15

def affinity(features, valid, target, source):
    z=F.normalize(features.float(),dim=-1,eps=1e-12)
    ti=torch.where(valid[target])[0];si=torch.where(valid[source])[0]
    out=torch.zeros(features.shape[1],features.shape[1],dtype=torch.float64)
    if len(ti) and len(si):
        c=z[target,ti]@z[source,si].T
        s,k=c.topk(min(20,len(si)),dim=1,sorted=False)
        w=(s/.1).softmax(1).double();w/=w.sum(1,keepdim=True)
        out[ti[:,None].expand_as(k),si[k]]=w
    return out

def retained_teacher(logits7, boxes7, masks7, image_shape, grid_shape, valid7):
    """Arguments are already sliced to seven; full-eight selected summaries forbidden."""
    assert logits7.shape[0]==boxes7.shape[0]==masks7.shape[0]==7
    q=int(logits7.sigmoid().mean(0).max(-1).values.argmax())
    full=F.interpolate(masks7[:,q,None].float(),size=image_shape,mode='bilinear',align_corners=False).sigmoid()
    seeds=F.interpolate(full,size=grid_shape,mode='area').squeeze(1).double()
    seeds*=valid7.reshape_as(seeds)
    return q, boxes7[:,q].double().clone(),seeds

def build(source,frame_ids,features,valid,positions7,seeds7,teacher_boxes7,event,shape,affinity_cache=None):
    """Source-only original slot and geometric guard; fixed v0 three-slot construction."""
    assert len(positions7)==7 and len(set(positions7))==7
    assert len(seeds7)==len(teacher_boxes7)==7
    assert source.shape==(len(frame_ids),4) and (source[:,2:]>0).all()
    T=len(source);H,W=shape;N=H*W
    assert features.shape==(T,N,256) and valid.shape==(T,N)
    seeds7=seeds7.double();teacher_boxes7=teacher_boxes7.double()
    positions=[int(x) for x in positions7];order=sorted(range(7),key=lambda j:positions[j])
    observed=torch.zeros(T,dtype=torch.bool);observed[positions]=True
    side=torch.zeros(T,2,H,W,dtype=torch.float64);reliability=torch.zeros(T,2,dtype=torch.float64)
    available=torch.zeros(T,2,dtype=torch.bool);parents=torch.full((T,2),-1,dtype=torch.long)
    pairs={} if affinity_cache is None else affinity_cache
    def A(t,s):
        if (t,s) not in pairs:pairs[t,s]=affinity(features,valid,t,s)
        return pairs[t,s]
    for t in range(T):
        if observed[t]:j=positions.index(t);neighbors=(j,j)
        else:
            past=[j for j in order if positions[j]<t];future=[j for j in order if positions[j]>t]
            neighbors=(past[-1] if past else None,future[0] if future else None)
        for k,j in enumerate(neighbors):
            if j is None:continue
            a=positions[j];parents[t,k]=a;available[t,k]=True
            if a==t:p=seeds7[j].flatten();r=1.
            else:
                p=A(t,a)@seeds7[j].flatten()
                r=float(soft_dice(A(a,t)@p,seeds7[j].flatten()))
            side[t,k]=p.reshape(H,W);reliability[t,k]=r
    targets=(side*reliability[:,:,None,None]).sum(1)/reliability.sum(1).clamp_min(EPS)[:,None,None]
    targets*=valid.reshape(T,H,W)
    confidence=reliability.sum(1)/available.sum(1).clamp_min(1)
    active=event & (confidence>=.1) & valid.any(1) & (targets.flatten(1).sum(1)>EPS)
    boxes=source.double()[:,None].expand(T,3,4).clone()
    okay=torch.zeros(T,3,dtype=torch.bool);okay[:,0]=True
    origin=torch.full((T,3),-1,dtype=torch.long)
    for t in range(T):
        if not active[t]:continue
        props=[]
        if observed[t]:
            j=positions.index(t);props.append((1,float(seeds7[j].sum()),-2,teacher_boxes7[j]))
        for k in range(2):
            if not available[t,k]:continue
            for c in component_boxes(side[t,k],valid[t].reshape(H,W)):
                props.append((0,float(reliability[t,k])*c['mass'],k,c['box']))
        props.sort(key=lambda p:(-p[0],-p[1],p[2],tuple(p[3].tolist())))
        slot=1
        for _,_,org,b in props:
            if any(float(iou(b,boxes[t,a]))>=.7 for a in range(slot)):continue
            boxes[t,slot]=b;okay[t,slot]=True;origin[t,slot]=org;slot+=1
            if slot==3:break
    occ,ox,oy=occupancy(boxes,H,W)
    px,py=targets.amax(1),targets.amax(2)
    sx=2*(ox*px[:,None]).sum(-1)/(ox.sum(-1)+px.sum(-1)[:,None]).clamp_min(EPS)
    sy=2*(oy*py[:,None]).sum(-1)/(oy.sum(-1)+py.sum(-1)[:,None]).clamp_min(EPS)
    u=.5*(sx+sy)
    dz=torch.cat(((boxes[...,:2]-source[:,None,:2])/source[:,None,2:],boxes[...,2:].log()-source[:,None,2:].log()),-1)
    guard=-torch.expm1(-(torch.linalg.vector_norm(dz,dim=-1)/2).square())
    assert torch.equal(boxes[:,0],source.double()) and not okay[~active,1:].any()
    return dict(boxes=boxes,valid=okay,origin=origin,projection=u,guard=guard,confidence=confidence,active=active,observed=observed,parents=parents,GT_read=False)
