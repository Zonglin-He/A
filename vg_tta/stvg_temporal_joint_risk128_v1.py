"""Metric-level temporal decisions with a frozen spatial quality proxy.
This is pseudo-world joint risk, not event probability or new OPD.
"""
import numpy as np
from vg_tta.stvg_posterior_decision_tree_v1 import expected_iou_fast,final_mbr


def xyxy(boxes):
    b=np.asarray(boxes,float);return np.maximum(0,np.c_[b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2])


def quality(ids,boxes,teacher_ids,teacher_boxes,frame_count):
    ids=np.asarray(ids,int);ti=np.asarray(teacher_ids,int);t=np.arange(frame_count);b=np.asarray(boxes);tb=np.asarray(teacher_boxes)
    if len(ti)<2 or tb.shape!=(len(ti),4)or not np.isfinite(tb).all():return np.ones(frame_count),dict(missing_teacher=True,known_frames=0)
    s=xyxy(b);ref=xyxy(tb);p=np.column_stack([np.interp(t,ids,s[:,k])for k in range(4)]);r=np.column_stack([np.interp(t,ti,ref[:,k])for k in range(4)]);inter=np.maximum(0,np.minimum(p[:,2:],r[:,2:])-np.maximum(p[:,:2],r[:,:2])).prod(1);pa=np.maximum(0,p[:,2:]-p[:,:2]).prod(1);ra=np.maximum(0,r[:,2:]-r[:,:2]).prod(1);union=pa+ra-inter;q=np.divide(inter,union,out=np.zeros_like(inter),where=union>0)
    known=(t>=ti[0])&(t<=ti[-1])&(ra>0);q[~known]=1.;q[(t<ids[0])|(t>ids[-1])]=0
    assert np.isfinite(q).all()and q.min()>=0 and q.max()<=1+1e-12
    return q,dict(missing_teacher=False,known_frames=int(known.sum()),outside_teacher_support=int((~known).sum()),constant=bool(np.ptp(q)<=1e-12),mean=float(q.mean()))


def joint_utility(ids,q,quality_curve):
    """Exact O(n^3) expectation of spatial-weighted tube IoU, true interval~q.
    Four endpoint-order regions; overlap numerators integrate spatial proxy,
    union denominators remain physical time. No GT or bbox optimization.
    """
    s=np.asarray(ids,dtype=np.float64);e=s+1;n=len(s);q=np.asarray(q,float);r=np.asarray(quality_curve,float);assert np.isfinite(r).all()and(r>=0).all()and(r<=1+1e-12).all()and len(r)>ids[-1]
    assert q.shape==(n,n)and(q>=0).all()and abs(q.sum()-1)<1e-10
    legal=np.triu(np.ones((n,n),bool),1);assert not q[~legal].any();F=np.r_[0.,np.cumsum(r)];fs=F[s.astype(int)];fe=F[e.astype(int)];lengths=e[None]-s[:,None];QL=q*np.maximum(fe[None]-fs[:,None],0);H=np.divide(q,lengths,out=np.zeros_like(q),where=legal)
    inside=QL[::-1].cumsum(0)[::-1].cumsum(1);outer=H.cumsum(0)[:,::-1].cumsum(1)[:,::-1];rowQ=q.cumsum(1);rowF=(q*fe[None]).cumsum(1);colQ=q.cumsum(0);colF=(q*fs[:,None]).cumsum(0);u=np.full((n,n),-np.inf)
    for a in range(n-1):
        bs=np.arange(a+1,n);L=e[bs]-s[a];value=inside[a,bs]/L
        if a>0:
            value+=(fe[bs]-fs[a])*np.r_[outer[a-1,bs[:-1]+1],0.];lo=int(np.searchsorted(e,s[a],side='right'));bQ=rowQ[:a,bs]-(rowQ[:a,lo-1,None]if lo else 0);bF=rowF[:a,bs]-(rowF[:a,lo-1,None]if lo else 0);value+=((bF-fs[a]*bQ)/(e[bs][None]-s[:a,None])).sum(0)
        cq=colQ[bs]-(colQ[a-1][None]if a else 0);cf=colF[bs]-(colF[a-1][None]if a else 0);mask=np.arange(n)[None]>bs[:,None];num=fe[bs,None]*cq-cf;denom=e[None]-s[a];term=np.divide(num,denom,out=np.zeros_like(num),where=mask);value+=term.sum(1);u[a,bs]=value
    assert np.isfinite(u[legal]).all()and u[legal].min()>-1e-12 and u[legal].max()<1+1e-10;u[legal]=np.maximum(u[legal],0);return u


def decisions(ids,q,r):
    base=final_mbr(ids,q);u=joint_utility(ids,q,r);flat=np.ptp(r)<=1e-12
    if flat:iv=base['interval']
    else:
        a,b=np.unravel_index(np.argmax(u),u.shape);iv=[int(ids[a]),int(ids[b])+1]
    weight=q*np.where(np.isfinite(u),u,0);total=weight.sum();target=q.copy()if total<=0 else weight/total
    return dict(interval=iv,utility=u,quality_flat_noop=bool(flat),target=target,target_interval=final_mbr(ids,target)['interval'],teacher_target='prior times nonnegative pseudo-tube utility, normalized; no temperature')


def brute(ids,q,r,picks=None):
    ii,jj=np.triu_indices(len(ids),1);ids=np.array(ids,int);ivs=np.stack([ids[ii],ids[jj]+1],1);F=np.r_[0.,np.cumsum(r)];out=[]
    for k in range(len(ivs))if picks is None else picks:
        a,b=ivs[k];c,d=ivs[:,0],ivs[:,1];lo=np.maximum(a,c);hi=np.minimum(b,d);v=np.where(hi>lo,F[np.maximum(hi,lo)]-F[lo],0);overlap=np.maximum(hi-lo,0);union=b-a+d-c-overlap;out.append(float((q[ii,jj]*v/union).sum()))
    return np.array(out)


def properties():
    rng=np.random.default_rng(41);checks=0;err=0
    for n in [4,7,11]:
        ids=np.cumsum(rng.integers(1,5,n)).tolist();q=np.triu(rng.random((n,n)),1);q/=q.sum();r=rng.random(ids[-1]+1);u=joint_utility(ids,q,r);ii,jj=np.triu_indices(n,1);actual=brute(ids,q,r);delta=float(abs(u[ii,jj]-actual).max());assert delta<1e-12;checks+=len(ii);err=max(err,delta)
        for c in [0.,.37,1.]:
            d=decisions(ids,q,np.full(len(r),c));assert d['interval']==final_mbr(ids,q)['interval'];assert abs(d['utility'][ii,jj]-c*expected_iou_fast(ids,q)[ii,jj]).max()<1e-12
    ids=[0,5,10];b=np.tile([.4,.5,.2,.4],(3,1));r,meta=quality(ids,b,ids,b,11);assert np.array_equal(r,np.ones(11))and meta['known_frames']==11
    r,meta=quality(ids,b,[],np.empty((0,4)),11);assert meta['missing_teacher']and np.array_equal(r,np.ones(11))
    return dict(status='pass',independent_weighted_tube_utilities=checks,max_error=err,flat_quality_original_time_MBR_exact=True,ones_recovers_temporal_utility=True,identical_spatial_boxes_quality_one=True)
