"""Physical interval opinion pooling; original MBR and observation logic unchanged."""
import time
import numpy as np
from stvg.temporal import probabilities,mass_table,final_mbr

def legal_q(q,U):
    q=np.asarray(q,dtype=np.float64)
    assert q.shape==(len(U),len(U))and np.isfinite(q).all()and(q>=0).all()
    assert not q[~np.triu(np.ones_like(q,dtype=bool),1)].any()and abs(q.sum()-1)<1e-10
    return q

def map_view(grid,U):
    U=list(map(int,U));assert all(b>a for a,b in zip(U,U[1:]));index={x:i for i,x in enumerate(U)}
    ids=np.asarray(grid['ids'],np.int64);ij=np.asarray(grid['ij']);p=np.asarray(grid['p'],np.float64)
    assert ij.shape==(2,len(p))and np.isfinite(p).all()and(p>=0).all()and abs(p.sum()-1)<1e-12
    assert (ij[0]<ij[1]).all()and all(int(x)in index for x in ids)
    start=np.asarray([index[int(x)]for x in ids[ij[0]]]);end=np.asarray([index[int(x)]for x in ids[ij[1]]])
    q=np.zeros((len(U),len(U)),np.float64);np.add.at(q,(start,end),p)
    # The column endpoint maps to U[b]+1, exactly as original final_mbr.
    return legal_q(q,U)

def mix_views(grids,U):
    assert len(grids)>0
    return legal_q(np.mean([map_view(g,U)for g in grids],axis=0),U)

def view_stats(grid):
    ids=np.asarray(grid['ids'],np.float64);ij=np.asarray(grid['ij']);p=np.asarray(grid['p'],np.float64)
    starts=ids[ij[0]];ends=ids[ij[1]]+1
    es=float(p@starts);ee=float(p@ends)
    return dict(expected_length_frames=float(p@(ends-starts)),expected_start_frame=es,expected_end_frame=ee,
        start_std_frames=float(np.sqrt(p@((starts-es)**2))),end_std_frames=float(np.sqrt(p@((ends-ee)**2))))

def hull_stats(grids,U,q):
    assert len(grids)==2;v=[view_stats(g)for g in grids]
    marginal=[]
    for g in grids:
        n=len(g['ids']);marginal.append([np.bincount(g['ij'][a],weights=g['p'],minlength=n)for a in(0,1)])
    a,b=[np.asarray(g['ids'],np.float64)for g in grids];distance=np.abs(a[:,None]-b[None,:])
    ds=float((distance*marginal[0][0][:,None]*marginal[1][0][None,:]).sum())
    de=float((distance*marginal[0][1][:,None]*marginal[1][1][None,:]).sum())
    U=np.asarray(U,np.float64);lengths=U[None,:]+1-U[:,None];h=float((q*lengths).sum())
    mean=.5*(v[0]['expected_length_frames']+v[1]['expected_length_frames']);extra=.5*(ds+de)
    error=abs(h-mean-extra);assert error<1e-7,'Physical hull length identity failed'
    return dict(views=v,mean_view_length_frames=mean,hull_expected_length_frames=h,
        endpoint_disagreement_frames=extra,start_absolute_disagreement_frames=ds,end_absolute_disagreement_frames=de,
        identity_abs_error_frames=error)

def global_distributions(records,layer_logits,ids):
    assert len(layer_logits)==2 and all(z.shape[0]==6 for z in layer_logits)
    views=[];hulls=[];statistics=[];hull_seconds=mix_seconds=stats_seconds=0.
    for layer in range(6):
        g=probabilities(records,[z[layer]for z in layer_logits]);views.append(g)
        tick=time.perf_counter();U,q=mass_table(g);hull_seconds+=time.perf_counter()-tick;assert U==list(ids)
        hulls.append(q);tick=time.perf_counter();statistics.append(hull_stats(g,ids,q));stats_seconds+=time.perf_counter()-tick
    tick=time.perf_counter();mixture=mix_views([g for layer in views for g in layer],ids);mix_seconds=time.perf_counter()-tick
    return dict(views=views,hull_layers=np.stack(hulls),hull=legal_q(np.mean(hulls,axis=0),ids),mix=mixture,statistics=statistics,
        timing=dict(hull_construct_seconds=hull_seconds,mix_construct_seconds=mix_seconds,diagnostic_stats_seconds=stats_seconds))

def choose(ids,q,rule='mbr'):
    q=legal_q(q,ids)
    if rule=='mbr':return final_mbr(ids,q)['interval']
    if rule!='map':raise ValueError('Fixed rule must be mbr or map')
    legal=np.triu(np.ones_like(q,dtype=bool),1);i,j=divmod(int(np.argmax(np.where(legal,q,-np.inf))),len(ids))
    return [int(ids[i]),int(ids[j])+1]
