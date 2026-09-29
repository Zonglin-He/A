"""CPU array statistics of a complete oracle field; no fitting or model call."""
import numpy as np


def field_statistics(array):
    a=np.asarray(array,dtype=np.float64)
    if a.ndim!=4 or min(a.shape)<1 or not np.isfinite(a).all():
        raise ValueError('Complete finite THWC field required')
    x=a.reshape(-1,a.shape[-1]);energy=float(np.sum(x*x));n=len(x)
    global_mean=x.mean(0);temporal_mean=a.mean((1,2))
    eg=float(n*np.sum(global_mean*global_mean))
    et=float(a.shape[1]*a.shape[2]*np.sum(temporal_mean*temporal_mean))
    eig=np.linalg.eigvalsh(x.T@x)[::-1]
    assert eig.min()>=-max(energy,1e-30)*1e-10
    spectrum=np.maximum(eig,0)
    metrics=dict(global_energy=eg/energy if energy else None,
        temporal_energy=et/energy if energy else None,
        temporal_extra_energy=(et-eg)/energy if energy else None,
        global_cosine=np.sqrt(eg/energy) if energy and eg else None,
        temporal_cosine=np.sqrt(et/energy) if energy and et else None)
    for k in (1,4,8,16,32):
        metrics['rank%d_energy'%k]=float(spectrum[:k].sum()/energy) if energy else None
    neighbors={}
    for axis,name in enumerate(('temporal','vertical','horizontal')):
        left=np.take(a,range(max(0,a.shape[axis]-1)),axis=axis).reshape(-1,a.shape[-1])
        right=np.take(a,range(1,a.shape[axis]),axis=axis).reshape(-1,a.shape[-1])
        dot=np.sum(left*right,axis=1);den=np.sqrt(np.sum(left*left,axis=1)*np.sum(right*right,axis=1))
        valid=den>0;c=dot[valid]/den[valid]
        diff=float(np.sum((right-left)**2));pairs=len(left)
        neighbors[name]=dict(pairs=pairs,defined=int(valid.sum()),undefined=int((~valid).sum()),
            mean=float(c.mean()) if len(c) else None,median=float(np.median(c)) if len(c) else None,
            difference_energy_over_total=diff/energy if energy else None,
            mean_squared_difference_over_mean_token_energy=(diff/pairs)/(energy/n) if pairs and energy else None)
    return dict(shape=list(a.shape),energy=energy,metrics=metrics,neighbors=neighbors,
                singular_value_squared=spectrum.tolist(),min_raw_eigenvalue=float(eig.min()))


def torch_reference(array):
    """Separate FP64 projection/Gram/neighbor implementation for full-support readback."""
    import torch
    a=torch.as_tensor(np.asarray(array).copy(),dtype=torch.float64)
    x=a.reshape(-1,a.shape[-1]);energy=x.square().sum()
    if not energy:return None
    g=a.mean((0,1,2),keepdim=True).expand_as(a)
    t=a.mean((1,2),keepdim=True).expand_as(a)
    metrics={}
    for name,p in [('global',g),('temporal',t)]:
        metrics[name+'_energy']=float(p.square().sum()/energy)
        den=a.norm()*p.norm()
        metrics[name+'_cosine']=float((a*p).sum()/den) if den else None
    metrics['temporal_extra_energy']=float((t-g).square().sum()/energy)
    eig=torch.linalg.eigvalsh(x.T@x).flip(0);s=eig.clamp_min(0)
    for k in (1,4,8,16,32):metrics['rank%d_energy'%k]=float(s[:k].sum()/energy)
    neighbors={}
    for axis,name in enumerate(('temporal','vertical','horizontal')):
        left=a.narrow(axis,0,a.shape[axis]-1).reshape(-1,a.shape[-1])
        right=a.narrow(axis,1,a.shape[axis]-1).reshape(-1,a.shape[-1])
        den=left.norm(dim=-1)*right.norm(dim=-1);valid=den>0
        c=(left*right).sum(-1)[valid]/den[valid]
        dif=(right-left).square().sum();pairs=len(left)
        neighbors[name]=dict(pairs=pairs,defined=int(valid.sum()),undefined=int((~valid).sum()),
            mean=float(c.mean()) if len(c) else None,
            median=float(torch.quantile(c,.5)) if len(c) else None,
            difference_energy_over_total=float(dif/energy),
            mean_squared_difference_over_mean_token_energy=float((dif/pairs)/(energy/len(x))) if pairs else None)
    return dict(energy=float(energy),metrics=metrics,neighbors=neighbors,singular_value_squared=s.tolist())


def summarize(records):
    def agg(values):
        vals=[x for x in values if x is not None]
        return dict(defined=len(vals),undefined=len(values)-len(vals),mean=float(np.mean(vals)) if vals else None,
            median=float(np.median(vals)) if vals else None,min=float(min(vals)) if vals else None,max=float(max(vals)) if vals else None)
    out={k:agg([r['metrics'][k] for r in records]) for k in records[0]['metrics']}
    for axis in ('temporal','vertical','horizontal'):
        for metric in ('mean','median','difference_energy_over_total','mean_squared_difference_over_mean_token_energy'):
            out[axis+'_neighbor_'+metric]=agg([r['neighbors'][axis][metric] for r in records])
        out[axis+'_neighbor_pair_counts']={k:sum(r['neighbors'][axis][k] for r in records) for k in ('pairs','defined','undefined')}
    return out
