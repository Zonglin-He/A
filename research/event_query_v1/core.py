"""Bounded temporal-query experiment; no labels or heavy model imports."""
import numpy as np

def score_feedback(curve,ids,positions):
    n=len(ids);pairs=np.stack(np.triu_indices(n,1),-1)
    anchors=np.asarray(ids)[positions]+.5
    membership=(anchors[None]>=np.asarray(ids)[pairs[:,0],None])&(anchors[None]<np.asarray(ids)[pairs[:,1],None]+1)
    raw=np.zeros(len(pairs));predict=np.zeros(len(pairs))
    if curve is None or np.std(curve)<1e-12:return pairs,raw,predict,dict(unknown=True)
    z=(np.asarray(curve)-np.mean(curve))/np.std(curve)
    patterns,which=np.unique(membership,axis=0,return_inverse=True)
    p_raw=[];p_predict=[]
    for m in patterns:
        inside=int(m.sum());outside=8-inside
        p_raw.append(float(z[m].mean()-z[~m].mean())if inside and outside else 0.)
        # Leave-one-anchor-out predictive gain, fixed slope ridge=1 and positive
        # effect. At least two observed anchors in each group; no invented labels.
        gain=0.
        if inside>=2 and outside>=2:
            for k in range(8):
                take=np.arange(8)!=k;x=m[take].astype(float);y=z[take]
                xc=x-x.mean();yc=y-y.mean();slope=max(0.,float(xc@yc/(xc@xc+1.)))
                predicted=y.mean()+slope*(float(m[k])-x.mean())
                gain+=.5*((z[k]-y.mean())**2-(z[k]-predicted)**2)
        p_predict.append(gain)
    from legacy_event import candidates,interval_scores
    # Same physical Voronoi integration as the archived contrast, now evaluated
    # on native legal intervals. No GT-dependent event duration enters this.
    edges,_=candidates(ids,positions,int(ids[-1])+1,[int(ids[0]),int(ids[-1])+1])
    raw_scores=interval_scores(z,edges,[[int(ids[a]),int(ids[b])+1]for a,b in pairs])
    return pairs,np.asarray([0. if v is None else v for v in raw_scores]),np.asarray(p_predict)[which],dict(unknown=False,patterns=len(patterns),predictive_supported_patterns=sum(2<=x.sum()<=6 for x in patterns),curve_std=float(np.std(curve)))

def target(q,pairs,score):
    w=np.log(np.maximum(q[pairs[:,0],pairs[:,1]],1e-300))+score
    p=np.exp(w-w.max());p/=p.sum();out=np.zeros_like(q);out[pairs[:,0],pairs[:,1]]=p
    return out

def torch_posterior(records,logits,ids):
    """Differentiable exact two-offset hull via CDF products, six-layer mean."""
    import torch
    n=len(ids);lookup={v:i for i,v in enumerate(ids)};tables=[]
    for layer in range(6):
        cdfs=[]
        for rec,z in zip(records,logits):
            v=z[layer,0].double();i,j=torch.triu_indices(len(v),len(v),1,device=v.device)
            probability=torch.softmax(v[i,0]+v[j,1],0)
            index=torch.tensor([lookup[x]for x in rec['frame_ids']],device=v.device)
            flat=torch.zeros(n*n,device=v.device,dtype=torch.float64)
            flat=flat.scatter_add(0,index[i]*n+index[j],probability).reshape(n,n)
            cdfs.append(flat.flip(0).cumsum(0).flip(0).cumsum(1))
        D=cdfs[0]*cdfs[1]
        below=torch.cat([D[1:],torch.zeros_like(D[:1])],0)
        left=torch.cat([torch.zeros_like(D[:,:1]),D[:,:-1]],1)
        diagonal=torch.cat([torch.zeros_like(below[:,:1]),below[:,:-1]],1)
        q=(D-below-left+diagonal).triu(1).clamp_min(0)
        tables.append(q/q.sum())
    return torch.stack(tables).mean(0)

def restore_inputs(offset,device):
    import torch
    result={}
    for key,value in offset['inputs'].items():
        layout=offset['input_layouts'][key];shape=tuple(layout['shape']);stride=tuple(layout['stride']);start=int(layout.get('storage_offset',0))
        size=start+1+sum((s-1)*d for s,d in zip(shape,stride));back=torch.empty(size,device=device,dtype=value.dtype)
        collapsed=tuple(1 if d==0 else s for s,d in zip(shape,stride));view=torch.as_strided(back,collapsed,stride,start)
        source=value[tuple(slice(0,1)if d==0 else slice(None)for d in stride)].to(device)
        view.copy_(source);result[key]=torch.as_strided(back,shape,stride,start)
    return result
