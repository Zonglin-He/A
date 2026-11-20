"""Read a measured component and its coordinate-permuted counterpart with the native frozen model."""
import numpy as np
import torch
from scipy.special import softmax
from physical_evidence import axes
from observe_lex import prompts

def sorted_rows(x):
    return x[np.lexsort((x[:,2],x[:,1],x[:,0]))]

def crop_views(points,yaw):
    raw=sorted_rows(np.unique(np.asarray(points,dtype=np.float32),axis=0))
    n=len(raw)
    if n==0:raise ValueError('empty crop')
    rng=np.random.default_rng(46913+n)
    if n>2048:raw=raw[np.sort(rng.choice(n,2048,replace=False))]
    n_used=len(raw);origin=raw.astype(float).mean(0)
    basis=axes(np.r_[np.zeros(6),yaw])
    local=(raw-origin)@basis.T
    synthetic=raw;changed=False
    if n_used>=3:
        for _ in range(8):
            moved=np.column_stack((local[:,0],local[rng.permutation(n_used),1],local[rng.permutation(n_used),2]))
            points_=np.ascontiguousarray(moved@basis+origin,dtype=np.float32)
            if len(np.unique(points_,axis=0))!=n_used:continue
            if np.array_equal(sorted_rows(moved),sorted_rows(local)):continue
            np.testing.assert_equal(np.sort(local,axis=0),np.sort(moved,axis=0))
            synthetic=sorted_rows(points_);changed=True;break
    # The same permutation and the same multiplicities are used in both clouds.
    order=np.random.default_rng(14111+n_used).permutation(n_used)
    ids=np.resize(order,2048)
    return np.stack([raw[ids],synthetic[ids]]),n,n_used,changed

def check_crop():
    cloud=np.random.default_rng(19).normal(size=(62,3)).astype(np.float32)
    a,n,k,altered=crop_views(cloud,.37)
    b,_,_,_=crop_views(cloud[::-1],.37)
    np.testing.assert_equal(a,b);assert altered and n==k==62
    frame=axes(np.r_[np.zeros(6),.37])
    first,second=[np.unique(c.astype(float),axis=0)@frame.T for c in a]
    np.testing.assert_allclose(np.sort(first,axis=0),np.sort(second,axis=0),atol=1.e-6)
    assert not crop_views(cloud[:1],0.)[-1]
    assert np.array_equal(*crop_views(cloud[:2],0.)[0])

def weighted_middle(values,weight):
    order=np.argsort(values,axis=0,kind='stable');cumulative=np.cumsum(weight[order],axis=0)
    ranks=np.argmax(cumulative>=.5*weight.sum(),axis=0)
    return values[order[ranks,np.arange(3)],np.arange(3)]

def summarize(tokens,dimensions,anchors):
    groups=[]
    for positions in anchors:
        _,inverse,count=np.unique(positions,axis=0,return_counts=True,return_inverse=True)
        groups.append(1./(len(count)*count[inverse]))
    weights=np.stack(groups)
    logits=(tokens*weights[:,:,None]).sum(axis=1).mean(0)
    posterior=softmax(tokens,axis=-1)
    dims=dimensions.reshape(-1,3).astype(float)
    ok=(dims>1.e-6).all(1)&np.isfinite(dims).all(1)
    dims=dims[ok];dims[:,:2]=np.sort(dims[:,:2],axis=1)[:,::-1]
    ext=np.full((tokens.shape[-1]-1,3),np.nan)
    if ok.any():
        for c in range(len(ext)):
            weight=(weights*posterior[:,:,c]).ravel()[ok]
            if weight.sum()>0:ext[c]=weighted_middle(dims,weight)
    return logits,ext

class LocalReader:
    def __init__(self,model,batch_size=16):
        if model.training or any(p.requires_grad for p in model.parameters()):raise ValueError('frozen model required')
        prepared=prompts(model)
        self.spec=[prepared[i] for i in (0,3)]
        self.model=model;self.device=next(model.parameters()).device;self.batch_size=batch_size;self.batches=0

    @torch.inference_mode()
    def read(self,crops):
        answer=[]
        for i in range(0,len(crops),self.batch_size):
            batch=crops[i:i+self.batch_size]
            x=torch.as_tensor(np.stack(batch),device=self.device)
            if not torch.isfinite(x).all():raise ValueError('nonfinite crop')
            scores=[];dims=[];locations=[]
            for text,_,_,spans in self.spec:
                z=self.model(dict(point_clouds=x,text=[text]*len(batch)))
                full=z['last_proj_queries']@z['proj_tokens'].transpose(1,2)/.07
                score=torch.stack([torch.logsumexp(full[:,:,span],axis=-1)-np.log(len(span)) for span in spans],-1)
                scores.append(score.float().cpu().numpy());dims.append(z['last_pred_size'].float().cpu().numpy())
                locations.append(z['query_points_xyz'].float().cpu().numpy());self.batches+=1
            for ls,ds,aa in zip(np.stack(scores,axis=1),np.stack(dims,axis=1),np.stack(locations,axis=1)):
                answer.append(summarize(ls,ds,aa))
        return answer

def observe_objects(reader,sets):
    views=[];props=[]
    for points,yaw in sets:
        v,n,k,changed=crop_views(points,yaw);views.extend(v);props.append((n,k,int(changed)))
    if not views:
        return dict(logit=np.empty((0,2,7)),dims=np.empty((0,2,6,3)),support=np.empty((0,3),int))
    vals=reader.read(views)
    logit=np.stack([v[0] for v in vals]).reshape(-1,2,7)
    dims=np.stack([v[1] for v in vals]).reshape(-1,2,6,3)
    support=np.asarray(props,int)
    equal=support[:,2]==0
    if equal.any():
        np.testing.assert_allclose(logit[equal,0],logit[equal,1],atol=2e-4,rtol=1e-5)
        # Equal point sets carry exactly zero intervention by construction.
        logit[equal,1]=logit[equal,0];dims[equal,1]=dims[equal,0]
    assert np.isfinite(logit).all()
    return dict(logit=logit,dims=dims,support=support)
