"""Read an existing source-trained motion classifier before its time reduction.
No new class dictionary, head, parameters, RGB, labels or localization updater.
"""
from contextlib import contextmanager
import math,types
import numpy as np
import torch
from vg_tta.stvg_soft_correspondence_v1 import cell_overlap


def coverage(boxes,valid,h,w):
    assert valid.shape==(len(boxes),h*w) and valid.all(), 'this qualified cache has no padding; do not guess padded geometry'
    out=cell_overlap(boxes,h,w).flatten(1).float();empty=out.sum(-1)<=0
    out[empty]=1.
    return out,empty


@contextmanager
def area_prior(head,area,trace=None):
    """Scope log fractional-area prior to only two existing attribute cross-attentions."""
    modules=[l.attention.self for l in head.layer_ca];saved=[]
    try:
        for m in modules:
            original=m.forward;had='forward' in m.__dict__;saved.append((m,original,had))
            def altered(self,query_states,key_states,value_states,_orig=original):
                if bool((area==1).all()):return _orig(query_states,key_states,value_states)
                q=self.transpose_for_scores(self.query(query_states));k=self.transpose_for_scores(self.key(key_states));v=self.transpose_for_scores(self.value(value_states))
                logits=q@k.transpose(-1,-2)/math.sqrt(self.attention_head_size)
                bias=area.to(logits).log()[:,None,None,:]
                assert logits.shape[0]==len(area) and logits.shape[-1]==area.shape[-1]
                prob=torch.softmax(logits+bias,-1);att=prob.clone();context=self.dropout(prob)@v
                if trace is not None:trace.append(dict(q=q.detach().cpu(),k=k.detach().cpu(),v=v.detach().cpu(),area=area.detach().cpu(),prob=att.detach().cpu(),context=context.detach().cpu()))
                context=context.permute(0,2,1,3).contiguous();context=context.view(*context.size()[:-2],self.all_head_size)
                return context,att
            m.forward=types.MethodType(altered,m)
        yield
    finally:
        for m,original,had in saved:
            if had:m.forward=original
            else:del m.forward


@torch.no_grad()
def per_frame(head,fmap,text,area=None,trace=None):
    saved=[];handle=head.head.register_forward_hook(lambda module,args,value:saved.append(value.detach()))
    try:
        if area is None:aggregate,_=head(fmap,text)
        else:
            with area_prior(head,area,trace):aggregate,_=head(fmap,text)
    finally:handle.remove()
    assert len(saved)==1 and saved[0].shape==(len(fmap),1,34)
    raw=saved[0][:,0,:]
    assert torch.isfinite(raw).all() and torch.equal(saved[0].mean(0),aggregate)
    return raw.cpu()


def prototype(raw,weights=None):
    x=np.asarray(raw,float);assert x.shape==(65,34) and np.isfinite(x).all()
    w=np.ones(65) if weights is None else np.asarray(weights,float)
    assert w.shape==(65,) and (w>=0).all() and w.sum()>0
    logits=(x*w[:,None]).sum(0)/w.sum();probs=1/(1+np.exp(-logits));return probs/probs.sum()


def curve(raw,q):
    x=np.asarray(raw,float);assert x.shape==(65,34) and np.asarray(q).shape==(34,)
    return (1/(1+np.exp(-x)))@q


def edges(ids,frame_count):
    x=np.asarray(ids,float);return np.r_[0.,(x[:-1]+x[1:])/2,float(frame_count)]


def properties():
    x=np.arange(65*34).reshape(65,34)/1000-1;q=prototype(x);assert abs(q.sum()-1)<1e-14 and np.isfinite(curve(x,q)).all()
    assert np.allclose(prototype(x,np.ones(65)*2),q,atol=1e-15)
    from vg_tta.stvg_temporal_increment128_v1 import feedback,tilt
    p=np.triu(np.ones((65,65)),1);p/=p.sum();z,a=feedback(np.arange(65),edges(np.arange(65),65),np.ones(65));assert not a['active'] and np.array_equal(tilt(p,z,.5),p) and np.array_equal(tilt(p,z,0),p)
    area,missing=coverage(torch.tensor([[.5,.5,1.,1.],[2.,2.,.01,.01],[.1,.3,.01,.01]],dtype=torch.float64),torch.ones(3,6,dtype=torch.bool),2,3)
    assert torch.equal(area[0],area[1]) and not missing[0] and missing[1] and not missing[2]
    return dict(status='pass',prototype_normalized=True,zero_lambda_and_flat_exact=True,full_and_empty_geometry_same=True,subcell_nonempty=True)
