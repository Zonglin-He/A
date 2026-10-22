"""Scoped native full-caption query reuse; no fitted state or attention edits."""
from contextlib import contextmanager
import torch


def expanded_like(stored,current):
    assert stored.shape==current.shape and current.ndim==3 and current.shape[1]==1
    assert current.stride()[0]==0 and torch.equal(stored,stored[:1].expand_as(stored))
    vector=stored[0,0].to(device=current.device,dtype=current.dtype).clone()
    out=torch.as_strided(vector,current.shape,current.stride())
    assert out.stride()==current.stride() and torch.equal(out,stored.to(out))
    return out


@contextmanager
def shared_second_pass(decoder,full_queries):
    assert len(full_queries)==2;calls=[]
    def before(m,args,kwargs):
        assert not args and len(calls)<4
        index=len(calls);o=index//2;record=dict(call=index,offset=o,replaced=index%2==1)
        if record['replaced']:
            q=kwargs['query_tgt'];out=expanded_like(full_queries[o],q)
            record.update(same_values=torch.equal(out,q),stride=list(q.stride()),delta_norm=float((out[0,0]-q[0,0]).norm().item()),original_norm=float(q[0,0].norm().item()))
            kwargs=dict(kwargs);kwargs['query_tgt']=out
        calls.append(record);return args,kwargs
    hook=decoder.register_forward_pre_hook(before,with_kwargs=True)
    try:yield calls
    finally:hook.remove()
    assert len(calls)==4
