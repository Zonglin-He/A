"""Synthetic fusion and strict frozen-head boundary tests; no private assets."""
import numpy as np
import pytest
import torch
from scripts.tube_fusion.core import head_logits,layer_posteriors,fuse
from stvg.temporal import probabilities,mass_table,final_mbr

def fixture_logits():
    gen=torch.Generator().manual_seed(27);ids=[2,7,13,22,35,51]
    records=[dict(offset=o,flip=False,frame_ids=ids[o::2])for o in(0,1)]
    z=[torch.randn(6,1,3,2,generator=gen)*2 for _ in records]
    return records,z,ids

def test_average_physical_posteriors_not_logits():
    records,z,ids=fixture_logits();q,layers=layer_posteriors(records,z,ids)
    independent=[]
    for k in range(6):
        physical,table=mass_table(probabilities(records,[a[k]for a in z]));assert physical==ids;independent.append(table)
    assert np.array_equal(layers,np.stack(independent))and np.array_equal(q,np.mean(independent,axis=0))
    _,incorrect=mass_table(probabilities(records,[a.mean(0)for a in z]))
    assert not np.allclose(q,incorrect)and abs(q.sum()-1)<1e-12
    assert final_mbr(ids,q)['interval'][0]in ids

def test_equal_fusion_and_identity():
    r,z,ids=fixture_logits();g,_=layer_posteriors(r,z,ids);t,_=layer_posteriors(r,[a.flip(0)*.5 for a in z],ids)
    assert np.array_equal(fuse(g,t),.5*g+.5*t)
    assert np.array_equal(fuse(g,g),g)and abs(fuse(g,t).sum()-1)<1e-12
    assert np.array_equal(g,np.triu(g,1))

def test_actual_full_six_layer_head_batches():
    torch.manual_seed(29)
    head=torch.nn.Sequential(torch.nn.Linear(256,256),torch.nn.ReLU(),torch.nn.Dropout(.3),torch.nn.Linear(256,2)).eval().requires_grad_(False)
    hidden=[torch.randn(6,1,n,256)for n in(4,3)];saved=[head(h)[-1].clone()for h in hidden];calls=[]
    hook=head.register_forward_pre_hook(lambda m,a:calls.append(tuple(a[0].shape)))
    out,cost=head_logits(head,hidden,saved);hook.remove()
    assert calls==[(6,1,4,256),(6,1,3,256)]and cost['head_calls']==2
    assert all(torch.equal(a[-1],b)for a,b in zip(out,saved))
    assert all(p.grad is None and not p.requires_grad for p in head.parameters())

def test_wrong_last_layer_is_rejected():
    head=torch.nn.Linear(256,2).eval().requires_grad_(False);hidden=[torch.ones(6,1,3,256)]*2
    saved=[head(h)[-1].clone()for h in hidden];saved[0][0,0,0]+=1
    with pytest.raises(AssertionError,match='bitwise'):head_logits(head,hidden,saved)
