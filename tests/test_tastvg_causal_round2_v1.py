import torch
from types import SimpleNamespace
from vg_tta.tastvg_causal_round2_v1 import pool,temporal_native,masks,accepts,losses

def test_bilinear_pooling_decomposition():
    h=torch.randn(9,3,256);d=torch.randn_like(h)*.01;a=torch.rand(2,4);ap=torch.rand(2,4);chosen=[0,2]
    q=pool(h,a,chosen,4,'app');qa=pool(h,ap,chosen,4,'app');qh=pool(h+d,a,chosen,4,'app');qah=pool(h+d,ap,chosen,4,'app')
    cross=pool(d,ap-a,chosen,4,'app')
    assert torch.allclose(qah-q,qa-q+qh-q+cross,atol=1e-6)

def test_native_temporal_formula_and_gradient():
    z=torch.tensor([[[.2,.3],[.5,-.2],[.1,.8]]],requires_grad=True);terms=[]
    for k,target in enumerate([0,2]):
        y=torch.exp(-(torch.arange(3)-target)**2/8)+1e-6;y=y/y.sum();p=z[0,:,k].softmax(0)
        terms.append((p*((p+1e-6)/y).log()).sum())
    expected=sum(terms)/3;actual=temporal_native(z,[0,2]);assert torch.equal(actual,expected)
    g,=torch.autograd.grad(actual,z);assert torch.isfinite(g).all() and g.norm()>0

def test_masks_leave_text_frozen_and_align_branches():
    h=torch.randn(11,3,256);data={'views':[{'H':h,'info':{'fea_map_size':[2,2]}}]}
    a=masks(data,'S')[0];m=masks(data,'T')[0];v=masks(data,'ST')[0]
    assert a[:4].all() and not a[4:].any() and m[-4:].all() and not m[:-4].any()
    assert torch.equal(a+m,v) and not v[4:-4].any()

def test_loss_descent_does_not_bypass_complement_preservation():
    b=torch.tensor([[.5,.5,.3,.3]]*4);p={'boxes':b,'indices':[1,3],'physical_interval':[10,31]}
    old={'S':2.,'T':1.,'ST':3.};new={'S':1.,'T':1.,'ST':2.}
    q={**p,'physical_interval':[0,31],'indices':[0,3]}
    assert accepts(old,new,'S',q,p,None)[0]
    assert not accepts(old,new,'S',q,p,'T')[0]
    changed={**p,'boxes':b+.2}
    assert not accepts(old,new,'S',changed,p,'S')[0]
