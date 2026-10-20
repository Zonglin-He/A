import numpy as np
import torch
from stvg.visual_cache import VisualCache
from stvg._host import inactive_modules_on_host

class Visual(torch.nn.Module):
    def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.ones(1),requires_grad=False);self.calls=0
    def forward(self,x,n=None):self.calls+=1;return {'3':x*2,'0':x*3}
class Other(torch.nn.Module):
    def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.ones(1),requires_grad=False);self.moves=[]
    def to(self,*a,**kw):self.moves.append(str(a[0]));return super().to(*a,**kw)
class Model(torch.nn.Module):
    def __init__(self):super().__init__();self.vis_encoder=Other();self.vid=Visual();self.decoder=Other()

def test_cache_probe_skips_staging_only_on_exact_hits():
    model=Model().eval();cache=VisualCache(4096);x=torch.arange(8).float()
    with torch.no_grad(),cache.context(model),inactive_modules_on_host(model):
        a=model.vid(x,8)
        assert len(model.decoder.moves)==2
        assert model.vid.forward.cache_will_hit(x,8)
        b=model.vid(x,8)
        assert len(model.decoder.moves)==2 and torch.equal(a['3'],b['3'])
        x.add_(1)
        c=model.vid(x,8)
        assert len(model.decoder.moves)==4 and torch.equal(c['3'],x*2)
    assert cache.hits==1 and cache.misses==2 and model.vid.calls==2

def test_probe_does_not_reuse_mutated_tensor_or_different_scalar():
    model=Model().eval();cache=VisualCache(4096);x=torch.arange(8).float()
    with torch.no_grad(),cache.context(model):
        model.vid(x,8)
        assert model.vid.forward.cache_will_hit(x,8)
        x.add_(1)
        y=model.vid(x,8)
        assert torch.equal(y['3'],x*2)
        assert not model.vid.forward.cache_will_hit(x,7)
        model.vid(x,7)
    assert cache.hits==0 and cache.misses==3


def test_geometry_cache_is_bitexact_and_grid_sensitive():
    from stvg.temporal import mass_table, probabilities
    rng=np.random.default_rng(91)
    for ids in [list(range(15)),[0,2,3,8,9,14,20,22],list(range(110))]:
        rec=[dict(frame_ids=ids[o::2])for o in (0,1)]
        for _ in range(3):
            g=probabilities(rec,[rng.normal(size=(1,len(r['frame_ids']),2)).astype(np.float32)for r in rec])
            ii=sorted({x for v in g for x in v['ids']});idx={x:i for i,x in enumerate(ii)};n=len(ii);ss=[];ee=[]
            for v in g:
                ss.append(np.array([idx[v['ids'][i]]for i in v['ij'][0]]));ee.append(np.array([idx[v['ids'][i]]for i in v['ij'][1]]))
            key=np.minimum(ss[0][:,None],ss[1][None,:])*n+np.maximum(ee[0][:,None],ee[1][None,:])
            w=g[0]['p'][:,None]*g[1]['p'][None,:]
            expected=np.bincount(key.ravel(),weights=w.ravel(),minlength=n*n).reshape(n,n)
            actual=mass_table(g)
            assert actual[0]==ii and np.array_equal(actual[1],expected)
