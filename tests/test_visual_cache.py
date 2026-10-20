import torch
from stvg.visual_cache import VisualCache

class Feature(torch.nn.Module):
    def __init__(self):super().__init__();self.calls=0
    def forward(self,x,n=None):self.calls+=1;return {'feature':x*2,'length':n}
class Model:
    def __init__(self):self.vis_encoder=Feature().eval();self.vid=Feature().eval()

def test_visual_cache_exact_inputs_and_no_state_aliasing():
    model=Model();cache=VisualCache(1024);x=torch.arange(8).float()
    with torch.no_grad(),cache.context(model):
        a=model.vid(x,8);b=model.vid(x.clone(),8)
        assert torch.equal(a['feature'],b['feature'])and model.vid.calls==1
        b['feature'].zero_()
        c=model.vid(x,8)
        # CPU hits must also be isolated from downstream mutations.
        assert torch.equal(c['feature'],a['feature'])
        model.vid(x+1,8);model.vid(x,7);model.vis_encoder(x,8)
    assert model.vid.calls==3 and model.vis_encoder.calls==1
    assert cache.hits==2 and cache.misses==4

def test_visual_cache_is_bounded_and_model_specific():
    cache=VisualCache(32);a=Model();b=Model()
    with torch.no_grad(),cache.context(a):
        a.vid(torch.ones(8));a.vid(torch.zeros(8))
    assert cache.bytes<=32 and len(cache.values)==1
    with torch.no_grad(),cache.context(b):b.vid(torch.zeros(8))
    assert b.vid.calls==1


def test_lazy_dictionary_preserves_all_keys_and_isolates_values():
    from stvg.visual_cache import DeviceDict
    original={'0':torch.arange(4).float(),'3':torch.ones(2)}
    view=DeviceDict(original,'cpu')
    assert set(view)=={'0','3'} and view.ready=={}
    assert torch.equal(view['3'],original['3'])and set(view.ready)=={'3'}
    view['3'].zero_();assert original['3'].sum()==2
    assert list(view.keys())==list(original.keys())
    assert set(view.copy())==set(original)
