from types import SimpleNamespace
import numpy as np
import torch
import pytest
from vg_tta import llava_st_teacher as module

@pytest.mark.parametrize('recipe',['greedy','official'])
def test_recipe_seed_and_100_frame_contract_without_GPU(monkeypatch,recipe):
    calls=[]
    model=SimpleNamespace(config=SimpleNamespace(max_frame=100),model=SimpleNamespace(vision_config={}))
    def generate(ids,**kw):
        calls.append(kw);return torch.tensor([[7,8]])
    model.generate=generate
    monkeypatch.setattr(module,'imports',lambda path:(lambda x,c:x,lambda x,t,has_image:torch.tensor([[1,2]]),lambda c:(c,{})))
    monkeypatch.setattr(torch.Tensor,'cuda',lambda t:t)
    tokenizer=SimpleNamespace(batch_decode=lambda x,**kw:['{0,1}'])
    processor=SimpleNamespace(preprocess=lambda x,**kw:{'pixel_values':torch.zeros(100,3,4,4)})
    pred,pixels=module.predict(tokenizer,model,processor,np.zeros((100,4,4,3),dtype=np.uint8),'the person','.',decode=recipe)
    assert calls[0]['do_sample']==(recipe=='official') and calls[0]['num_beams']==1
    if recipe=='official':assert calls[0]['temperature']==.01 and calls[0]['top_p'] is None
    else:assert 'temperature' not in calls[0]
    assert pred['seed']==20260928 and pred['max_frame']==100 and pred['raw_text']=='{0,1}'
    model.config.max_frame=32
    with pytest.raises(AssertionError):module.predict(tokenizer,model,processor,np.zeros((100,4,4,3)),'q','.')
