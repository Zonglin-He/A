"""Synthetic CPU graphs only: no pretrained model, video, label or CUDA calls."""
import ast
import copy
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from stvg import temporal_capture as fast
from stvg.source_replay import SpatialReplay

ROOT=Path(__file__).resolve().parents[1]


def original_query_forward():
    path=ROOT/'external/TA-STVG/models/grounding_model/query_decoder.py'
    tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='QueryDecoder')
    node=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='forward')
    scope={'torch':torch}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    return scope['forward']


class TimeEmbedding(nn.Module):
    def forward(self,n):return torch.arange(n,dtype=torch.float32)[:,None,None].expand(n,1,256)/10


class TimeDecoder(nn.Module):
    def __init__(self):super().__init__();self.history=[]
    def forward(self,query_tgt,query_content,query_time,encoded_feature,**kw):
        base=query_tgt+query_content+query_time+encoded_feature.mean(0)[:,None,:]*.01
        out=torch.stack([base+(i+1)/10 for i in range(6)]).transpose(1,2)
        self.history.append(out.detach().clone());return out


class PosDecoder(nn.Module):
    def __init__(self):super().__init__();self.calls=0
    def forward(self,query_tgt,**kw):
        self.calls+=1
        return torch.zeros(6,query_tgt.shape[0],1,4)


class FullDecoder(nn.Module):
    forward=original_query_forward()
    def __init__(self):
        super().__init__();self.d_model=256
        self.pos_fc=nn.Linear(256,4);self.time_fc=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,256))
        self.time_embed=TimeEmbedding();self.time_decoder=TimeDecoder();self.decoder=PosDecoder()


class TemporalClassifier(nn.Module):
    def __init__(self,fallback):super().__init__();self.fallback=fallback
    def forward(self,x,query):
        a=torch.arange(len(x));return torch.full((len(x),),-2.)if self.fallback else torch.where(a%2==0,torch.tensor(2.),torch.tensor(-2.))


class SpatialClassifier(nn.Module):
    def __init__(self):super().__init__();self.calls=0
    def forward(self,x,query):
        self.calls+=1
        return torch.zeros(1),torch.linspace(.2,.8,x.shape[-2]*x.shape[-1]).expand(len(x),-1)


class Actions(nn.Module):
    def __init__(self,fallback):super().__init__();self.fallback=fallback
    def forward(self,h):
        shape=h.shape[-2];pattern=torch.full((shape,),-2.)if self.fallback else torch.where(torch.arange(shape)%2==0,torch.tensor(2.),torch.tensor(-2.))
        return h[...,:1]*0+pattern[None,None,:,None]


def synthetic(fallback=False,theta=.45):
    torch.manual_seed(7)
    model=nn.Module();model.ground_encoder=nn.Module();model.ground_encoder.encoder=nn.Module()
    model.ground_encoder.encoder.norm=nn.LayerNorm(256);model.ground_decoder=FullDecoder();model.temp_embed=nn.Linear(256,2)
    model.t_temporal_clas=TemporalClassifier(fallback);model.s_temporal_clas=TemporalClassifier(fallback)
    model.t_spatial_clas=SpatialClassifier();model.s_spatial_clas=SpatialClassifier();model.action_embed=Actions(fallback);model.theta=theta
    model.eval().requires_grad_(False)
    views=[]
    for nf in [4,3]:
        prefix=torch.randn(10,nf,256)
        info=dict(encoded_mask=torch.zeros(nf,10,dtype=torch.bool),fea_map_size=(2,2),durations=[nf])
        views.append(dict(prefix=prefix,info=info,vis_pos=torch.randn(nf,256,2,2)))
    return model,views


@pytest.mark.parametrize('fallback',[False,True])
@pytest.mark.parametrize('theta',[.45,.7])
def test_full_fp32_reference_temporal_path_is_bitexact(fallback,theta):
    model,views=synthetic(fallback,theta)
    assert fast.verify_query_forward(model.ground_decoder)==fast.QUERY_FORWARD_AST
    old=SimpleNamespace(model=model,n=7,views=views,norm=copy.deepcopy(model.ground_encoder.encoder.norm),decoder=copy.deepcopy(model.ground_decoder),head=copy.deepcopy(model.temp_embed).float(),delta=torch.zeros(256),spatial_inputs=[],temporal_inputs=[])
    old._indices=SpatialReplay._indices
    old._queries=lambda *a:SpatialReplay._queries(old,*a)
    with torch.no_grad():expected=SpatialReplay.reference_values(old,capture=True)
    rng=torch.get_rng_state().clone();state={k:v.clone()for k,v in model.state_dict().items()}
    new=fast.TemporalReplay(model,views,audit=True)
    assert torch.equal(torch.get_rng_state(),rng)
    assert all(torch.equal(v,state[k])for k,v in model.state_dict().items())
    assert new.zero['gates']==expected['gates']
    assert old.decoder.decoder.calls==4
    assert not hasattr(new.decoder,'decoder')and not hasattr(new,'delta')
    assert len(new.decoder.time_decoder.history)==4
    for i in range(2):
        assert torch.equal(new.zero['logits'][i],expected['logits'][i])
        assert torch.equal(new.zero['actions'][i],expected['actions'][i])
        assert torch.equal(new.temporal_inputs[i],old.temporal_inputs[i])
        assert torch.equal(new.zero['audit'][i]['first_hidden'],old.decoder.time_decoder.history[i*2])
        assert torch.equal(new.zero['audit'][i]['second_logits_all_layers'],old.head(old.temporal_inputs[i]))
        assert new.zero['audit'][i]['first_used_fallback']==fallback
        assert new.zero['audit'][i]['second_used_fallback']==fallback
    assert not any(p.requires_grad for p in new.decoder.parameters())


class PrefixHarness(nn.Module):
    def __init__(self,model,view,fail=False):
        super().__init__();self.ground_encoder=model.ground_encoder;self.ground_decoder=model.ground_decoder;self.view=view;self.fail=fail
    def forward(self,*args,**kwargs):
        H=self.ground_encoder.encoder.norm(self.view['prefix'])
        if self.fail:raise ValueError('synthetic prefix failure')
        info={**self.view['info'],'encoded_feature':H,'frames_cls':H.mean(0),'videos_cls':H.mean(0).mean(0)}
        return self.ground_decoder(encoded_info=info,vis_pos=self.view['vis_pos'],itq=None,isq=None)


@pytest.mark.parametrize('fail',[False,True])
def test_prefix_owned_exit_restores_hooks_without_running_decoder(monkeypatch,fail):
    model,views=synthetic();harness=PrefixHarness(model,views[0],fail)
    monkeypatch.setattr(torch,'autocast',lambda *a,**kw:nullcontext())
    before=(len(harness.ground_encoder.encoder.norm._forward_pre_hooks),len(harness.ground_decoder._forward_pre_hooks))
    view=dict(videos=None,texts=['synthetic'],targets=[])
    if fail:
        with pytest.raises(ValueError,match='synthetic prefix failure'):fast._prefix_view(harness,view,torch.device('cpu'))
    else:
        captured=fast._prefix_view(harness,view,torch.device('cpu'))
        assert torch.equal(captured['prefix'],views[0]['prefix'])
    assert before==(len(harness.ground_encoder.encoder.norm._forward_pre_hooks),len(harness.ground_decoder._forward_pre_hooks))
    assert harness.ground_decoder.decoder.calls==0 and harness.ground_decoder.time_decoder.history==[]


def test_unknown_query_decoder_fails_closed():
    class Different(nn.Module):
        def forward(self,x):return x
    with pytest.raises(RuntimeError,match='audited QueryDecoder'):fast.verify_query_forward(Different())
