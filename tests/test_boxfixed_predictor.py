"""CPU mocks of native BoxFixed contracts; no real RGB/model/cache or GT."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import stvg.predictor as predictor
from stvg.native import FrozenNativeReplay


class SpatialFlags(nn.Module):
    def forward(self,feature,text):
        n,_,h,w=feature.shape
        return None,torch.ones(n,h*w)


class TemporalFlags(nn.Module):
    def forward(self,feature,text):return torch.ones(len(feature))


class PositionDecoder(nn.Module):
    def __init__(self):super().__init__();self.calls=[]
    def forward(self,*,query_tgt,encoded_feature,encoded_pos,encoded_mask):
        assert encoded_feature.dtype==torch.float32
        self.calls.append(dict(dtype=encoded_feature.dtype,nframes=encoded_feature.shape[1]))
        return encoded_feature[0,:,:4][None,:,None].repeat(6,1,1,1)


class TimeDecoder(nn.Module):
    def __init__(self):super().__init__();self.calls=0
    def forward(self,*,memory):
        self.calls+=1
        return memory[0][None].repeat(6,1,1)


class GroundDecoder(nn.Module):
    def __init__(self):
        super().__init__();self.decoder=PositionDecoder();self.time_decoder=TimeDecoder();self.calls=0
    def forward(self,*,encoded_info,vis_pos,itq,isq):
        self.calls+=1;H=encoded_info['encoded_feature'];nf=H.shape[1]
        position=self.decoder(query_tgt=H[:1],encoded_feature=H,encoded_pos=vis_pos,encoded_mask=torch.zeros(nf,len(H),dtype=torch.bool))
        return position,self.time_decoder(memory=H)


class ActionHead(nn.Module):
    def forward(self,h):return h[...,:1]*0+1


class NativeModel(nn.Module):
    def __init__(self):
        super().__init__();self.ground_encoder=SimpleNamespace(encoder=SimpleNamespace(norm=nn.Identity()))
        self.ground_decoder=GroundDecoder();self.t_spatial_clas=SpatialFlags();self.s_spatial_clas=SpatialFlags()
        self.t_temporal_clas=TemporalFlags();self.s_temporal_clas=TemporalFlags();self.theta=.5
        self.action_embed=ActionHead();self.temp_embed=nn.Identity();self.eval().requires_grad_(False)


def test_native_FP32_two_pass_offset_mapping_and_no_optimizer(monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('No optimizer may be constructed')
    monkeypatch.setattr(torch.optim,'Adam',forbidden)
    model=NativeModel();views=[];T=9;count=2
    for offset in (0,1):
        nf=len(range(offset,T,2));prefix=torch.ones(2*count+2,nf,256,dtype=torch.float16)
        for local,node in enumerate(range(offset,T,2)):
            prefix[0,local,:4]=torch.tensor([node,.5,.2,.3])
        views.append(dict(prefix=prefix,info={'fea_map_size':(1,2)},vis_pos=torch.ones_like(prefix)))
    out=FrozenNativeReplay(model,views,T)
    assert model.ground_decoder.calls==4 # two original FP32 passes per offset.
    assert len(out.spatial_inputs)==len(out.time_inputs)==len(out.temporal_inputs)==2
    assert len(model.ground_decoder.decoder.calls)==model.ground_decoder.time_decoder.calls==4
    assert [len(v['encoded_mask'])for v in out.spatial_inputs]==[5,4]
    assert out.zero['boxes'].shape==(T,4)and torch.equal(out.zero['boxes'][:,0],torch.arange(T).float())
    assert all(v['encoded_feature'].dtype==torch.float32 for v in out.spatial_inputs)
    assert all(not p.requires_grad for p in model.parameters())
    assert not hasattr(out,'delta')and not hasattr(out,'named')and not hasattr(out,'optimizer')


class NeverIterate:
    def __iter__(self):raise AssertionError('Native224 cannot assemble six-layer Strong posteriors')


class CaptureModel:
    def __init__(self):
        self.head_calls=0;self.cuda_calls=0;self.cpu_calls=0
        self.ground_decoder=SimpleNamespace(decoder=nn.Identity(),time_decoder=nn.Identity())
    def cuda(self):self.cuda_calls+=1;return self
    def cpu(self):self.cpu_calls+=1;return self
    def temp_embed(self,*args):
        self.head_calls+=1
        raise AssertionError('Native224 must use cached last-layer native logits')


class Geometry:
    def __init__(self,E):self.E=torch.as_tensor(E);self.calls=[]
    def for_full_timeline(self):self.calls.append(('full',None));return ('full',self)
    def for_interval(self,interval):self.calls.append(('interval',list(interval)));return ('interval',self)
    def temporal_reliability(self):return torch.linspace(0,1,len(self.E),dtype=torch.float64)


def setup_capture(monkeypatch):
    T=9;positions=list(range(8));frames=np.zeros((T,4,5,3),dtype=np.uint8);ids=list(range(T))
    source=CaptureModel();calls=dict(native=0,teacher=0,evidence=[],geometry=[],DP=[],time_cache=0,time_replay=[])
    boxes=torch.tensor([[.5,.5,.2,.3]]*T,dtype=torch.float32)
    spatial=[{'actual_offset':0},{'actual_offset':1}]
    base=SimpleNamespace(zero=dict(logits=[torch.zeros(5,2),torch.zeros(4,2)],boxes=boxes),
        spatial_inputs=spatial,temporal_inputs=NeverIterate())
    def native(*a,**kw):
        calls['native']+=1
        for _ in range(8):
            source.ground_decoder.decoder(torch.zeros(1))
            source.ground_decoder.time_decoder(torch.zeros(1))
        return object(),[{},{}],base
    monkeypatch.setattr(predictor,'frozen_native_forward',native)
    monkeypatch.setattr(predictor,'prediction',lambda *a,**kw:dict(boxes=boxes,indices=[0,T-1]))
    monkeypatch.setattr(predictor,'probabilities',lambda *a,**kw:'native-probability')
    monkeypatch.setattr(predictor,'mass_table',lambda *a,**kw:(ids,np.array([.1,.9])))
    monkeypatch.setattr(predictor,'final_mbr',lambda *a,**kw:dict(interval=[1,7]))
    monkeypatch.setattr(predictor,'observation_positions',lambda *a,**kw:positions)
    monkeypatch.setattr(predictor,'initialize',lambda source,pos,teacher,cal:(source.copy(),dict(delta=[0.]*4)))
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    def forbidden(*a,**kw):raise AssertionError('Forbidden heavy/optimization path')
    monkeypatch.setattr(torch.optim,'Adam',forbidden)
    import stvg.temporal_capture as tc
    monkeypatch.setattr(tc,'frozen_temporal_forward',forbidden)
    teacher=dict(boxes=boxes[:8].clone(),native_mask_logits=torch.ones(8,2,3),
        input_shape=[8,3,4,5],selected_query=2,nested_mask_all_false=True,cost={},query_forwards=1)
    def service(*a,**kw):calls['teacher']+=1;return teacher
    def evidence(*a,**kw):calls['evidence'].append((a,kw));return 'frozen-evidence'
    monkeypatch.setattr(predictor.SourceEvidence,'from_capture',evidence)
    def geometry(ev,E,native):
        assert ev=='frozen-evidence'
        calls['geometry'].append((ev,np.array(E),native));return Geometry(E)
    monkeypatch.setattr(predictor,'build_geometry',geometry)
    def DP(view):
        mode,g=view;calls['DP'].append(mode)
        return dict(final=g.E.clone(),score=.4,costs={'updates':0})
    monkeypatch.setattr(predictor,'evidence_dp',DP)
    def time_cache(*a,**kw):calls['time_cache']+=1;return {'actual_native_decoder_inputs':True}
    def replay(model,cache,b,r,**kw):
        calls['time_replay'].append(dict(cache=cache,boxes=b.clone(),reliability=r.clone(),**kw))
        return dict(interval=[2,8],posterior=np.array([.3,.7]),cost={'extra_decoder_calls':2})
    monkeypatch.setitem(sys.modules,'stvg.tube_replay',SimpleNamespace(cache_temporal=time_cache,replay_temporal=replay))
    config=dict(dataset='hc2',E_calibration={},initialization_gamma=.5,temporal_mode='native224')
    obj=predictor.BoxFixedPredictor(config=config,source=source);obj.teacher_service=service
    return obj,frames,ids,dict(caption='synthetic object',height=4,width=5),teacher,calls,source


def test_native224_no448_no_layer_fusion_one_eight_frame_expert(monkeypatch):
    obj,frames,ids,metadata,teacher,calls,source=setup_capture(monkeypatch)
    cache=obj.capture(frames,ids,metadata,parses={'subject':'object'},temporal_mode='native224')
    assert calls['native']==calls['teacher']==1 and source.head_calls==0
    assert calls['time_cache']==0 and len(calls['evidence'])==len(calls['geometry'])==1
    args,kw=calls['evidence'][0]
    assert args[1]==ids and args[2]==list(range(8))and args[3]is teacher['native_mask_logits']
    assert kw['selected_query']==teacher['selected_query']
    assert set(cache['intervals'])==set(cache['posterior'])=={'Native'}
    assert cache['cost']['Source448_calls']==cache['cost']['Source448_seconds']==0
    assert cache['cost']['native_time_decoder_calls']==cache['cost']['native_position_decoder_calls']==8
    assert cache['geometry'].calls==[] # Geometry construction receives no Strong/event support.
    final=obj.finish(cache,'native224')
    assert final['interval']==[1,7]and final['cost']['actual_updates']==0
    assert final['cost']['extra_decoder_calls']==0 and calls['DP']==['interval']
    assert cache['geometry'].calls==[('interval',[1,7])]
    assert final['GT_read']is False


def test_tube_replay_fulltimeline_then_one_final_interval_and_same_D_budget(monkeypatch):
    obj,frames,ids,metadata,_,calls,source=setup_capture(monkeypatch)
    cache=obj.capture(frames,ids,metadata,parses={'subject':'object'},temporal_mode='tube_replay')
    assert calls['time_cache']==1 and source.head_calls==0 and cache['cost']['Source448_calls']==0
    c=obj.finish(cache,'tube_replay');d=obj.finish(cache,'tube_replay',permuted=True)
    assert calls['DP']==['full','interval','full','interval']
    assert cache['geometry'].calls==[('full',None),('interval',[2,8]),('full',None),('interval',[2,8])]
    assert c['cost']['extra_decoder_calls']==d['cost']['extra_decoder_calls']==2
    first,second=calls['time_replay']
    assert torch.equal(first['boxes'],second['boxes'])and torch.equal(first['reliability'],second['reliability'])
    assert first['alpha']==second['alpha']==.5 and not first['permuted']and second['permuted']
    assert c['cost']['actual_updates']==d['cost']['actual_updates']==0


def test_teacher_override_binding_and_missing_mask_fail_closed(monkeypatch):
    obj,frames,ids,metadata,teacher,calls,_=setup_capture(monkeypatch)
    override={**teacher,'evidence_binding_verified':True,'positions8':list(range(8)),'frame_ids':ids}
    cache=obj.capture(frames,ids,metadata,parses={'subject':'object'},teacher_override=override)
    assert calls['teacher']==0 and cache['teacher_native']is override
    bad={**override,'positions8':list(reversed(range(8)))}
    with pytest.raises(ValueError,match='observation grid'):
        obj.capture(frames,ids,metadata,parses={'subject':'object'},teacher_override=bad)
    teacher.pop('native_mask_logits')
    with pytest.raises(ValueError,match='lacks selected native masks'):
        obj.capture(frames,ids,metadata,parses={'subject':'object'})


def test_default_modules_have_no_history_import_or_trainable_adaptation():
    root=Path(__file__).resolve().parents[1]
    for name in ('predictor.py','native.py','evidence.py','spatial.py','_spatial_math.py'):
        tree=ast.parse((root/'stvg'/name).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom):
                assert not(node.module or'').startswith('scripts')
                assert (node.module or'').split('.')[-1]not in ('legacy_predictor','policy','state')
            if isinstance(node,ast.Call):
                f=node.func
                assert not(isinstance(f,ast.Attribute)and f.attr in ('Adam','SGD','backward'))
