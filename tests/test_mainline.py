"""Main-only CPU contract tests; no real RGB, checkpoints, GPU or labels."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from stvg.global6 import posterior
from stvg.temporal import mass_table,probabilities,final_mbr
ROOT=Path(__file__).resolve().parents[1]

def test_default_and_cli_expose_only_main_method():
 cfg=json.loads((ROOT/'configs/default.json').read_text());assert cfg['temporal']['default']=='global6'
 assert all(c['temporal_mode']=='global6'for c in cfg['datasets'].values())
 assert not any((ROOT/'stvg'/x).exists()for x in ['legacy_predictor.py','policy.py','state.py','cgstvg.py','tube_replay.py','temporal_capture.py','source_replay.py'])
 source=(ROOT/'stvg/__main__.py').read_text();assert '--temporal'not in source
 from stvg.predictor import Predictor
 with pytest.raises(ValueError,match='fixed Global6'):Predictor('hc2').predict(None,None,None,temporal_mode='strong')

def test_global6_is_mean_of_six_independent_two_offset_hulls():
 ids=[0,2,7,10,19,23];recs=[dict(offset=i,frame_ids=ids[i::2])for i in [0,1]]
 gen=np.random.default_rng(7);logits=[torch.tensor(gen.normal(size=(6,1,3,2)),dtype=torch.float32)for _ in [0,1]]
 actual=posterior(recs,logits,ids);expected=np.zeros_like(actual);loc={f:i for i,f in enumerate(ids)}
 for l in range(6):
  grids=probabilities(recs,[z[l]for z in logits])
  for (a,b),pa in zip(grids[0]['ij'].T,grids[0]['p']):
   for (c,d),pb in zip(grids[1]['ij'].T,grids[1]['p']):
    i=loc[min(recs[0]['frame_ids'][a],recs[1]['frame_ids'][c])];j=loc[max(recs[0]['frame_ids'][b],recs[1]['frame_ids'][d])];expected[i,j]+=pa*pb/6
 assert np.allclose(actual,expected,atol=2e-15,rtol=0)
 assert np.array_equal(actual,np.mean([mass_table(probabilities(recs,[z[l]for z in logits]))[1]for l in range(6)],axis=0))

def test_native_teacher_geometry_pipeline_and_final_readout(monkeypatch):
 import stvg.predictor as mod
 import stvg.temporal_hidden as export
 T=9;ids=list(range(T));frames=np.zeros((T,4,5,3),dtype=np.uint8);boxes=torch.tensor([[.5,.5,.2,.3]]*T)
 class Head(nn.Linear):
  def cuda(self):return self
 head=Head(256,2,bias=False).eval().requires_grad_(False);head.weight.zero_()
 source=SimpleNamespace(cuda=lambda:source,cpu=lambda:source,temp_embed=head,ground_decoder=SimpleNamespace(time_decoder=nn.Identity(),decoder=nn.Identity()))
 hidden=[torch.zeros(6,1,len(ids[i::2]),256)for i in [0,1]];saved=[head(h)[-1]for h in hidden];recs=[dict(offset=i,frame_ids=ids[i::2])for i in [0,1]];base=SimpleNamespace(zero=dict(logits=saved,boxes=boxes),spatial_inputs=[{},{}]);calls=[]
 monkeypatch.setattr(mod,'frozen_native_forward',lambda *a:(None,recs,base));monkeypatch.setattr(torch.cuda,'synchronize',lambda:None);monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
 monkeypatch.setattr(mod,'observation_positions',lambda *a:list(range(8)))
 monkeypatch.setattr(mod,'initialize',lambda source,*a:(source.copy(),dict(delta=[0.]*4)))
 monkeypatch.setattr(export,'cache_temporal',lambda *a:dict(offsets=[dict(hidden=h,native_logits=z)for h,z in zip(hidden,saved)]))
 monkeypatch.setattr(mod.SourceEvidence,'from_capture',lambda *a,**kw:'evidence')
 class Geometry:
  def for_interval(self,interval):calls.append(interval);return self
 monkeypatch.setattr(mod,'build_geometry',lambda *a:Geometry());monkeypatch.setattr(mod,'evidence_dp',lambda *a:dict(final=boxes,costs=dict(updates=0)))
 obj=mod.Predictor(config=dict(dataset='hc2',temporal_mode='global6',initialization_gamma=.5,E_calibration={}),source=source)
 teacher=dict(boxes=boxes[:8],native_mask_logits=torch.ones(8,2,3),input_shape=[8,3,4,5],selected_query=1,nested_mask_all_false=True,cost={})
 obj.teacher_service=lambda *a,**kw:teacher
 def forbidden(*a,**kw):raise AssertionError('No parameter adaptation')
 monkeypatch.setattr(torch.optim,'Adam',forbidden)
 cap=obj.capture(frames,ids,dict(caption='synthetic object'),parses={'subject':'object'});result=obj.finish(cap)
 expected=posterior(recs,[head(h)for h in hidden],ids)
 assert np.array_equal(result['posterior'],expected)and result['interval']==final_mbr(ids,expected)['interval']
 assert len(calls)==1 and result['temporal_mode']=='global6'and result['cost']['actual_updates']==result['cost']['extra_decoder_calls']==0
 assert cap['cost']['Source448_calls']==0 and result['GT_read']is False

def test_main_core_has_no_experiment_or_optimizer_dependency():
 for f in (ROOT/'stvg').glob('*.py'):
  for n in ast.walk(ast.parse(f.read_text())):
   if isinstance(n,ast.ImportFrom):assert not(n.module or'').startswith(('scripts','methods','vg_tta'))
   if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute):assert n.func.attr not in ['backward','Adam','SGD']
