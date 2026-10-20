"""Synthetic CPU checks for caching, resident requests and exact resume binding."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from stvg.teacher_bridge import ResidentTeacher, evidence_key
from stvg import worker


def context():
    return dict(positions=list(range(8)),frame_ids=list(range(16)),fingerprint='frozen-all-models-and-code',dataset='hc2')


def evidence():
    return dict(boxes=torch.ones(8,4),cost=dict(expert_seconds=1.,forward_seconds=.5),
                model_load_seconds=2.,peak_GPU_bytes=123,load_audit={'expert_instances':1},process_wall_seconds=3.)


def test_evidence_key_uses_actual_rgb_text_positions_models_and_grid():
    frames=np.zeros((8,2,2,3),np.uint8);cfg=context()
    expected,_=evidence_key(frames,'A person',**cfg)
    changed=frames.copy();changed[0,0,0,0]=1
    assert evidence_key(changed,'A person',**cfg)[0]!=expected
    assert evidence_key(frames,'a person',**cfg)[0]!=expected
    for name,value in [('positions',list(range(1,9))),('frame_ids',list(range(17))),('fingerprint','changed'),('dataset','vidstg')]:
        cc={**cfg,name:value}
        assert evidence_key(frames,'A person',**cc)[0]!=expected


def test_teacher_cache_exact_hit_and_tamper_rejected(tmp_path,monkeypatch):
    frames=np.zeros((8,2,2,3),np.uint8);cc=context();fp=cc.pop('fingerprint');calls=[]
    with ResidentTeacher(tmp_path/'cache',fp)as teacher:
        def infer(frames,caption):calls.append(caption);return evidence()
        monkeypatch.setattr(teacher,'_call',infer)
        a=teacher(frames,'Text',**cc);b=teacher(frames,'Text',**cc)
        assert len(calls)==1 and not a['cache_hit'] and b['cache_hit']
        assert torch.equal(a['boxes'],b['boxes'])and b['query_forwards']==0 and b['model_load_seconds']==0
        teacher(frames,'Other text',**cc);assert len(calls)==2
        payload=tmp_path/'cache'/(a['cache_key']+'.pt');payload.write_bytes(b'tamper')
        with pytest.raises(ValueError,match='binding/hash mismatch'):teacher(frames,'Text',**cc)


def record():
    return dict(id='query',dataset='hc2',key='unchanged-sampling-alias',metadata=dict(caption='A person walks.',index=0,width=2,height=2,frame_ids=list(range(8))))


class SyntheticSession:
    calls=0
    def __init__(self,*args):self.predictors={'hc2':type('P',(),{'config':{'fixed':True}})()}
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def predict(self,record):
        type(self).calls+=1
        boxes=np.ones((8,4));out={k:boxes.copy()for k in ['Source','E_Ref','Live_raw','Live']}
        out.update(fit={'positions':list(range(8))},intervals={'Native':[0,8],'Strong':[0,8]},capture_cost={},cost={})
        return out,list(range(8))


def test_worker_resume_uses_exact_request_key_and_prediction_hash(tmp_path,monkeypatch):
    SyntheticSession.calls=0
    monkeypatch.setattr(worker,'runtime_fingerprint',lambda *args:('frozen-fingerprint',{'synthetic':True}))
    monkeypatch.setattr(worker,'ResidentSession',SyntheticSession)
    manifest=tmp_path/'manifest.jsonl';manifest.write_text(json.dumps(record())+'\n');out=tmp_path/'out'
    fingerprint=worker.run(manifest,out)
    assert SyntheticSession.calls==1
    assert worker.run(manifest,out,resume_hash=fingerprint)==fingerprint and SyntheticSession.calls==1
    with pytest.raises(ValueError,match='resume hash'):worker.run(manifest,out,resume_hash='another')
    r=record();r['key']='a-different-seed-alias';assert worker.request_hash(r)!=worker.request_hash(record())
    prediction=out/'predictions/query.npz';prediction.write_bytes(b'changed')
    with pytest.raises(ValueError,match='exact run'):worker.run(manifest,out)


def test_worker_rejects_label_fields_before_fingerprinting(tmp_path,monkeypatch):
    def forbidden():pytest.fail('Must reject label-bearing manifest before model fingerprinting')
    monkeypatch.setattr(worker,'runtime_fingerprint',forbidden)
    r=record();r['metadata']['boxes']=[[0,0,1,1]]
    manifest=tmp_path/'manifest.jsonl';manifest.write_text(json.dumps(r)+'\n')
    with pytest.raises(ValueError,match='Unsupported input'):worker.run(manifest,tmp_path/'out')
