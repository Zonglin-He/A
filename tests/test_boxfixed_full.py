import json
import numpy as np
import pytest
from scripts.boxfixed_tuning.cache import binding,RGBCache
from scripts.boxfixed_tuning.full import identity,alive,controller
from scripts.boxfixed_tuning.full_score import cluster

def test_capture_binding_ignores_alias_and_location_but_not_any_model_input():
    a=dict(id='alias1',dataset='hc2',metadata=dict(video_path='/old',video_sha256='bytes',caption='caption',frame_ids=[0,3],index=2))
    b=dict(a,id='alias2',metadata=dict(a['metadata'],video_path='/new'))
    assert binding(a,'weights')==binding(b,'weights')
    assert binding(a,'weights')!=binding(a,'different_weights')
    for key,value in [('caption','other'),('frame_ids',[0,4]),('index',3)]:
        assert binding(a,'weights')!=binding(dict(a,metadata=dict(a['metadata'],**{key:value})),'weights')

def test_RGB_cache_returns_private_bytes_and_checks_content(monkeypatch):
    import scripts.boxfixed_tuning.cache as mod
    data=np.zeros((2,3,4,3),np.uint8);calls=[]
    def decode(m,d):calls.append(1);return data.copy(),[0,3]
    monkeypatch.setattr(mod,'decode',decode);monkeypatch.setattr(mod,'sha',lambda p:'bytes')
    c=RGBCache();m=dict(video_sha256='bytes',video_path='path',frame_ids=[0,3],width=4,height=3)
    a,_,r=c.get(m,'hc2');assert not r['hit'];a.fill(9)
    b,_,r=c.get(m,'hc2');assert r['hit']and not b.any()and len(calls)==1
    monkeypatch.setattr(mod,'sha',lambda p:'changed')
    with pytest.raises(AssertionError):c.get(m,'hc2')

def test_PID_identity_requires_startticks_and_live_child_blocks_resume(monkeypatch,tmp_path):
    import scripts.boxfixed_tuning.full as mod
    current=identity();assert alive(current)and not alive(dict(current,startticks=current['startticks']+1))
    monkeypatch.setattr(mod,'RUN',tmp_path);monkeypatch.setattr(mod,'validate_plan',lambda:dict(fingerprint='locked'))
    (tmp_path/'state.json').write_text(json.dumps(dict(fingerprint='locked',children=[current])))
    with pytest.raises(AssertionError,match='still alive'):controller(True)

def test_cluster_bootstrap_preserves_query_weighted_mean():
    rows=[]
    for parent,delta in [('A',1.),('A',1.),('B',0.)]:
        x=dict(v=delta,t=delta,unobserved_actual_sIoU=delta,observed_actual_sIoU=delta,boundary_MAE_seconds=delta)
        rows.append(dict(parent=parent,values=dict(Tuned=x,Source={k:0. for k in x})))
    p=cluster(rows,'Tuned','Source');assert p['v']['parents']==2 and p['v']['queries']==3
    assert p['v']['mean']==pytest.approx(200/3)
    assert p['v']['CI95']==[0.,100.]
