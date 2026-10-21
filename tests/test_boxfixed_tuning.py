import numpy as np
import torch
from scripts.boxfixed_tuning.core import BASE,spatial,temporal
from stvg.evidence import SourceEvidence
from stvg.spatial import build_geometry,evidence_dp
from stvg.temporal import probabilities,mass_table,final_mbr

def cap():
    T=11;H,W=4,5;N=H*W
    raw=torch.zeros(T,N,256);raw[:,torch.arange(N),torch.arange(N)]=1
    ev=SourceEvidence.from_arrays(raw,torch.ones(T,N,dtype=torch.bool),torch.arange(T)*3,
        [0,1,2,4,6,8,9,10],torch.ones(8,H,W,dtype=torch.float64))
    E=torch.tensor([.4,.5,.2,.3],dtype=torch.float64).repeat(T,1)
    native=torch.tensor([.5,.5,.4,.5],dtype=torch.float64).repeat(8,1)
    return dict(geometry=build_geometry(ev,E,native).state_dict())

def test_default_dp_is_original_and_immutable():
    c=cap();g=build_geometry.__globals__['SpatialGeometry'].__new__(build_geometry.__globals__['SpatialGeometry']);g.__dict__.update(c['geometry'])
    before={k:v.clone()for k,v in c['geometry'].items()if torch.is_tensor(v)}
    expected=evidence_dp(g.for_interval([0,31]));actual=spatial(c,[0,31],dict(BASE))
    assert torch.equal(actual['actions'],expected['actions']) and torch.equal(actual['final'],expected['final'])
    assert all(torch.equal(c['geometry'][k],v)for k,v in before.items())

def test_no_observation_reward_no_links_no_guard_keeps_observation_E():
    c=cap();out=spatial(c,[0,31],dict(BASE,observed_weight=0.,guard_weight=0.,link_weight=0.))
    assert out['actions'][c['geometry']['observed']].eq(0).all()

def test_large_geometric_protection_restores_all_E():
    c=cap();out=spatial(c,[0,31],dict(BASE,guard_weight=1e6))
    assert out['actions'].eq(0).all()

def test_temperature_one_recovers_original_and_uniform_is_invariant():
    ids=list(range(0,18,3));records=[dict(frame_ids=ids[o::2])for o in(0,1)]
    zs=[torch.zeros(6,1,3,2)for _ in records]
    c=dict(records=records,logits=zs,frame_ids=ids)
    U,q=mass_table(probabilities(records,[z[0]for z in zs]))
    old=np.mean([q]*6,axis=0);a=temporal(c,1.);b=temporal(c,2.)
    assert np.array_equal(a['q'],old) and np.array_equal(a['q'],b['q'])
    assert a['interval']==final_mbr(U,old)['interval']
