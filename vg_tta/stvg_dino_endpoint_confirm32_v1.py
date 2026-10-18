"""Frozen endpoint rule and center controls; no GT, optimization or dynamic loop."""
import numpy as np
from vg_tta.stvg_error_state_correction_v1 import encode,decode
ARMS=['E_MBR','Center_full_local','Samples24_endpoint','Center_p_local']

def combine(E,positions,sample_details,center_details,active=True):
    E=np.asarray(E,float);ez=encode(E);outputs={a:E.copy()for a in ARMS};delta=np.zeros((4,4));counts=np.zeros(4,int)
    center_accepted=np.zeros(4,bool)
    if active:
        assert len(sample_details)==24
        for d in sample_details:
            for j,pos in enumerate(positions):
                v=d['anchors'].get(pos)
                if v and v['accepted']:
                    assert d['legal']and v['score']>=.35
                    delta[j]+=encode(np.asarray(v['box'],float)[None])[0]-ez[pos];counts[j]+=1
        delta/=24
        for j,pos in enumerate(positions):
            if np.any(delta[j]!=0):outputs['Samples24_endpoint'][pos]=decode(ez[pos:pos+1]+delta[j])[0]
            v=center_details['anchors'].get(pos)
            if v and v['accepted']:
                assert center_details['legal']and v['score']>=.35;center_accepted[j]=True
                target=np.asarray(v['box'],float);outputs['Center_full_local'][pos]=target
                d=encode(target[None])[0]-ez[pos]
                if counts[j]>0 and np.any(d!=0):outputs['Center_p_local'][pos]=decode(ez[pos:pos+1]+d*(counts[j]/24))[0]
    unobserved=~np.isin(np.arange(len(E)),positions)
    assert all(np.array_equal(v[unobserved],E[unobserved])for v in outputs.values())
    return dict(outputs=outputs,accepted_counts=counts,center_accepted=center_accepted,p=counts/24,
                common_admitted=center_accepted&(counts>0),endpoint_delta=delta)

def properties():
    E=np.tile([.4,.5,.2,.3],(6,1));pos=[0,1,3,5]
    empty=[dict(legal=True,anchors={})for _ in range(24)];center=dict(legal=True,anchors={})
    x=combine(E,pos,empty,center);assert all(np.array_equal(v,E)for v in x['outputs'].values())
    target=np.array([.5,.5,.2,.3]);fb=dict(box=target,score=.8,accepted=True)
    center['anchors'][0]=fb
    for d in empty[:6]:d['anchors'][0]=fb
    x=combine(E,pos,empty,center);assert x['p'][0]==.25
    assert np.array_equal(x['outputs']['Center_full_local'][0],target)
    assert np.allclose(x['outputs']['Samples24_endpoint'][0],x['outputs']['Center_p_local'][0],atol=1e-15,rtol=0)
    assert np.array_equal(x['outputs']['Center_p_local'][1:],E[1:])
    rejected=dict(legal=True,anchors={0:dict(box=target,score=.1,accepted=False)})
    y=combine(E,pos,empty,rejected);assert np.array_equal(y['outputs']['Center_p_local'],E)
    return dict(status='pass',no_feedback_exact_noop=True,rejected_center_exact_E=True,
                p_same_scalar_not_matched_norm=True,mechanism_control_requires24_admissions=True,
                equal_endpoints_give_equal_p_outputs=True,unobserved_exact_E=True)
