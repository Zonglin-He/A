"""Cached teacher-endpoint control; changes the target, not the admission set."""
import numpy as np
from vg_tta.stvg_error_state_correction_v1 import encode,decode
ARMS=['E_MBR','Residual_local','Endpoint_local']

def construct(old,local):
    e=np.asarray(old['outputs']['E_MBR'],float);out=e.copy();pos=old['planned_positions'];delta=np.zeros((4,4));counts=np.zeros(4,int)
    assert np.array_equal(e,local['outputs']['E_MBR'])
    if old['active']:
        details=old['matched']['B_aligned']['details'];assert len(details)==24
        for detail in details:
            for j,i in enumerate(pos):
                v=detail['anchors'].get(i)
                if v is None:continue
                assert detail['legal']and v['accepted']==bool(v['score']>=.35)
                if v['accepted']:
                    delta[j]+=encode(np.asarray(v['box'],float)[None])[0]-encode(e[i:i+1])[0];counts[j]+=1
        delta/=24
        for j,i in enumerate(pos):
            if np.any(delta[j]!=0):out[i]=decode(encode(e[i:i+1])+delta[j])[0]
    assert np.array_equal(counts,local['accepted_counts'])
    unobs=~np.isin(np.arange(len(e)),pos);assert np.array_equal(out[unobs],e[unobs])
    return dict(outputs=dict(zip(ARMS,[e.copy(),local['outputs']['Independent4_observed'].copy(),out])),endpoint_delta=delta,accepted_counts=counts)

def properties():
    e=np.array([.4,.3,-1.,-1.5]);s=e+np.array([.2,-.1,.1,-.2]);t=s.copy()
    endpoint=(t-e)/24;residual=(t-s)/24
    assert np.array_equal(residual,np.zeros(4))and np.linalg.norm(endpoint)>0
    assert np.array_equal(np.zeros((24,4)).sum(0)/24,np.zeros(4))
    assert np.array_equal((np.tile(e,(24,1))-e).sum(0)/24,np.zeros(4))
    return dict(status='pass',no_admission_exact_noop=True,teacher_equals_E_exact_noop=True,
        identity_teacher_equals_sample_not_guaranteed_noop=True,changed_estimation_target_explicit=True,denominator=24)
