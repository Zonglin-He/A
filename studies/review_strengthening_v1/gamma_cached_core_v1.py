"""Five gamma full-chain outputs and cold sufficient-state replay, no GT/model."""
from review_common import *
import numpy as np,torch
from stvg.spatial import SpatialGeometry,evidence_dp
from stvg.initialization import initialize
from stvg.predictor import output_writeback
from stvg.representation import extract
from stvg.prototype import one_step
from stvg.current_shared_roi import map_interval

VALUES=[0,.25,.5,.75,1]
NAMES=['Gamma0','Gamma025','Gamma05','Gamma075','Gamma1']


def replay(compact):
    ids=compact['frame_ids'];out=dict(frame_ids=np.asarray(ids),positions8=np.asarray(compact['positions8']))
    for name in NAMES:
        c=compact['arms'][name];q,_=one_step(c['features'],compact['q0'],kappa=0,beta=1)
        iv=map_interval(ids,q);g=SpatialGeometry.from_cached_geometry(c['geometry'])
        out[name]=evidence_dp(g.for_interval(iv))['final'].numpy()
        out[name+'_interval']=np.asarray(iv)
        assert np.array_equal(q,c['posterior']) and np.array_equal(iv,c['interval'])
    assert all(np.isfinite(a).all() for a in out.values())
    return out


def build(c,raw,ds,current):
    # Reuse original seed evidence and rebuild every gamma-dependent box,
    # geometry, actual pretext ROI pool, prototype/posterior and final interval.
    ev=selected.spatial().raw_evidence(c,raw)
    cfg=read(R/'configs/default.json')['datasets'][ds]
    native=np.asarray(c['geometry']['native_boxes'])
    shared,_=initialize(c['Source'],c['positions8'],native,cfg['E_calibration'])
    compact=dict(frame_ids=c['frame_ids'],positions8=c['positions8'],q0=c['native_prior'],arms={},GT_read=False)
    arrays={};default=None
    for gamma,name in zip(VALUES,NAMES):
        E=output_writeback(c['Source'],shared,gamma)
        g=SpatialGeometry(ev,E,native);f,_=extract(raw,E)
        q,_=one_step(f,c['native_prior'],kappa=0,beta=1);iv=map_interval(c['frame_ids'],q)
        boxes=evidence_dp(g.for_interval(iv))['final'].numpy()
        compact['arms'][name]=dict(gamma=gamma,E=E,features=f,posterior=q,
                                  interval=np.asarray(iv),geometry=g.state_dict())
        arrays[name]=boxes;arrays[name+'_interval']=np.asarray(iv)
        if gamma==cfg['initialization_gamma']:
            assert np.array_equal(E,c['E'])
            assert np.array_equal(boxes,current['FullMAP']) and np.array_equal(iv,current['FullMAP_interval'])
            default=name
    assert default is not None
    arrays.update(frame_ids=np.asarray(c['frame_ids']),positions8=np.asarray(c['positions8']))
    again=replay(compact);assert set(arrays)==set(again) and all(np.array_equal(arrays[k],again[k]) for k in arrays)
    return arrays,compact,default
