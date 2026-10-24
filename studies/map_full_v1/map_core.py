"""Final posterior MAP followed by the unchanged interval-conditioned spatial DP."""
import sys
from pathlib import Path
R=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(R),str(R/'tools'),str(R/'studies/paper_review_suite_v1'),str(R/'studies/paper_followup_v1')]
import numpy as np
import torch
from stvg.prototype import one_step
from stvg.temporal import final_mbr
from stvg.spatial import SpatialGeometry,evidence_dp
from stvg.readout import prediction

ARMS=['Source','SourcePlusMBR','SourceJointMAP','FullMBR','FullMAPFixedBoxes','FullMAP']

def map_interval(ids,q):
    q=np.asarray(q,dtype=np.float64)
    assert q.shape==(len(ids),len(ids))and np.isfinite(q).all()and(q>=0).all()and abs(q.sum()-1)<1e-10
    a,b=np.unravel_index(int(q.argmax()),q.shape)
    assert q[a,b]>0 and a<=b
    return [int(ids[a]),int(ids[b])+1]

def derive(backbone,ds,c):
    ids=c['frame_ids'];g=SpatialGeometry.from_cached_geometry(c['geometry'])
    if backbone=='TA':
        features=c['content_features'];q0=c['native_prior']
        q1,_=one_step(features,q0)
        native=prediction(c['native_logits'],torch.as_tensor(c['Source']),c['records'],ids)['physical_interval']
        prior_mbr=c['native_interval']
        def solve(iv):return evidence_dp(g.for_interval(iv))['final'].numpy()
        source=np.asarray(c['Source'])
    else:
        from studies.cgstvg_full_v1.core import solve as cg_solve
        from cgstvg_runtime_v9 import clip_native_boxes
        profile=c['profile'];features=c['pooled_features'];q0=c['q0']
        q1,_=one_step(features,q0,kappa=profile['kappa'],beta=profile['beta'])
        native=c['native_map'];prior_mbr=c['native_mbr'];source=np.asarray(c['Source'])
        def solve(iv):return clip_native_boxes(cg_solve(g,iv,profile))
    iv_mbr=final_mbr(ids,q1)['interval'];iv_map=map_interval(ids,q1);iv_prior_map=map_interval(ids,q0)
    assert final_mbr(ids,q0)['interval']==list(prior_mbr)
    mbr=solve(iv_mbr);mapped=solve(iv_map)
    arrays=dict(Source=source.copy(),Source_interval=np.asarray(native),
        SourcePlusMBR=source.copy(),SourcePlusMBR_interval=np.asarray(prior_mbr),
        SourceJointMAP=source.copy(),SourceJointMAP_interval=np.asarray(iv_prior_map),
        FullMBR=mbr,FullMBR_interval=np.asarray(iv_mbr),
        FullMAPFixedBoxes=mbr.copy(),FullMAPFixedBoxes_interval=np.asarray(iv_map),
        FullMAP=mapped,FullMAP_interval=np.asarray(iv_map),
        frame_ids=np.asarray(ids),positions8=np.asarray(c['positions8']))
    assert np.array_equal(arrays['FullMAP_interval'],arrays['FullMAPFixedBoxes_interval'])
    assert all(np.isfinite(arrays[a]).all()for a in ARMS)
    if backbone=='CG'and ds=='hc2':
        unobserved=np.ones(len(ids),bool);unobserved[c['positions8']]=False
        assert np.array_equal(mapped[unobserved],source[unobserved])
    return arrays
